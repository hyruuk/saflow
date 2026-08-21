"""Tests for the sweep's shard ordering and its materialized feature tensor.

Both guard the failure mode of 2026-08-20, when a 40-task sweep array spent its
whole wall time inside the loader and every shard was killed before it wrote a
single cell.
"""

import numpy as np
import pytest

from code.classification import run_classification as RC
from code.classification import run_multifeature_sweep as S


def test_features_sharing_a_file_are_loaded_in_one_walk(monkeypatch):
    """The eight psd_* bands must cost one file walk, not eight.

    Each welch npz holds a (n_windows, n_spatial, n_freqs) cube — ~1.9 GB
    uncompressed per run file — and one band is a slice of it. Loading the
    bands one at a time decompressed that cube eight times per family and made
    a 23-feature stack an ~11 h job on /scratch.
    """
    calls = []
    n_trials, n_spatial = 12, 5

    def _fake_group(*, folder_prefix, file_suffix, sub_keys, **kwargs):
        calls.append((folder_prefix, tuple(sub_keys)))
        X = {k: np.full((n_trials, n_spatial), abs(hash(k)) % 97, dtype=float)
             for k in sub_keys}
        y = np.array([0] * 6 + [1] * 6)
        groups = np.repeat(np.arange(2), 6)
        return X, y, groups, {"spatial_names": [f"p-{i}" for i in range(n_spatial)]}

    monkeypatch.setattr(RC, "_load_feature_group", _fake_group)
    features = ["fooof_exponent", "fooof_offset",
                "psd_alpha", "psd_theta", "psd_gamma1",
                "psd_corrected_alpha", "psd_corrected_theta",
                "complexity_lzc_median"]
    X, y, groups, meta = RC.load_combined_features(
        features=features, space="schaefer_400", inout_bounds=(25, 75),
        config={}, n_events_window=8,
    )

    assert len(calls) == 4, f"one walk per source file, got {calls}"
    assert dict(calls) == {
        "fooof": ("exponent", "offset"),
        "welch_psds": ("alpha", "theta", "gamma1"),
        "welch_psds_corrected": ("alpha", "theta"),
        "complexity": ("lzc_median",),
    }
    # Stacking order must still follow the caller's feature order, not the
    # order the groups happened to be walked in.
    assert X.shape == (n_trials, n_spatial, len(features))
    assert meta["features"] == features
    for i, feat in enumerate(features):
        expected = abs(hash(RC.parse_feature(feat, n_events_window=8)[2])) % 97
        assert np.all(X[:, :, i] == expected), feat


def _grid(bounds=((25, 75),)):
    return [
        (b, n, f, r, e)
        for b in bounds
        for n in S.DEFAULT_NORMALIZATIONS
        for f in S.DEFAULT_FEATURE_SETS
        for r in S.DEFAULT_REDUCTIONS
        for e in S.DEFAULT_ESTIMATORS
    ]


def test_shard_cells_partitions_the_grid():
    cells = _grid()
    n_shards = 40
    seen = [c for k in range(n_shards) for c in S.shard_cells(cells, n_shards, k)]
    assert sorted(seen, key=cells.index) == cells
    assert len(seen) == len(set(seen)) == len(cells)


def test_shard_cells_keeps_cheapest_first_order():
    """Every shard must start on cheap reductions, not on ``flat``/``rank``.

    Sorting a shard's slice by the cell keys orders reduction and normalization
    alphabetically, which front-loads the two most expensive levels
    (``flat``, ``rank``) in every shard. A wall-time kill then costs the whole
    grid instead of its tail.
    """
    cells = _grid()
    order = {c: i for i, c in enumerate(cells)}
    for k in (0, 7, 39):
        mine = S.shard_cells(cells, 40, k)
        assert [order[c] for c in mine] == sorted(order[c] for c in mine)
        first_norm, first_red = mine[0][1], mine[0][3]
        assert first_norm == S.DEFAULT_NORMALIZATIONS[0]
        assert first_red != "flat"
        assert S.DEFAULT_REDUCTIONS.index(first_red) < S.DEFAULT_REDUCTIONS.index("flat")


@pytest.fixture
def fake_features(monkeypatch):
    """Stand in for the per-subject feature walk, counting how often it runs."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6, 3))
    y = np.array([0] * 20 + [1] * 20)
    groups = np.repeat(np.arange(4), 10)
    calls = {"n": 0}

    def _fake_load(**kwargs):
        calls["n"] += 1
        meta = {"spatial_names": [f"p-{i}" for i in range(X.shape[1])]}
        return X, y, groups, meta

    monkeypatch.setattr(S, "load_combined_features", _fake_load)
    return X, y, groups, calls


def _cache(tmp_path, features, **kw):
    return S.DataCache(
        features=features, space="test_space", config={},
        trial_type="alltrials", n_events_window=8, inout_selection="strict",
        keep_bad_trials=False, seed=42, cache_dir=tmp_path, **kw
    )


def test_tensor_cache_roundtrip(tmp_path, fake_features):
    X, y, groups, calls = fake_features
    features = ["a", "b", "c"]

    first = _cache(tmp_path, features)
    first.load((25, 75))
    assert calls["n"] == 1

    second = _cache(tmp_path, features)
    second.load((25, 75))
    assert calls["n"] == 1, "second load must come from the tensor cache"
    # The tensor is deliberately held in float32 (see TENSOR_DTYPE): the round
    # trip is exact to that precision, not to the float64 the fake loader
    # returns.
    assert first.X.dtype == S.TENSOR_DTYPE
    assert second.X.dtype == S.TENSOR_DTYPE
    np.testing.assert_allclose(second.X, X, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(second.X, first.X)
    np.testing.assert_array_equal(second.y, y)
    np.testing.assert_array_equal(second.groups, groups)
    assert second.spatial_names == first.spatial_names


def test_legacy_float64_cache_is_downcast_on_read(tmp_path, fake_features):
    """A cache file written before the float32 switch must not reinflate X.

    Old caches on /scratch are float64. Reading one back as float64 would put
    the shard right back at the memory ceiling that killed the 2026-08-20 run,
    so the read casts.
    """
    X, y, groups, _ = fake_features
    features = ["a", "b", "c"]

    cache = _cache(tmp_path, features)
    cache.load((25, 75))
    x_path, _ = S._cache_files(cache._cache_stem((25, 75)))
    np.save(x_path, np.asarray(X, dtype=np.float64))   # simulate a legacy file
    assert np.load(x_path).dtype == np.float64

    reloaded = _cache(tmp_path, features)
    reloaded.load((25, 75))
    assert reloaded.X.dtype == S.TENSOR_DTYPE
    np.testing.assert_allclose(reloaded.X, X, rtol=1e-6, atol=0)


def test_tensor_cache_is_keyed_on_the_feature_list(tmp_path, fake_features):
    """A cache written for one feature list must never be read back for another."""
    _, _, _, calls = fake_features
    _cache(tmp_path, ["a", "b", "c"]).load((25, 75))
    assert calls["n"] == 1

    # Same length, different features: the stored names must force a reload.
    _cache(tmp_path, ["a", "b", "d"]).load((25, 75))
    assert calls["n"] == 2


def test_refresh_cache_reloads_and_overwrites(tmp_path, fake_features):
    _, _, _, calls = fake_features
    features = ["a", "b", "c"]
    _cache(tmp_path, features).load((25, 75))
    _cache(tmp_path, features, refresh_cache=True).load((25, 75))
    assert calls["n"] == 2
    # The refreshed cache is readable again without another walk.
    _cache(tmp_path, features).load((25, 75))
    assert calls["n"] == 2


def test_unreadable_cache_falls_back_to_the_loader(tmp_path, fake_features):
    _, _, _, calls = fake_features
    features = ["a", "b", "c"]
    cache = _cache(tmp_path, features)
    cache.load((25, 75))
    stem = cache._cache_stem((25, 75))
    S._cache_files(stem)[0].write_bytes(b"not an npy")

    _cache(tmp_path, features).load((25, 75))
    assert calls["n"] == 2


def test_only_one_normalization_is_resident(tmp_path, fake_features):
    """Materializing a second normalization must free the first.

    Holding every mode at once cost a full extra copy of X per mode. On
    2026-08-20 that killed 28 of 40 shards, each at the exact cell where
    `rank` was built on top of a still-live `zscore`.
    """
    _, _, _, _ = fake_features
    cache = _cache(tmp_path, ["a", "b", "c"])
    cache.load((25, 75))

    zscored = cache.normalized("zscore")
    assert list(cache._norm_cache) == ["zscore"]
    ranked = cache.normalized("rank")
    assert list(cache._norm_cache) == ["rank"], "zscore must have been evicted"

    # Both are real normalizations of the same tensor, not views of it.
    assert zscored.dtype == S.TENSOR_DTYPE
    assert ranked.dtype == S.TENSOR_DTYPE
    assert not np.shares_memory(ranked, cache.X)


def test_normalizations_are_visited_in_contiguous_blocks():
    """Eviction is only free because a shard sees each mode once.

    ``enumerate_cells`` puts normalization outermost after the bounds and
    ``shard_cells`` preserves that order, so every shard runs all of one mode
    before starting the next. If that ever stops holding, eviction would
    rebuild `rank` — minutes of work — many times per shard.
    """
    import types
    args = types.SimpleNamespace(
        inout_bounds=[(25, 75)], normalizations=["zscore", "rank"],
        feature_sets=["all", "psds"], reductions=["yeo7-mean", "flat"],
        estimators=["logistic:C=1", "lda:shrinkage=auto"])
    cells = S.enumerate_cells(args)
    for idx in range(4):
        norms = [c[1] for c in S.shard_cells(cells, 4, idx)]
        # collapse runs of equal values; one run per distinct mode is the goal
        runs = [n for i, n in enumerate(norms) if i == 0 or n != norms[i - 1]]
        assert len(runs) == len(set(runs)), f"shard {idx} interleaves: {norms}"


def _write_shard_csv(out_base, idx, n_shards, cells):
    import csv
    path = out_base.with_name(out_base.name + f"_sweep_shard-{idx}of{n_shards}.csv")
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=S.SWEEP_FIELDS)
        w.writeheader()
        for bounds, norm, fset, red, est in cells:
            w.writerow({
                "config_id": S._config_id(bounds, norm, fset, red, est),
                "inout": S.inout_bounds_to_string(bounds), "normalization": norm,
                "feature_set": fset, "reduction": red, "estimator": est,
                "mean_auc": 0.55, "wilcoxon_p": 0.01,
            })
    return path


@pytest.fixture
def sweep_grid():
    import types
    args = types.SimpleNamespace(
        inout_bounds=[(25, 75)], normalizations=["zscore", "rank"],
        feature_sets=["all", "psds"], reductions=["yeo7-mean", "flat"],
        estimators=["logistic:C=1", "lda:shrinkage=auto"])
    return S.enumerate_cells(args)


def test_merge_rejects_a_truncated_grid(tmp_path, sweep_grid):
    """A killed shard still leaves a CSV, so merge must count cells, not files.

    On 2026-08-20 merge logged "Merging 40/40 shard CSV(s)" and picked a winner
    from 424 of 700 cells, two thirds of the `rank` arm missing, and confirm
    and importance then ran to exit 0 on it.
    """
    out_base = tmp_path / "space-x_type-alltrials_w8_sweep"
    survivors = ([c for c in sweep_grid if c[1] == "zscore"]
                 + [c for c in sweep_grid if c[1] == "rank"][:2])
    _write_shard_csv(out_base, 0, 2, survivors[::2])
    _write_shard_csv(out_base, 1, 2, survivors[1::2])

    with pytest.raises(SystemExit, match="of the sweep grid completed"):
        S.stage_merge(out_base, n_shards=2, expected_cells=sweep_grid,
                      min_grid_fraction=0.9)

    # The partial merge is still written, so the run can be inspected.
    merged = out_base.with_name(out_base.name + "_sweep.csv")
    assert merged.exists()


def test_merge_accepts_a_complete_grid(tmp_path, sweep_grid):
    out_base = tmp_path / "space-x_type-alltrials_w8_sweep"
    _write_shard_csv(out_base, 0, 2, sweep_grid[::2])
    _write_shard_csv(out_base, 1, 2, sweep_grid[1::2])
    csv_path = S.stage_merge(out_base, n_shards=2, expected_cells=sweep_grid,
                             min_grid_fraction=0.9)
    assert csv_path.exists()


def test_merge_gate_is_opt_out(tmp_path, sweep_grid):
    """--min-grid-fraction 0 must let a deliberately partial grid through."""
    out_base = tmp_path / "space-x_type-alltrials_w8_sweep"
    _write_shard_csv(out_base, 0, 2, sweep_grid[:3])
    S.stage_merge(out_base, n_shards=2, expected_cells=sweep_grid,
                  min_grid_fraction=0.0)
