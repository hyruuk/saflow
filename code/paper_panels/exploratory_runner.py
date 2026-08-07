"""Run clearly separated corrected-paper exploratory sidekick analyses."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from code.paper_panels.contracts import PANEL1_FEATURES
from code.utils.config import load_config


def run_exploratory(args: argparse.Namespace) -> Path:
    """Run sensor diagnostics, correct/lapse variants, and complexity sidekicks."""
    config = load_config(args.config)
    invoke = str(Path(config["paths"]["venv"]) / "bin" / "invoke")
    feature_value = " ".join(PANEL1_FEATURES) + " complexity"
    commands = []
    for space, trial_types in (
        ("sensor", ("alltrials", "correct", "lapse")),
        ("schaefer_400", ("correct", "lapse")),
    ):
        for trial_type in trial_types:
            commands.extend(
                [
                    [
                        invoke,
                        "analysis.stats",
                        f"--features={feature_value}",
                        f"--space={space}",
                        f"--trial-type={trial_type}",
                        "--correction=fdr",
                        "--analysis-level=average",
                    ],
                    [
                        invoke,
                        "analysis.classify",
                        f"--features={feature_value}",
                        f"--space={space}",
                        f"--trial-type={trial_type}",
                        "--clf=logistic",
                        "--cv=logo",
                        "--analysis-level=epoch",
                        f"--n-permutations={args.n_permutations}",
                        f"--n-jobs={args.jobs}",
                    ],
                ]
            )
    commands.extend(
        [
            [
                invoke,
                "analysis.networks.classify",
                "--space=schaefer_400",
                "--scope=all",
                "--trial-type=all",
                "--clf=logistic",
                "--cv=logo",
                f"--n-permutations={args.n_permutations}",
            ],
            [
                invoke,
                "analysis.networks.coherence",
                "--space=schaefer_400",
                "--trial-type=all",
            ],
        ]
    )
    for command in commands:
        subprocess.run(command, check=True)
    root = (
        Path(args.analysis_root)
        if args.analysis_root
        else Path(config["paths"]["data_root"])
        / "processed"
        / config.get("paper_panels", {}).get("processed_directory", "paper_panels")
    )
    output = root / args.analysis_id / "exploratory" / "sidekick_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "analysis_id": args.analysis_id,
                "status": "complete",
                "classification": "exploratory only",
                "paper_masks_replaced": False,
                "features": [*PANEL1_FEATURES, "complexity shortcut"],
                "spaces": ["sensor", "schaefer_400"],
                "trial_types": ["alltrials sensor diagnostic", "correct", "lapse"],
                "network_restricted_decoding": "exploratory only",
                "legacy_contrast_pattern_coherence": (
                    "exploratory only; not described as co-activation"
                ),
                "commands": commands,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main() -> None:
    """Run exploratory analyses from scheduler arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--n-permutations", type=int, default=1_000)
    parser.add_argument("--jobs", type=int, default=4)
    run_exploratory(parser.parse_args())


if __name__ == "__main__":
    main()
