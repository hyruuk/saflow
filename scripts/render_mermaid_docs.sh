#!/bin/bash
# ==============================================================================
# Render every mermaid code block under docs/ into PNG files.
#
# Output layout:
#   docs/<name>.md  (with N mermaid blocks)
#   -> docs/images/<name>-1.png, docs/images/<name>-2.png, ...
#
# Usage:
#   scripts/render_mermaid_docs.sh                # render everything
#   scripts/render_mermaid_docs.sh --check        # exit non-zero if any PNG
#                                                 # is missing or out-of-date
#                                                 # (intended for CI)
#   scripts/render_mermaid_docs.sh docs/foo.md    # render just one file
#
# Requires `@mermaid-js/mermaid-cli` (mmdc). Falls back to `npx` if mmdc
# is not on PATH. Set MMDC_BIN to override (e.g. for offline runs).
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
IMAGES_DIR="$DOCS_DIR/images"

CHECK_MODE=false
TARGETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_MODE=true; shift ;;
        -h|--help)
            sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) TARGETS+=("$1"); shift ;;
    esac
done

# Resolve mmdc binary.
if [[ -n "${MMDC_BIN:-}" ]]; then
    MMDC=("$MMDC_BIN")
elif command -v mmdc >/dev/null 2>&1; then
    MMDC=("mmdc")
elif command -v npx >/dev/null 2>&1; then
    MMDC=(npx -y -p @mermaid-js/mermaid-cli mmdc)
else
    echo "ERROR: neither 'mmdc' nor 'npx' found on PATH." >&2
    echo "  Install: npm install -g @mermaid-js/mermaid-cli" >&2
    echo "  Or set MMDC_BIN=/path/to/mmdc" >&2
    exit 2
fi

mkdir -p "$IMAGES_DIR"

# Default target set: every .md under docs/ that contains a ```mermaid block.
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    while IFS= read -r f; do
        TARGETS+=("$f")
    done < <(grep -rl '^```mermaid' "$DOCS_DIR" --include='*.md' | sort)
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "No markdown files with mermaid blocks found under $DOCS_DIR"
    exit 0
fi

# Count mermaid blocks in a .md file.
count_blocks() {
    grep -c '^```mermaid' "$1" 2>/dev/null || true
}

# Newest mtime among a list of PNGs (or 0 if none exist).
newest_mtime() {
    local newest=0
    for p in "$@"; do
        [[ -f "$p" ]] || { echo 0; return; }
        local m
        m=$(stat -c %Y "$p" 2>/dev/null || stat -f %m "$p" 2>/dev/null || echo 0)
        (( m > newest )) && newest=$m
    done
    echo "$newest"
}

EXIT=0
RENDERED=0
SKIPPED=0
STALE=()

for md in "${TARGETS[@]}"; do
    [[ -f "$md" ]] || { echo "SKIP: $md (not found)"; continue; }
    n=$(count_blocks "$md")
    if [[ "$n" -eq 0 ]]; then
        echo "SKIP: $md (no mermaid blocks)"
        continue
    fi

    stem=$(basename "$md" .md)
    # Expected output paths: <stem>-1.png ... <stem>-N.png
    expected=()
    for ((i=1; i<=n; i++)); do
        expected+=("$IMAGES_DIR/${stem}-${i}.png")
    done

    md_mtime=$(stat -c %Y "$md" 2>/dev/null || stat -f %m "$md")
    out_mtime=$(newest_mtime "${expected[@]}")

    if [[ "$CHECK_MODE" == true ]]; then
        for p in "${expected[@]}"; do
            if [[ ! -f "$p" ]] || [[ "$md_mtime" -gt "$out_mtime" ]]; then
                STALE+=("$p")
                EXIT=1
            fi
        done
        continue
    fi

    # Skip if every expected PNG exists and is newer than the source.
    fresh=true
    for p in "${expected[@]}"; do
        if [[ ! -f "$p" ]] || [[ "$md_mtime" -gt "$out_mtime" ]]; then
            fresh=false; break
        fi
    done
    if [[ "$fresh" == true ]]; then
        echo "FRESH: $md ($n block(s) up-to-date)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "RENDER: $md -> $IMAGES_DIR/${stem}-*.png ($n block(s))"
    # mmdc writes <out_stem>-1.png ... <out_stem>-N.png when input is .md.
    # -s 3 = 3x scale (legible text on wide LR flowcharts).
    "${MMDC[@]}" -i "$md" -o "$IMAGES_DIR/${stem}.png" \
        -b transparent -t neutral -s 3
    RENDERED=$((RENDERED + 1))
done

if [[ "$CHECK_MODE" == true ]]; then
    if [[ ${#STALE[@]} -gt 0 ]]; then
        echo
        echo "STALE OR MISSING (${#STALE[@]}):"
        for p in "${STALE[@]}"; do echo "  $p"; done
        echo
        echo "Run: scripts/render_mermaid_docs.sh"
        exit 1
    fi
    echo "All mermaid PNGs are up to date."
    exit 0
fi

echo
echo "Done: rendered $RENDERED file(s), $SKIPPED unchanged."
