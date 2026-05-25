#!/usr/bin/env bash
# Render reports/manuscript/main.tex to main.pdf.
#
# Usage:
#   scripts/render_manuscript.sh          # build PDF
#   scripts/render_manuscript.sh --clean  # remove latex aux files (keeps main.pdf)
#   scripts/render_manuscript.sh --open   # build then open the PDF

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANUSCRIPT_DIR="$PROJECT_ROOT/reports/manuscript"
MAIN="main"

clean_aux() {
    cd "$MANUSCRIPT_DIR"
    rm -f "$MAIN".{aux,bbl,bcf,blg,log,out,run.xml,toc,fls,fdb_latexmk,synctex.gz}
}

case "${1:-}" in
    --clean)
        clean_aux
        echo "[render_manuscript] cleaned auxiliary files in $MANUSCRIPT_DIR"
        exit 0
        ;;
esac

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "[render_manuscript] ERROR: pdflatex not found in PATH" >&2
    exit 1
fi
if ! command -v biber >/dev/null 2>&1; then
    echo "[render_manuscript] ERROR: biber not found in PATH" >&2
    exit 1
fi

cd "$MANUSCRIPT_DIR"

echo "[render_manuscript] pdflatex pass 1"
pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex >/dev/null

echo "[render_manuscript] biber"
biber "$MAIN" >/dev/null

echo "[render_manuscript] pdflatex pass 2"
pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex >/dev/null

echo "[render_manuscript] pdflatex pass 3 (resolve cross-refs)"
pdflatex -interaction=nonstopmode -halt-on-error "$MAIN".tex >/dev/null

if grep -q "Warning: There were undefined references" "$MAIN".log 2>/dev/null; then
    echo "[render_manuscript] WARNING: undefined references remain in $MAIN.log" >&2
fi
if grep -q "Warning: There were multiply-defined labels" "$MAIN".log 2>/dev/null; then
    echo "[render_manuscript] WARNING: multiply-defined labels in $MAIN.log" >&2
fi

PDF_PATH="$MANUSCRIPT_DIR/$MAIN.pdf"
echo "[render_manuscript] wrote $PDF_PATH"

if [[ "${1:-}" == "--open" ]]; then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$PDF_PATH" >/dev/null 2>&1 &
    elif command -v open >/dev/null 2>&1; then
        open "$PDF_PATH"
    fi
fi
