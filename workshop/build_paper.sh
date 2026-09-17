#!/usr/bin/env bash
# Build workshop/mechanism_paper.pdf with tectonic (a self-contained LaTeX
# engine; no TeX Live needed). Downloads the binary on first use.
#   ./workshop/build_paper.sh            # writes workshop/mechanism_paper.pdf
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
if ! command -v "$TECTONIC" >/dev/null 2>&1; then
  echo "tectonic not found at $TECTONIC; fetching the latest release binary" >&2
  mkdir -p "$(dirname "$TECTONIC")"
  url=$(curl -s https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest \
        | grep -oE 'https://[^"]*x86_64-unknown-linux-musl\.tar\.gz' | head -1)
  curl -sL "$url" | tar xz -C "$(dirname "$TECTONIC")" tectonic
fi
"$TECTONIC" --keep-logs mechanism_paper.tex
echo "wrote $(pwd)/mechanism_paper.pdf"
