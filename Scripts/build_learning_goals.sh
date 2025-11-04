#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

IN=learning_goals.md
OUT=learning_goals.pdf

if [ ! -f "$IN" ]; then
  echo "ERROR: $IN not found at repo root."
  exit 1
fi

echo "Building $OUT from $IN ..."
# gfm reader + math (both $...$ and \( \) / \[ \ ])
pandoc "$IN" \
  --from=markdown+tex_math_dollars+tex_math_single_backslash \
  --pdf-engine=tectonic \
  -V geometry:margin=1in \
  -o "$OUT"


echo "Built $OUT"

