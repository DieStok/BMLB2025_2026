# #!/usr/bin/env bash
# set -euo pipefail

# if [ "$#" -ne 2 ]; then
#   echo "Usage: $0 <input.md> <output.pdf>"
#   exit 1
# fi

# IN="$1"
# OUT="$2"

# REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# cd "$REPO_ROOT"

# if [ ! -f "$IN" ]; then
#   echo "ERROR: $IN not found at repo root."
#   exit 1
# fi

# echo "Building $OUT from $IN ..."
# pandoc "$IN" \
#   --from=markdown+tex_math_dollars+tex_math_single_backslash \
#   --pdf-engine=tectonic \
#   -V geometry:margin=1in \
#   -o "$OUT"

# echo "Built $OUT"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input.md> <output.pdf>"
  exit 1
fi

IN="$1"
OUT="$2"

IN_DIR="$(cd "$(dirname "$IN")" && pwd)"
IN_BASE="$(basename "$IN")"
OUT_BASE="$(basename "$OUT")"

cd "$IN_DIR"

if [ ! -f "$IN_BASE" ]; then
  echo "ERROR: $IN_BASE not found in $IN_DIR."
  exit 1
fi

echo "Building \"$OUT_BASE\" from \"$IN_BASE\" in $IN_DIR ..."
pandoc "$IN_BASE" \
  --from=markdown+tex_math_dollars+tex_math_single_backslash \
  --pdf-engine=tectonic \
  -V geometry:margin=1in \
  -o "$OUT_BASE"

echo "Built \"$OUT_BASE\" in $IN_DIR"