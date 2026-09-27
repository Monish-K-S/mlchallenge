#!/usr/bin/env bash
# Build <team>_submission.zip with the exact layout the challenge expects.
#
# Layout produced:
#   output/matching_results.tsv
#   output/candidate_pairs.tsv
#   code/business_entity_resolution/{src/,README.md,APPROACH.md,requirements.txt}
#   Documentation_template.md
#
# Usage (from student_resource/):
#   TEAM_NAME=yourteam code/business_entity_resolution/scripts/package_submission.sh
set -euo pipefail

TEAM_NAME="${TEAM_NAME:-team}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # -> student_resource/
PKG="$(cd "$(dirname "$0")/.." && pwd)"          # -> business_entity_resolution/
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging from $ROOT"
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp "$ROOT/output/matching_results.tsv" "$STAGE/output/"
cp "$ROOT/output/candidate_pairs.tsv" "$STAGE/output/"
cp -R "$PKG/src" "$STAGE/code/business_entity_resolution/"
cp "$PKG/README.md" "$PKG/APPROACH.md" "$PKG/requirements.txt" "$STAGE/code/business_entity_resolution/"
rm -rf "$STAGE/code/business_entity_resolution/src/__pycache__"
cp "$ROOT/Documentation_template.md" "$STAGE/"

OUT_ZIP="$(pwd)/${TEAM_NAME}_submission.zip"
( cd "$STAGE" && zip -qr "$OUT_ZIP" . )
echo "Wrote $OUT_ZIP"
unzip -l "$OUT_ZIP" | tail -20