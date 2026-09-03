#!/usr/bin/env bash
set -euo pipefail
readonly REPO="$1" OUT="$2" EXPECTED_SHA="$3"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
"/data/home/scvi806/.conda/envs/Filament_python/bin/python" "$REPO/Filament_python/tools/preflight_hr4e2c_real.py" --sources "$REPO/Filament_python/tools/hr4e2c_real_sources.json" --out "$OUT" >/dev/null
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","preflight":"%s"}\n' "$OUT"
