#!/usr/bin/env bash
# Run the scalar-only HR-2E error-localization analysis on completed NPZs.
set -euo pipefail

REPO=""
RUN_ROOT=""
OUTPUT_DIR=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) exit 64 ;;
    esac
done

[[ -n "$REPO" && -n "$RUN_ROOT" && -n "$OUTPUT_DIR" ]] || exit 64
[[ "$REPO" == /data/run01/scvi806/* && -d "$REPO" && ! -L "$REPO" ]] || exit 65
[[ "$RUN_ROOT" == /data/run01/scvi806/* && -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || exit 65
[[ "$OUTPUT_DIR" == "$RUN_ROOT"/* ]] || exit 65

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export PYTHONPATH="$REPO/Filament_python"

python "$REPO/Filament_python/tools/hr2e_error_localization.py" \
    --coarse "$RUN_ROOT/hr2e_120fs_coarse/hr2e_120fs_coarse.npz" \
    --candidate "$RUN_ROOT/hr2e_120fs_candidate/hr2e_120fs_candidate.npz" \
    --fine "$RUN_ROOT/hr2e_120fs_fine/hr2e_120fs_fine.npz" \
    --output-dir "$OUTPUT_DIR"

for name in error_localization_summary.json error_localization_segments.csv cumulative_delta_energy_candidate_vs_fine.png; do
    [[ -s "$OUTPUT_DIR/$name" ]] || exit 74
done
