#!/usr/bin/env bash
# Run the repository HR-2E conservative comparison on completed remote NPZs.

set -euo pipefail

REPO=""
RUN_ROOT=""
PULSE=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --pulse) PULSE="$2"; shift 2 ;;
        *) echo "unknown argument" >&2; exit 64 ;;
    esac
done
[[ -n "$REPO" && -n "$RUN_ROOT" && ( "$PULSE" == 40fs || "$PULSE" == 120fs ) ]] || exit 64
[[ "$RUN_ROOT" == /data/run01/scvi806/* && -d "$RUN_ROOT" ]] || exit 65

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export PYTHONPATH="$REPO/Filament_python"
candidate="$RUN_ROOT/hr2e_${PULSE}_candidate/hr2e_${PULSE}_candidate.npz"
candidate_metadata="$RUN_ROOT/hr2e_${PULSE}_candidate/hr2e_${PULSE}_candidate_job_metadata.json"
fine="$RUN_ROOT/hr2e_${PULSE}_fine/hr2e_${PULSE}_fine.npz"
fine_metadata="$RUN_ROOT/hr2e_${PULSE}_fine/hr2e_${PULSE}_fine_job_metadata.json"
output_dir="$RUN_ROOT/analysis_${PULSE}"
test -f "$candidate"
test -f "$candidate_metadata"
test -f "$fine"
test -f "$fine_metadata"
manifest="$REPO/Filament_python/results/hr2e_schedule_convergence/stage1_preflight/hr2e_stage1_preflight_manifest.json"
test -f "$manifest"
execution_manifest="$RUN_ROOT/execution_manifest.json"
test -f "$execution_manifest"
coarse="$RUN_ROOT/hr2e_${PULSE}_coarse/hr2e_${PULSE}_coarse.npz"
coarse_metadata="$RUN_ROOT/hr2e_${PULSE}_coarse/hr2e_${PULSE}_coarse_job_metadata.json"
if [[ -f "$coarse" ]]; then
    test -f "$coarse_metadata"
    python "$REPO/Filament_python/tools/hr2e_schedule_convergence.py" compare \
        --coarse "$coarse" --coarse-metadata "$coarse_metadata" \
        --candidate "$candidate" --candidate-metadata "$candidate_metadata" \
        --fine "$fine" --fine-metadata "$fine_metadata" \
        --manifest "$manifest" --execution-manifest "$execution_manifest" --output-dir "$output_dir"
else
    python "$REPO/Filament_python/tools/hr2e_schedule_convergence.py" compare \
        --candidate "$candidate" --candidate-metadata "$candidate_metadata" \
        --fine "$fine" --fine-metadata "$fine_metadata" \
        --manifest "$manifest" --execution-manifest "$execution_manifest" --output-dir "$output_dir"
fi
