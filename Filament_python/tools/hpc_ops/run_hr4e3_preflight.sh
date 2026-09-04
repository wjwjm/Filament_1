#!/usr/bin/env bash
set -euo pipefail
readonly REPO="$1" OUT="$2" EXPECTED_SHA="$3"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly PROVENANCE="${OUT%.json}_provenance_v2.json"
readonly SOURCE_MANIFEST="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc/post_reference/post_reference_manifest.json"
readonly SOURCE_ROOT="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc/post_reference"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$OUT" && test ! -e "$PROVENANCE"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" create --repo "$REPO" --output "$PROVENANCE" \
    --tracked Filament_python/KHz_filament/hr4.py Filament_python/KHz_filament/hr4e_domain.py Filament_python/KHz_filament/hr4e_real_spatial.py Filament_python/KHz_filament/hr4e_spatial.py Filament_python/KHz_filament/hr4e_timestep.py Filament_python/tools/hr4e2c_real_sources.json Filament_python/tools/preflight_hr4e3_domain.py Filament_python/tools/run_hr4e3_domain_case.py Filament_python/tools/summarize_hr4e3_domain.py Filament_python/tools/hr4e3_domain_case.sbatch Filament_python/tools/hpc_ops/run_hr4e3_preflight.sh Filament_python/tools/hpc_ops/submit_hr4e3_synthetic.sh Filament_python/tools/hpc_ops/submit_hr4e3_real.sh \
    --external "$SOURCE_MANIFEST" "$SOURCE_ROOT/screen_front_delta_n.npy" "$SOURCE_ROOT/screen_peak_delta_n.npy" "$SOURCE_ROOT/screen_rear_delta_n.npy" >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$REPO" --manifest "$PROVENANCE" --require-hash-scope >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/preflight_hr4e3_domain.py" --sources "$REPO/Filament_python/tools/hr4e2c_real_sources.json" --out "$OUT" --require-cupy >/dev/null
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","preflight":"%s","provenance":"%s"}\n' "$OUT" "$PROVENANCE"
