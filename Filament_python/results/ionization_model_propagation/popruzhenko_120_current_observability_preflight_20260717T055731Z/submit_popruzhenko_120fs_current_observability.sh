#!/bin/bash
#SBATCH -J pop120obs
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH -o /data/run01/scvi806/user_Wangjimin/Filament_1/Filament_python/outputs/ionization_model_propagation/popruzhenko_120_current_observability_20260717T055731Z/logs/pop120obs-%j.out
#SBATCH -e /data/run01/scvi806/user_Wangjimin/Filament_1/Filament_python/outputs/ionization_model_propagation/popruzhenko_120_current_observability_20260717T055731Z/logs/pop120obs-%j.err
set -euo pipefail
RUN_ROOT="/data/run01/scvi806/user_Wangjimin/Filament_1/Filament_python/outputs/ionization_model_propagation/popruzhenko_120_current_observability_20260717T055731Z"
CODE_DIR="$RUN_ROOT/source/Filament_python"
CFG="$RUN_ROOT/configs/120fs_popruzhenko_full_model_current_observability.json"
OUT="$RUN_ROOT/cases/120fs_popruzhenko_full_model_current_observability/result.npz"
ANALYSIS="$RUN_ROOT/analysis"
META="$RUN_ROOT/cases/120fs_popruzhenko_full_model_current_observability/run_metadata.json"
mkdir -p "$(dirname "$OUT")" "$RUN_ROOT/logs" "$RUN_ROOT/cases/120fs_popruzhenko_full_model_current_observability/figures" "$ANALYSIS"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export CUDA_DEVICE_ORDER=PCI_BUS_ID UPPE_USE_GPU=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
CODE_SHA="$(cat "$RUN_ROOT/EXECUTION_GIT_SHA")"
CFG_SHA="$(sha256sum "$CFG" | awk '{print $1}')"
write_meta() {
  STATUS="$1" RC="$2" CODE_SHA="$CODE_SHA" CFG_SHA="$CFG_SHA" python - "$META" <<'PY'
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); data={}
if p.exists(): data=json.loads(p.read_text(encoding='utf-8'))
data.update({'case_id':'120fs_popruzhenko_full_model_current_observability','status':os.environ['STATUS'],'exit_code':int(os.environ['RC']),
             'slurm_job_id':os.environ.get('SLURM_JOB_ID',''),'execution_git_sha':os.environ['CODE_SHA'],
             'config_sha256':os.environ['CFG_SHA'],'config_path':'configs/120fs_popruzhenko_full_model_current_observability.json',
             'output_npz':'cases/120fs_popruzhenko_full_model_current_observability/result.npz','diagnostic_schema':'khz_filament.propagation_observability.v1'})
p.write_text(json.dumps(data,indent=2)+"\n",encoding='utf-8')
PY
}
trap 'rc=$?; write_meta failed "$rc"; exit "$rc"' ERR
write_meta running 0
cd "$CODE_DIR"
python test_run.py --cfg "$CFG" --gpu --dtype fp32 --out "$OUT" --fig-dir "$RUN_ROOT/cases/120fs_popruzhenko_full_model_current_observability/figures" --fig-dpi 200
write_meta completed 0
python tools/validate_current_observability_baseline.py --npz "$OUT" --config "$CFG" --run-metadata "$META" --out-dir "$ANALYSIS"
