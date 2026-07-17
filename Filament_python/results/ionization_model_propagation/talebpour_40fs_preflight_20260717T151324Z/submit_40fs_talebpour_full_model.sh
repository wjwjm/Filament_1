#!/bin/bash
#SBATCH -J tal40
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH -o /data/run01/scvi806/user_Wangjimin/Filament_1/Filament_python/outputs/ionization_model_propagation/talebpour_40fs_20260717T151324Z/logs/tal40-%j.out
#SBATCH -e /data/run01/scvi806/user_Wangjimin/Filament_1/Filament_python/outputs/ionization_model_propagation/talebpour_40fs_20260717T151324Z/logs/tal40-%j.err
set -euo pipefail
RUN_ROOT="/data/run01/scvi806/user_Wangjimin/Filament_1/Filament_python/outputs/ionization_model_propagation/talebpour_40fs_20260717T151324Z"
CODE_DIR="$RUN_ROOT/source/Filament_python"
CFG="$RUN_ROOT/configs/40fs_talebpour_full_model.json"
OUT="$RUN_ROOT/cases/40fs_talebpour_full_model/result.npz"
META="$RUN_ROOT/cases/40fs_talebpour_full_model/run_metadata.json"
mkdir -p "$(dirname "$OUT")" "$RUN_ROOT/logs" "$RUN_ROOT/cases/40fs_talebpour_full_model/figures" "$RUN_ROOT/analysis"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export CUDA_DEVICE_ORDER=PCI_BUS_ID UPPE_USE_GPU=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CODE_SHA="$(cat "$RUN_ROOT/EXECUTION_GIT_SHA")" CFG_SHA="$(sha256sum "$CFG" | awk '{print $1}')"
STATUS=running RC=0
python - "$META" <<'PY'
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'case_id':'40fs_talebpour_full_model','status':'running','exit_code':0,'slurm_job_id':os.environ.get('SLURM_JOB_ID',''),'execution_git_sha':os.environ['CODE_SHA'],'config_sha256':os.environ['CFG_SHA'],'diagnostic_schema':'khz_filament.propagation_observability.v1'},indent=2)+'\n')
PY
cd "$CODE_DIR"
python test_run.py --cfg "$CFG" --gpu --dtype fp32 --out "$OUT" --fig-dir "$RUN_ROOT/cases/40fs_talebpour_full_model/figures" --fig-dpi 200
STATUS=completed python - "$META" <<'PY'
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); d['status']=os.environ['STATUS']; p.write_text(json.dumps(d,indent=2)+'\n')
PY
python tools/validate_current_observability_baseline.py --npz "$OUT" --config "$CFG" --run-metadata "$META" --out-dir "$RUN_ROOT/analysis"
