#!/bin/bash
#SBATCH -p gpu

set -euo pipefail

# 进入提交目录（Slurm 批处理推荐），兜底到脚本目录
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# 可按需覆盖：CFG/OUT/DTYPE
CFG="${CFG:-khz_config.json}"
OUT="${OUT:-khzfil_out.npz}"
DTYPE="${DTYPE:-fp32}"
CONVERT_TO_MAT="${CONVERT_TO_MAT:-1}"
MAT_DIR="${MAT_DIR:-matlab保存数据}"
MAT_NAME="${MAT_NAME:-}"
REMOVE_NPZ="${REMOVE_NPZ:-1}"
GENERATE_FIGURES="${GENERATE_FIGURES:-1}"
FIG_DIR="${FIG_DIR:-figures}"
FIG_SELECT="${FIG_SELECT:-all}"
FIG_DPI="${FIG_DPI:-200}"
Z_SHIFT_CM="${Z_SHIFT_CM:-0}"
STAGE_ID="${STAGE_ID:-standalone}"
STAGE_NAME="${STAGE_NAME:-standalone_simulation}"
RUN_ID="${RUN_ID:-}"
CASE_ID="${CASE_ID:-}"
CASE_LABEL="${CASE_LABEL:-}"
PULSE_WIDTH_FS="${PULSE_WIDTH_FS:-}"
PROFILE_TYPE="${PROFILE_TYPE:-}"
RUN_METADATA="${RUN_METADATA:-}"
export CFG OUT MAT_DIR MAT_NAME FIG_DIR STAGE_ID STAGE_NAME RUN_ID CASE_ID CASE_LABEL PULSE_WIDTH_FS PROFILE_TYPE RUN_METADATA

write_run_metadata() {
  local status="$1"
  local exit_code="${2:-0}"
  [[ -z "$RUN_METADATA" ]] && return 0
  mkdir -p "$(dirname "$RUN_METADATA")"
  STAGE_STATUS="$status" STAGE_EXIT_CODE="$exit_code" python - "$RUN_METADATA" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
data = {}
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
data.update({
    "stage_id": os.environ.get("STAGE_ID", "standalone"),
    "stage_name": os.environ.get("STAGE_NAME", "standalone_simulation"),
    "run_id": os.environ.get("RUN_ID", ""),
    "case_id": os.environ.get("CASE_ID", ""),
    "case_label": os.environ.get("CASE_LABEL", ""),
    "pulse_width_fs": float(os.environ["PULSE_WIDTH_FS"]) if os.environ.get("PULSE_WIDTH_FS") else None,
    "profile_type": os.environ.get("PROFILE_TYPE", ""),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "config_path": os.environ.get("CFG", ""),
    "output_npz": os.environ.get("OUT", ""),
    "output_mat": str(Path(os.environ.get("MAT_DIR", ".")) / (os.environ.get("MAT_NAME") or Path(os.environ.get("OUT", "result.npz")).with_suffix(".mat").name)),
    "figure_dir": os.environ.get("FIG_DIR", ""),
    "status": os.environ.get("STAGE_STATUS"),
    "exit_code": int(os.environ.get("STAGE_EXIT_CODE", "0")),
})
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then write_run_metadata "failed" "$rc" || true; fi
}
trap on_exit EXIT

mkdir -p "$(dirname "$OUT")" "$MAT_DIR" "$FIG_DIR"
write_run_metadata "running" 0

echo "[stage] STAGE_ID=$STAGE_ID STAGE_NAME=$STAGE_NAME RUN_ID=$RUN_ID"
echo "[stage] CASE_ID=$CASE_ID CASE_LABEL=$CASE_LABEL PULSE_WIDTH_FS=$PULSE_WIDTH_FS PROFILE_TYPE=$PROFILE_TYPE SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[stage] CFG=$CFG OUT=$OUT MAT_DIR=$MAT_DIR FIG_DIR=$FIG_DIR"

if [[ ! -f "$CFG" ]]; then
  echo "[fatal] config not found: $CFG"
  exit 3
fi

if [[ ! -f "test_run.py" ]]; then
  echo "[fatal] test_run.py not found in $(pwd)"
  exit 3
fi

if [[ "$REMOVE_NPZ" == "1" && "$CONVERT_TO_MAT" != "1" ]]; then
  echo "[fatal] REMOVE_NPZ=1 requires CONVERT_TO_MAT=1; the raw NPZ must not be deleted when no MAT exists."
  exit 3
fi

# Use cluster miniforge directly, no module required
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1

# Fail before the long propagation if optional post-processing dependencies are absent.
python - "$CONVERT_TO_MAT" "$GENERATE_FIGURES" <<'PY'
import sys

convert_to_mat, generate_figures = sys.argv[1:]
checks = []
if convert_to_mat == "1":
    checks.append(("scipy", "CONVERT_TO_MAT=1"))
if generate_figures == "1":
    checks.append(("matplotlib", "GENERATE_FIGURES=1"))

for module, reason in checks:
    try:
        __import__(module)
    except Exception as exc:
        print(f"[预检] {reason} requires Python package {module!r}: {exc}")
        raise SystemExit(4)
    print(f"[预检] {module} available ({reason})")
PY

# 与作业申请线程数对齐（若设置了 --cpus-per-task）
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

python - <<'PY'
import os, sys
try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    if n > 0:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(
            f"[预检] 运行环境: UPPE_USE_GPU={os.environ.get('UPPE_USE_GPU')} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
            f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')} | "
            f"device_count={n} | using={dev.id}:{name}"
        )
    else:
        print("[预检] 未检测到可见GPU，任务终止。")
        sys.exit(2)
except Exception as e:
    print(f"[预检] CuPy/驱动初始化失败: {e}")
    sys.exit(1)
PY

CMD=(python test_run.py --cfg "$CFG" --gpu --dtype "$DTYPE" --out "$OUT")
if [[ "$GENERATE_FIGURES" == "1" ]]; then
  CMD+=(--fig-dir "$FIG_DIR" --fig-select "$FIG_SELECT" --fig-dpi "$FIG_DPI" --z-shift-cm "$Z_SHIFT_CM")
  if [[ -n "$RUN_METADATA" ]]; then CMD+=(--fig-metadata-json "$RUN_METADATA"); fi
else
  CMD+=(--no-plots)
fi
if [[ "$CONVERT_TO_MAT" == "1" ]]; then
  CMD+=(--mat-dir "$MAT_DIR")
  if [[ -n "$MAT_NAME" ]]; then
    CMD+=(--mat-name "$MAT_NAME")
  fi
  if [[ "$REMOVE_NPZ" == "1" ]]; then
    CMD+=(--remove-npz)
  fi
fi
"${CMD[@]}"
write_run_metadata "completed" 0
