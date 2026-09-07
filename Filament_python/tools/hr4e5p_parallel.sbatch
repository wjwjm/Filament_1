#!/usr/bin/env bash
#SBATCH --job-name=hr4e5p
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
set -euo pipefail
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${CASE_DIR:?}" "${MODE:?}" "${STATE_JSON:?}" "${PARTITION_JSON:?}" "${GPU_COUNT:?}"
cd "${REPO_DIR}"
test "$(git rev-parse HEAD)" = "${EXPECTED_GIT_SHA}"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="${REPO_DIR}/Filament_python" PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${CASE_DIR}/pycache" CUPY_CACHE_DIR="${CASE_DIR}/cupy_cache" XDG_CACHE_HOME="${CASE_DIR}/xdg_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${CASE_DIR}"
resource_monitor_pid=""
if [[ -n "${P4_RESOURCE_MONITOR_OUT:-}" ]]; then
  : > "${P4_RESOURCE_MONITOR_OUT}"
  (
    while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "${P4_RESOURCE_MONITOR_OUT}"
      sleep 0.2
    done
  ) &
  resource_monitor_pid="$!"
  trap 'if [[ -n "${resource_monitor_pid}" ]]; then kill "${resource_monitor_pid}" 2>/dev/null || true; wait "${resource_monitor_pid}" 2>/dev/null || true; fi' EXIT
fi
if [[ -n "${P5_RESOURCE_MONITOR_OUT:-}" ]]; then
  : "${CUDA_VISIBLE_DEVICES:?P5 resource monitor requires CUDA_VISIBLE_DEVICES}"
  : > "${P5_RESOURCE_MONITOR_OUT}"
  (
    while true; do
      nvidia-smi --id "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,memory.used --format=csv,noheader,nounits >> "${P5_RESOURCE_MONITOR_OUT}"
      sleep 0.2
    done
  ) &
  p5_resource_monitor_pid="$!"
  trap 'if [[ -n "${p5_resource_monitor_pid}" ]]; then kill "${p5_resource_monitor_pid}" 2>/dev/null || true; wait "${p5_resource_monitor_pid}" 2>/dev/null || true; fi' EXIT
fi
"${PYTHON}" -c 'from KHz_filament.device import debug_backend; assert debug_backend()["backend"] == "cupy", debug_backend()'
if [[ "${MODE}" == "serial" ]]; then
  "${PYTHON}" "${REPO_DIR}/Filament_python/tools/run_hr4e5p.py" serial --state "${STATE_JSON}" --out "${CASE_DIR}/serial_result.json" --n-hydro-steps "${N_HYDRO_STEPS:-1000}" --batch-intervals "${BATCH_INTERVALS:-1}"
elif [[ "${MODE}" == "parallel" ]]; then
  test "${SLURM_GPUS_ON_NODE:-${GPU_COUNT}}" -ge "${GPU_COUNT}"
  srun --ntasks="${GPU_COUNT}" --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" "${PYTHON}" "${REPO_DIR}/Filament_python/tools/run_hr4e5p.py" worker --partition "${PARTITION_JSON}" --out-dir "${CASE_DIR}" --n-hydro-steps "${N_HYDRO_STEPS:-1000}"
  "${PYTHON}" "${REPO_DIR}/Filament_python/tools/run_hr4e5p.py" gather --partition "${PARTITION_JSON}" --worker-dir "${CASE_DIR}" --out "${CASE_DIR}/gather_result.json" --batch-intervals "${BATCH_INTERVALS:-1}" --n-hydro-steps "${N_HYDRO_STEPS:-1000}"
else
  echo "invalid MODE=${MODE}" >&2
  exit 64
fi
