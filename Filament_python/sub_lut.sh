#!/bin/bash
#SBATCH -J ionlut_build
#SBATCH -p gpu
#SBATCH -x g0601,g0605
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o ionlut_build.%j.out
#SBATCH -e ionlut_build.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

CFG="${CFG:-khz_config_lut.json}"

if [[ ! -f "$CFG" ]]; then
  echo "[fatal] config not found: $CFG"
  exit 3
fi

if [[ ! -f "../build_ion_lut.py" ]]; then
  echo "[fatal] build_ion_lut.py not found at ../build_ion_lut.py"
  exit 3
fi

module load miniforge/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Filament_python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU="${UPPE_USE_GPU:-1}"
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

echo "[ionlut] CFG=$CFG"
echo "[ionlut] UPPE_USE_GPU=$UPPE_USE_GPU"
echo "[ionlut] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

python ../build_ion_lut.py --config "$CFG"
