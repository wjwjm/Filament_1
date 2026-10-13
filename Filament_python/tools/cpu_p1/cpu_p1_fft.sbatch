#!/usr/bin/env bash
#SBATCH --job-name=cpu-p1-fft
#SBATCH --partition=v6_384
#SBATCH --account=bscc-t
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=320G
#SBATCH --time=01:30:00
#SBATCH --output=cpu-p1-fft-%j.out
#SBATCH --error=cpu-p1-fft-%j.err
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo 'usage: sbatch cpu_p1_fft.sbatch BENCH_ROOT NEW_OUTPUT_ROOT VENDOR_DIR' >&2
  exit 2
fi
BENCH_ROOT="$1"
OUTPUT_ROOT="$2"
VENDOR_DIR="$3"
FIXED_PYTHON=/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python

export UPPE_USE_GPU=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$BENCH_ROOT/Filament_python:$VENDOR_DIR"
export PYTHONNOUSERSITE=1

[[ -x "$FIXED_PYTHON" ]] || { echo 'fixed Python interpreter unavailable' >&2; exit 2; }
[[ -f "$BENCH_ROOT/Filament_python/tools/hpc_ops/audit_batch_entry.py" ]] || { echo 'BENCH_ROOT invalid' >&2; exit 2; }
[[ -d "$VENDOR_DIR" ]] || { echo 'vendor directory unavailable' >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo 'output root already exists' >&2; exit 2; }

"$FIXED_PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/hpc_ops/audit_batch_entry.py" \
  --batch "$BENCH_ROOT/Filament_python/tools/cpu_p1/cpu_p1_fft.sbatch" \
  --fixed-python "$FIXED_PYTHON"
"$FIXED_PYTHON" -s -B -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'

srun --ntasks=1 --cpus-per-task=96 --cpu-bind=cores \
  "$FIXED_PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/cpu_p1/cpu_p1_fft.py" \
  --output-root "$OUTPUT_ROOT" --vendor-dir "$VENDOR_DIR" \
  --deadline-seconds 4500 --memory-limit-bytes 343597383680

