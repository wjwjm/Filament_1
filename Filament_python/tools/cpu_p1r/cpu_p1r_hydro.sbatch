#!/usr/bin/env bash
#SBATCH --partition=v6_384
#SBATCH --account=bscc-t
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=320G
#SBATCH --time=01:30:00

set -Eeuo pipefail
readonly PYTHON=/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python
readonly BENCH_ROOT=${1:?usage: sbatch cpu_p1r_hydro.sbatch BENCH_ROOT OUTPUT_ROOT VENDOR_DIR INPUT_ROOT WORKLOAD}
readonly OUTPUT_ROOT=${2:?missing OUTPUT_ROOT}
readonly VENDOR_DIR=${3:?missing VENDOR_DIR}
readonly INPUT_ROOT=${4:?missing INPUT_ROOT}
readonly WORKLOAD=${5:?missing frozen WORKLOAD_768.json}
readonly MANIFEST=${INPUT_ROOT}/candidate_input_manifest.json
readonly RUNNER=${BENCH_ROOT}/Filament_python/tools/cpu_p1r/cpu_p1r_hydro.py
readonly AUDIT=${BENCH_ROOT}/Filament_python/tools/hpc_ops/audit_batch_entry.py
readonly SELF=${BASH_SOURCE[0]}

export UPPE_USE_GPU=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=${BENCH_ROOT}/Filament_python:${VENDOR_DIR}
export P1R_ALLOCATION_START_EPOCH=$(date +%s)

bash -n "$SELF"
"$PYTHON" -s -B "$AUDIT" --batch "$SELF" --fixed-python "$PYTHON"
test -x "$PYTHON"
test -f "$RUNNER"
test -f "$MANIFEST"
test -f "$WORKLOAD"
test -d "$VENDOR_DIR"

exec "$PYTHON" -s -B "$RUNNER" \
  --input-root "$INPUT_ROOT" \
  --manifest "$MANIFEST" \
  --workload "$WORKLOAD" \
  --output-root "$OUTPUT_ROOT" \
  --vendor-dir "$VENDOR_DIR"
