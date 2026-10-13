#!/usr/bin/env bash
#SBATCH --partition=v6_384
#SBATCH --account=bscc-t
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=320G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -Eeuo pipefail
readonly PYTHON=/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python
readonly BENCH_ROOT=${1:?missing bench root}
readonly OUTPUT_ROOT=${2:?missing output root}
readonly VENDOR_DIR=${3:?missing vendor}
readonly PROVENANCE=${4:?missing provenance}
readonly SOURCE_ROOT=${5:?missing source root}
readonly SOURCE_JOB=${6:?missing source job}
readonly SOURCE_REPO=${7:?missing source repo}
readonly SOURCE_MANIFEST=${8:?missing source manifest}
export UPPE_USE_GPU=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH=${BENCH_ROOT}/Filament_python:${VENDOR_DIR}
bash -n "${BASH_SOURCE[0]}"
"$PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "${BASH_SOURCE[0]}" --fixed-python "$PYTHON"
"$PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$BENCH_ROOT" --manifest "$PROVENANCE" --require-hash-scope
"$PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/cpu_p1r2/allocation_probe.py" --output "${OUTPUT_ROOT}_allocation_probe"
exec timeout --signal=TERM --kill-after=30s 900 "$PYTHON" -s -B -m tools.cpu_p1r2.cpu_p1r2_finalize --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --source-job "$SOURCE_JOB" --source-repo "$SOURCE_REPO" --source-manifest "$SOURCE_MANIFEST"