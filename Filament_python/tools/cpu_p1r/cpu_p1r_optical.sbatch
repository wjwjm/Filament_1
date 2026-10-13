#!/usr/bin/env bash
#SBATCH --partition=v6_384
#SBATCH --account=bscc-t
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=320G
#SBATCH --time=02:00:00
set -Eeuo pipefail
readonly PYTHON=/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python
readonly BENCH_ROOT=${1:?missing BENCH_ROOT}
readonly OUTPUT_ROOT=${2:?missing OUTPUT_ROOT}
readonly VENDOR_DIR=${3:?missing VENDOR_DIR}
readonly INPUT_ROOT=${4:?missing INPUT_ROOT}
readonly SELF=${BASH_SOURCE[0]}
export UPPE_USE_GPU=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=${BENCH_ROOT}/Filament_python:${VENDOR_DIR}
bash -n "$SELF"
"$PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$SELF" --fixed-python "$PYTHON"
export P1R_ALLOCATION_START_EPOCH
P1R_ALLOCATION_START_EPOCH=$("$PYTHON" -s -B -c 'import subprocess,os,re,datetime; s=subprocess.check_output(["scontrol","show","job",os.environ["SLURM_JOB_ID"],"-o"],text=True); v=re.search(r"(?:^| )StartTime=(\S+)",s).group(1); print(datetime.datetime.fromisoformat(v).timestamp())')
exec "$PYTHON" -s -B "$BENCH_ROOT/Filament_python/tools/cpu_p1r/cpu_p1r_optical.py" --bench-root "$BENCH_ROOT" --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" --vendor-dir "$VENDOR_DIR"
