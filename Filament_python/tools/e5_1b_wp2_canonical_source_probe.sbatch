#!/usr/bin/env bash
# One-GPU, source-only E5-1B WP2 qualification; g0603 exclusion is temporary.
#SBATCH --job-name=e5-wp2-source-load
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclude=g0603
set -Eeuo pipefail
umask 077

: "${REPO_DIR:?}" "${EXPECTED_GIT_SHA:?}" "${RUN_ROOT:?}" "${CAMPAIGN_ID:?}"
: "${EXPECTED_PROBE_SHA:?}" "${EXPECTED_BATCH_SHA:?}" "${SOURCE_MANIFEST:?}"
: "${EXPECTED_SOURCE_MANIFEST_SHA:?}" "${EXPECTED_IMPLEMENTATION_SHA:?}"
: "${SLURM_JOB_ID:?}" "${CUDA_VISIBLE_DEVICES:?}"
[[ "$RUN_ROOT" == /data/run01/scvi806/* ]]
test "$(hostname -s)" != g0603
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum -- "$0" | cut -d ' ' -f 1)" = "$EXPECTED_BATCH_SHA"
test "$(sha256sum -- "$REPO_DIR/Filament_python/tools/run_e5_1b_wp2_canonical_source_probe.py" | cut -d ' ' -f 1)" = "$EXPECTED_PROBE_SHA"
test "$(sha256sum -- "$SOURCE_MANIFEST" | cut -d ' ' -f 1)" = "$EXPECTED_SOURCE_MANIFEST_SHA"
test ! -e "$RUN_ROOT"

readonly ENV_PREFIX=/data/home/scvi806/.conda/envs/Filament_python
readonly PYTHON="$ENV_PREFIX/bin/python"
test -x "$PYTHON"
"$PYTHON" -s -B "$REPO_DIR/Filament_python/tools/hpc_ops/audit_batch_entry.py" \
  --batch "$0" --fixed-python "$PYTHON" >/dev/null
"$PYTHON" -s -B "$REPO_DIR/Filament_python/tools/hpc_ops/provenance_v2.py" validate \
  --repo "$REPO_DIR" --manifest "$SOURCE_MANIFEST" --require-hash-scope >/dev/null
actual_identity="$(PYTHONPATH="$REPO_DIR/Filament_python" UPPE_USE_GPU=0 \
  "$PYTHON" -s -B -c 'from KHz_filament.hr4e5_recovery_identity import capture_implementation_identity; print(capture_implementation_identity()["identity_sha256"])')"
test "$actual_identity" = "$EXPECTED_IMPLEMENTATION_SHA"

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
test "$(realpath -e -- "$CONDA_PREFIX")" = "$(realpath -e -- "$ENV_PREFIX")"
mkdir -m 700 -- "$RUN_ROOT"
mkdir -m 700 -- "$RUN_ROOT/runtime" "$RUN_ROOT/runtime/home" \
  "$RUN_ROOT/runtime/tmp" "$RUN_ROOT/runtime/xdg" \
  "$RUN_ROOT/runtime/cupy" "$RUN_ROOT/runtime/pycache" "$RUN_ROOT/case"
export HOME="$RUN_ROOT/runtime/home" TMPDIR="$RUN_ROOT/runtime/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/runtime/xdg" CUPY_CACHE_DIR="$RUN_ROOT/runtime/cupy"
export PYTHONPYCACHEPREFIX="$RUN_ROOT/runtime/pycache" PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_DIR/Filament_python" UPPE_USE_GPU=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON" -s -B "$REPO_DIR/Filament_python/tools/run_e5_1b_wp2_canonical_source_probe.py" \
  --run-root "$RUN_ROOT/case" --campaign-id "$CAMPAIGN_ID"
