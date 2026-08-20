#!/usr/bin/env bash
# Submit the single historical f_R-mixture causal job from an isolated run dir.
#
# Slurm opens its stdout/stderr files before the batch script starts.  Do not
# submit from REPO_DIR: the automatic slurm-<jobid>.out would become an
# untracked file and correctly trip the script's immutable-worktree guard.
set -euo pipefail

: "${REPO_DIR:?missing REPO_DIR}"
: "${RUN_DIR:?missing RUN_DIR}"

SCRIPT="${REPO_DIR}/Filament_python/tools/historical_fr_mixture_full.sbatch"
[[ -f "${SCRIPT}" ]] || { echo "FATAL: batch script not found: ${SCRIPT}" >&2; exit 2; }
[[ -d "${REPO_DIR}/.git" ]] || { echo "FATAL: REPO_DIR is not a Git worktree: ${REPO_DIR}" >&2; exit 2; }

# Create the destination before sbatch opens its log files.  --chdir controls
# the Slurm allocation's initial cwd; --output/--error make that invariant
# explicit and leave REPO_DIR clean for the provenance guard in the batch job.
mkdir -p "${RUN_DIR}"
exec sbatch \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/slurm-%j.out" \
  --error="${RUN_DIR}/slurm-%j.err" \
  "${SCRIPT}"
