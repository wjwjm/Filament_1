#!/usr/bin/env bash
set -euo pipefail
: "${REPO_DIR:?missing REPO_DIR}"
: "${RUN_DIR:?missing RUN_DIR}"
SCRIPT="${REPO_DIR}/Filament_python/tools/raman_off_kerr085_full.sbatch"
[[ -f "${SCRIPT}" ]] || { echo "FATAL: batch script not found: ${SCRIPT}" >&2; exit 2; }
[[ -d "${REPO_DIR}/.git" ]] || { echo "FATAL: REPO_DIR is not a Git worktree" >&2; exit 2; }
mkdir -p "${RUN_DIR}"
exec sbatch \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/slurm-%j.out" \
  --error="${RUN_DIR}/slurm-%j.err" \
  "${SCRIPT}"
