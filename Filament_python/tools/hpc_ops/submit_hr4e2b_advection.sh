#!/usr/bin/env bash
# Submit exactly the three HR-4E-2B advection-only diagnostics.
set -euo pipefail

readonly REPO="$1"
readonly EXPECTED_SHA="$2"
readonly RUN_ROOT="$3"
readonly BATCH_ENTRY="$REPO/Filament_python/tools/hr4e2_spatial_case.sbatch"

test -d "$REPO" && test ! -L "$REPO"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$BATCH_ENTRY" && test ! -L "$BATCH_ENTRY"
test ! -e "$RUN_ROOT"

umask 077
mkdir -m 700 -- "$RUN_ROOT"
receipt="$RUN_ROOT/submission_receipt.tsv"
printf 'case_id\tdx_um\tdt_hydro_us\tadvection_cfl\tjob_id\tpartition\tgpu_count\tgit_sha\n' >"$receipt"
chmod 600 -- "$receipt"

submit_case() {
    local case_id="$1" dx_um="$2" cfl="$3" case_dir job_id
    case_dir="$RUN_ROOT/$case_id"
    job_id="$(sbatch --parsable \
        --export=ALL,EXPECTED_GIT_SHA="$EXPECTED_SHA",REPO_DIR="$REPO",CASE_DIR="$case_dir",CASE_ID="$case_id",FAMILY=E2-B,DX_UM="$dx_um",DT_US=0.125 \
        "$BATCH_ENTRY")"
    case "${job_id}" in
        *[!0-9]*|'') exit 1 ;;
    esac
    printf '%s\t%s\t0.125\t%s\t%s\tgpu\t1\t%s\n' "$case_id" "$dx_um" "$cfl" "$job_id" "$EXPECTED_SHA" >>"$receipt"
}

submit_case E2B_dx20um_dt0p125us 20 0.001875
submit_case E2B_dx10um_dt0p125us 10 0.00375
submit_case E2B_dx5um_dt0p125us 5 0.0075

printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","run_root":"%s"}\n' "$RUN_ROOT"
