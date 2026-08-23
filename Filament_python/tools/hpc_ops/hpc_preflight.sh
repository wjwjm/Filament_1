#!/usr/bin/env bash
# Read-only HPC preflight. It never creates a run directory, lock, receipt,
# scheduler job, or production result. Its only stdout is the JSON report.

set -o pipefail

RC_ARGS=64
RC_ACCOUNT_ROOT=65
RC_REPO=66
RC_GIT=67
RC_PROXY=68
RC_BUNDLE=69
RC_TOOLS=70
RC_PYTHON=71
RC_SCHEDULER=72
RC_EXPECTED=73
RC_OUTPUT=74

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_proxy_env.sh
. "$SCRIPT_DIR/hpc_proxy_env.sh"

ACCOUNT=""
REMOTE_ROOT=""
REPO=""
EXPECTED_HEAD=""
EXPECTED_BRANCH=""
PROXY_ENV=""
GITHUB_URL=""
GITHUB_REF=""
BUNDLE=""
BUNDLE_SHA=""
MINIFORGE_ROOT="/data/apps/miniforge/25.3.0-3"

FAIL=0
FAIL_RC=0
SOURCE_CLASS="none"
ACCOUNT_ROOT_OK=false
REPO_OK=false
TOOLS_OK=false
PYTHON_OK=false
PROXY_OR_BUNDLE_OK=false
GITHUB_URL_OK=false
GITHUB_REF_OK=false
EXPECTED_HEAD_OK=false
EXPECTED_BRANCH_OK=false
declare -a ERRORS=()

add_error() {
    local message="$1" code="$2"
    ERRORS+=("$message")
    FAIL=1
    if [[ "$FAIL_RC" == 0 ]]; then
        FAIL_RC="$code"
    fi
}

json_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '%s' "$value"
}

json_errors() {
    local error first=1
    printf '['
    for error in "${ERRORS[@]}"; do
        [[ "$first" == 1 ]] || printf ','
        first=0
        printf '"%s"' "$(json_escape "$error")"
    done
    printf ']'
}

check_remote_root() {
    local expected="" root_real
    case "$ACCOUNT" in
        scvi806) expected="/data/run01/scvi806" ;;
        t0s000727) expected="/publicfs01/fs1-t/home/t0s000727" ;;
        *) add_error "unsupported account" "$RC_ACCOUNT_ROOT"; return ;;
    esac
    if [[ "$REMOTE_ROOT" == *".."* || "$REMOTE_ROOT" == *'\'* || "$REMOTE_ROOT" =~ [[:cntrl:]] ]]; then
        add_error "remote root contains an unsafe path component" "$RC_ACCOUNT_ROOT"
        return
    fi
    case "$REMOTE_ROOT" in
        *';'*|*'|'*|*'&'*|*'`'*|*'$('*|*')'*|*'{'*|*'}'*|*'<'*|*'>'*|*'!'*|*\"*|*"'"*)
            add_error "remote root contains shell metacharacters" "$RC_ACCOUNT_ROOT"
            return ;;
    esac
    if [[ "$REMOTE_ROOT" != "$expected" && "$REMOTE_ROOT" != "$expected/"* ]]; then
        add_error "remote root does not match account mapping" "$RC_ACCOUNT_ROOT"
        return
    fi
    command -v realpath >/dev/null 2>&1 || { add_error "realpath command is required" "$RC_ACCOUNT_ROOT"; return; }
    [[ -d "$REMOTE_ROOT" && ! -L "$REMOTE_ROOT" ]] || { add_error "remote root is not a regular directory" "$RC_ACCOUNT_ROOT"; return; }
    if ! root_real="$(realpath -e -- "$REMOTE_ROOT" 2>/dev/null)"; then
        add_error "remote root could not be resolved" "$RC_ACCOUNT_ROOT"
        return
    fi
    if [[ "$root_real" != "$REMOTE_ROOT" ]]; then
        add_error "remote root must not traverse a symlink" "$RC_ACCOUNT_ROOT"
        return
    fi
    ACCOUNT_ROOT_OK=true
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --account) [[ "$#" -ge 2 ]] || return 1; ACCOUNT="${2,,}"; shift 2 ;;
            --remote-root) [[ "$#" -ge 2 ]] || return 1; REMOTE_ROOT="$2"; shift 2 ;;
            --repo) [[ "$#" -ge 2 ]] || return 1; REPO="$2"; shift 2 ;;
            --expected-head) [[ "$#" -ge 2 ]] || return 1; EXPECTED_HEAD="$2"; shift 2 ;;
            --expected-branch) [[ "$#" -ge 2 ]] || return 1; EXPECTED_BRANCH="$2"; shift 2 ;;
            --proxy-env) [[ "$#" -ge 2 ]] || return 1; PROXY_ENV="$2"; shift 2 ;;
            --github-url) [[ "$#" -ge 2 ]] || return 1; GITHUB_URL="$2"; shift 2 ;;
            --github-ref) [[ "$#" -ge 2 ]] || return 1; GITHUB_REF="$2"; shift 2 ;;
            --bundle) [[ "$#" -ge 2 ]] || return 1; BUNDLE="$2"; shift 2 ;;
            --bundle-sha) [[ "$#" -ge 2 ]] || return 1; BUNDLE_SHA="$2"; shift 2 ;;
            --json) shift ;;
            *) return 1 ;;
        esac
    done
    [[ -n "$ACCOUNT" && -n "$REMOTE_ROOT" && -n "$REPO" && -n "$EXPECTED_HEAD" && -n "$EXPECTED_BRANCH" && -n "$PROXY_ENV" && -n "$GITHUB_URL" && -n "$GITHUB_REF" ]]
}

check_repo() {
    local actual_head actual_branch status repo_real root_real
    [[ -d "$REPO" && ! -L "$REPO" ]] || { add_error "repository directory is missing or is a symlink" "$RC_REPO"; return; }
    if ! repo_real="$(realpath -e -- "$REPO" 2>/dev/null)" || ! root_real="$(realpath -e -- "$REMOTE_ROOT" 2>/dev/null)"; then
        add_error "repository or remote root could not be resolved" "$RC_REPO"
        return
    fi
    if [[ "$repo_real" != "$root_real/"* ]]; then
        add_error "repository is outside the remote root" "$RC_REPO"
        return
    fi
    if ! actual_head="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"; then
        add_error "git HEAD could not be read" "$RC_GIT"
        return
    fi
    if ! actual_branch="$(git -C "$REPO" symbolic-ref --short -q HEAD 2>/dev/null)"; then
        add_error "repository is not on a named branch" "$RC_GIT"
        return
    fi
    if ! status="$(git -C "$REPO" status --porcelain=v1 --untracked-files=all 2>/dev/null)"; then
        add_error "repository status could not be read" "$RC_GIT"
        return
    fi
    [[ "$actual_head" == "$EXPECTED_HEAD" ]] || add_error "repository HEAD does not match expected head" "$RC_EXPECTED"
    [[ "$actual_branch" == "$EXPECTED_BRANCH" ]] || add_error "repository branch does not match expected branch" "$RC_EXPECTED"
    [[ -z "$status" ]] || add_error "repository worktree is not clean" "$RC_REPO"
    if [[ "$actual_head" == "$EXPECTED_HEAD" && "$actual_branch" == "$EXPECTED_BRANCH" && -z "$status" ]]; then
        REPO_OK=true
    fi
}

check_tools() {
    local tool missing=0
    for tool in git sha256sum sbatch sacct scontrol; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            add_error "required command is missing: $tool" "$RC_TOOLS"
            missing=1
        fi
    done
    [[ "$missing" == 0 ]] && TOOLS_OK=true
}

check_python_env() {
    local conda_script="$MINIFORGE_ROOT/etc/profile.d/conda.sh" expected_prefix="$MINIFORGE_ROOT/envs/Filament_python"
    local conda_rc numpy_rc cupy_rc python_bin python_real prefix_real
    if [[ ! -f "$conda_script" ]]; then
        add_error "configured Miniforge conda hook is missing" "$RC_PYTHON"
        return
    fi
    # This is a fixed, known environment path; no user-controlled file is read.
    . "$conda_script" >/dev/null 2>&1 || { add_error "could not load configured conda hook" "$RC_PYTHON"; return; }
    conda activate Filament_python >/dev/null 2>&1
    conda_rc=$?
    [[ "$conda_rc" == 0 ]] || { add_error "Filament_python conda environment could not be activated" "$RC_PYTHON"; return; }
    if [[ "${CONDA_PREFIX:-}" != "$expected_prefix" ]]; then
        add_error "Filament_python conda prefix does not match the configured environment" "$RC_PYTHON"
        return
    fi
    if ! python_bin="$(command -v python)" || ! python_real="$(realpath -e -- "$python_bin" 2>/dev/null)" || ! prefix_real="$(realpath -e -- "$expected_prefix" 2>/dev/null)"; then
        add_error "Filament_python interpreter could not be resolved" "$RC_PYTHON"
        return
    fi
    if [[ "$python_real" != "$prefix_real/"* ]]; then
        add_error "active Python is outside the configured Filament_python environment" "$RC_PYTHON"
        return
    fi
    python -c 'import numpy' >/dev/null 2>&1
    numpy_rc=$?
    python -c 'import cupy' >/dev/null 2>&1
    cupy_rc=$?
    [[ "$numpy_rc" == 0 ]] || add_error "Filament_python numpy import failed" "$RC_PYTHON"
    [[ "$cupy_rc" == 0 ]] || add_error "Filament_python cupy import failed" "$RC_PYTHON"
    if [[ "$numpy_rc" == 0 && "$cupy_rc" == 0 ]]; then
        PYTHON_OK=true
    fi
}

check_proxy() {
    local rc
    hpc_proxy_load "$PROXY_ENV"
    rc=$?
    if [[ "$rc" != 0 ]]; then
        PROXY_FAILED=1
        return
    fi
    hpc_proxy_git_ls_remote "$GITHUB_URL" "$GITHUB_REF" "$EXPECTED_HEAD" "${HPC_PROXY_GIT_TIMEOUT_SECONDS:-30}"
    rc=$?
    if [[ "$rc" == 0 ]]; then
        SOURCE_CLASS="strict_remote_verified"
        PROXY_OR_BUNDLE_OK=true
        return
    fi
    PROXY_FAILED=1
}

check_expected_head() {
    if [[ "$EXPECTED_HEAD" =~ ^[0-9a-fA-F]+$ &&
          ( ${#EXPECTED_HEAD} -eq 40 || ${#EXPECTED_HEAD} -eq 64 ) ]]; then
        EXPECTED_HEAD_OK=true
    else
        add_error "expected head must be a full SHA-1 or SHA-256 hex value" "$RC_EXPECTED"
    fi
}

check_github_ref() {
    if [[ "$GITHUB_REF" == *$'\n'* || "$GITHUB_REF" == *$'\r'* || "$GITHUB_REF" == *$'\t'* ||
          ! "$GITHUB_REF" =~ ^refs/(heads|tags)/[A-Za-z0-9][A-Za-z0-9._/-]*$ ||
          "$GITHUB_REF" == *".."* || "$GITHUB_REF" == *"//"* || "$GITHUB_REF" == */ ]]; then
        add_error "Git ref must be a safe refs/heads or refs/tags value" "$RC_PROXY"
    else
        GITHUB_REF_OK=true
    fi
}

check_expected_branch() {
    if [[ "$EXPECTED_BRANCH" == *$'\n'* || "$EXPECTED_BRANCH" == *$'\r'* || "$EXPECTED_BRANCH" == *$'\t'* ||
          ! "$EXPECTED_BRANCH" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ||
          "$EXPECTED_BRANCH" == *".."* || "$EXPECTED_BRANCH" == *"//"* || "$EXPECTED_BRANCH" == */ ]]; then
        add_error "expected branch is not a safe named branch" "$RC_EXPECTED"
    else
        EXPECTED_BRANCH_OK=true
    fi
}

check_bundle() {
    local actual_sha heads
    [[ -n "$BUNDLE" && -n "$BUNDLE_SHA" ]] || return
    [[ -f "$BUNDLE" && ! -L "$BUNDLE" ]] || { add_error "bundle is not a regular file" "$RC_BUNDLE"; return; }
    [[ "$BUNDLE_SHA" =~ ^[0-9a-fA-F]{64}$ ]] || { add_error "bundle SHA256 must be 64 hexadecimal characters" "$RC_BUNDLE"; return; }
    actual_sha="$(sha256sum -- "$BUNDLE" 2>/dev/null | awk '{print $1}' || true)"
    [[ "${actual_sha,,}" == "${BUNDLE_SHA,,}" ]] || { add_error "bundle SHA256 does not match expected value" "$RC_BUNDLE"; return; }
    git -C "$REPO" bundle verify "$BUNDLE" >/dev/null 2>&1 || { add_error "git bundle verification failed" "$RC_BUNDLE"; return; }
    heads="$(git bundle list-heads "$BUNDLE" 2>/dev/null || true)"
    if ! printf '%s\n' "$heads" | awk -v h="$EXPECTED_HEAD" -v r="$GITHUB_REF" '$1 == h && $2 == r { found=1 } END { exit(found ? 0 : 1) }'; then
        add_error "bundle does not contain expected head and ref" "$RC_BUNDLE"
        return
    fi
    SOURCE_CLASS="verified_bundle_non_strict"
    PROXY_OR_BUNDLE_OK=true
}

build_json() {
    local ok=false
    [[ "$FAIL" == 0 ]] && ok=true
    printf '{"schema":"filament.hpc_preflight.v1","ok":%s,"account":"%s","remote_root":"%s","source_class":"%s","checks":{"account_root":%s,"repo":%s,"tools":%s,"python_env":%s,"proxy_or_bundle":%s},"errors":' \
        "$ok" "$(json_escape "$ACCOUNT")" "$(json_escape "$REMOTE_ROOT")" "$(json_escape "$SOURCE_CLASS")" \
        "$ACCOUNT_ROOT_OK" "$REPO_OK" "$TOOLS_OK" "$PYTHON_OK" "$PROXY_OR_BUNDLE_OK"
    json_errors
    printf '}\n'
}

check_github_url() {
    if [[ "$GITHUB_URL" == *$'\n'* || "$GITHUB_URL" == *$'\r'* || "$GITHUB_URL" == *$'\t'* ||
          ! "$GITHUB_URL" =~ ^https://github\.com/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(\.git)?$ ]]; then
        add_error "GitHub URL must be https://github.com/<owner>/<repo>[.git] without credentials, query, or fragment" "$RC_PROXY"
    else
        GITHUB_URL_OK=true
    fi
}

if ! parse_args "$@"; then
    ERRORS+=("required arguments are missing or invalid")
    FAIL=1
    FAIL_RC="$RC_ARGS"
else
    check_remote_root
    check_github_url
    check_github_ref
    check_expected_head
    check_expected_branch
    check_repo
    check_tools
    check_python_env
    if [[ "$GITHUB_URL_OK" == true && "$GITHUB_REF_OK" == true && "$EXPECTED_HEAD_OK" == true ]]; then
        check_proxy
        if [[ "$SOURCE_CLASS" == none ]]; then
            check_bundle
        fi
        if [[ "$SOURCE_CLASS" == none ]]; then
            add_error "neither proxy nor verified bundle provenance passed" "$RC_PROXY"
        fi
    else
        add_error "neither proxy nor verified bundle provenance passed" "$RC_PROXY"
    fi
fi

REPORT="$(build_json)"
printf '%s' "$REPORT"
if [[ "$FAIL" == 0 ]]; then
    exit 0
fi
exit "${FAIL_RC:-$RC_ARGS}"
