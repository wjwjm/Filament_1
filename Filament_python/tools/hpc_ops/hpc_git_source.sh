#!/usr/bin/env bash
# Protected GitHub-or-bundle acquisition for an HPC checkout.
#
# Clone mode creates one clean checkout at the expected branch/SHA. Fetch mode
# only refreshes FETCH_HEAD for an already-clean checkout at that same
# branch/SHA; it never resets, merges, commits, pushes, or submits a job.

set -o pipefail

RC_ARGS=64
RC_URL=65
RC_PROXY=66
RC_BUNDLE=67
RC_TARGET=68
RC_GIT=69
RC_OUTPUT=70

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_proxy_env.sh
. "$SCRIPT_DIR/hpc_proxy_env.sh"

ACCOUNT=""
REMOTE_ROOT=""
STAGING_ROOT=""
MODE=""
URL=""
REF=""
EXPECTED_HEAD=""
EXPECTED_BRANCH=""
PROXY_ENV=""
TARGET=""
BUNDLE=""
BUNDLE_SHA=""
OPERATION_ID=""
STATE_FILE=""
TIMEOUT_SECONDS="${HPC_PROXY_GIT_TIMEOUT_SECONDS:-30}"
DRY_RUN=0
INSPECT_STATE=0
SOURCE_CLASS="none"
TARGET_HEAD=""
TARGET_BRANCH=""
FETCH_HEAD_VALUE=""
WORK_ROOT=""
STABLE_BUNDLE=""
VERIFY_ROOT=""
STATE_TMP=""
STATE_LAST_STATE=""
STATE_ERROR=""
STATE_RECEIPT_READY=0
BUNDLE_ERROR=""
inspect_real=""
declare -a ERRORS=()

json_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '%s' "$value"
}

emit_json() {
    local ok="$1" state="$2" error
    printf '{"schema":"filament.hpc_ops.git_source.v1","ok":%s,"state":"%s","source_class":"%s","mode":"%s","account":"%s","target":"%s","ref":"%s","expected_head":"%s","expected_branch":"%s","target_head":"%s","target_branch":"%s","fetch_head":"%s","operation_id":"%s","state_file":"%s","acquisition_only":false' \
        "$ok" "$state" "$(json_escape "$SOURCE_CLASS")" "$(json_escape "$MODE")" \
        "$(json_escape "$ACCOUNT")" "$(json_escape "$TARGET")" "$(json_escape "$REF")" \
        "$(json_escape "$EXPECTED_HEAD")" "$(json_escape "$EXPECTED_BRANCH")" \
        "$(json_escape "$TARGET_HEAD")" "$(json_escape "$TARGET_BRANCH")" \
        "$(json_escape "$FETCH_HEAD_VALUE")" "$(json_escape "$OPERATION_ID")" \
        "$(json_escape "$STATE_FILE")"
    printf ',"errors":['
    local first=1
    for error in "${ERRORS[@]}"; do
        [[ "$first" == 1 ]] || printf ','
        first=0
        printf '"%s"' "$(json_escape "$error")"
    done
    printf ']}\n'
}

emit_state_json() {
    local state="$1"
    printf '{"schema":"filament.hpc_git_acquisition.v2","operation_id":"%s","state":"%s","mode":"%s","target":"%s","expected_head":"%s","expected_branch":"%s","source_class":"%s","state_file":"%s","target_head":"%s","target_branch":"%s","fetch_head":"%s"' \
        "$(json_escape "$OPERATION_ID")" "$(json_escape "$state")" \
        "$(json_escape "$MODE")" "$(json_escape "$TARGET")" \
        "$(json_escape "$EXPECTED_HEAD")" "$(json_escape "$EXPECTED_BRANCH")" \
        "$(json_escape "$SOURCE_CLASS")" "$(json_escape "$STATE_FILE")" \
        "$(json_escape "$TARGET_HEAD")" "$(json_escape "$TARGET_BRANCH")" \
        "$(json_escape "$FETCH_HEAD_VALUE")"
    if [[ -n "$STATE_ERROR" ]]; then
        printf ',"error":"%s"' "$(json_escape "$STATE_ERROR")"
    fi
    printf '}\n'
}

write_state() {
    local next_state="$1" error_message="${2:-}" temporary
    [[ "$STATE_RECEIPT_READY" == 1 ]] || return 1
    case "$STATE_LAST_STATE:$next_state" in
        ":started"|":failed"|"started:acquiring"|"started:failed"|\
        "acquiring:acquiring"|"acquiring:checkout_verified"|"acquiring:failed"|\
        "checkout_verified:completed"|"checkout_verified:failed") ;;
        *) return 1 ;;
    esac
    case "$next_state" in
        started|acquiring|checkout_verified|completed|failed) ;;
        *) return 1 ;;
    esac
    STATE_ERROR="$error_message"
    temporary="$(mktemp -- "${STATE_FILE}.tmp.XXXXXX" 2>/dev/null)" || return 1
    STATE_TMP="$temporary"
    if ! chmod 600 -- "$temporary" || ! emit_state_json "$next_state" > "$temporary"; then
        rm -f -- "$temporary" || true
        STATE_TMP=""
        return 1
    fi
    if ! mv -f -- "$temporary" "$STATE_FILE"; then
        rm -f -- "$temporary" || true
        STATE_TMP=""
        return 1
    fi
    STATE_TMP=""
    STATE_LAST_STATE="$next_state"
    return 0
}

cleanup() {
    local status=$?
    trap - EXIT HUP INT TERM
    if [[ "$status" != 0 && "$STATE_RECEIPT_READY" == 1 &&
          "$STATE_LAST_STATE" != failed && "$STATE_LAST_STATE" != completed ]]; then
        write_state failed "acquisition terminated before completion" || true
    fi
    if [[ -n "$STATE_TMP" ]]; then rm -f -- "$STATE_TMP" || true; fi
    if [[ -n "$STABLE_BUNDLE" ]]; then rm -f -- "$STABLE_BUNDLE" || true; fi
    if [[ -n "$VERIFY_ROOT" ]]; then rm -rf -- "$VERIFY_ROOT" || true; fi
    if [[ -n "$WORK_ROOT" ]]; then rm -rf -- "$WORK_ROOT" || true; fi
    exit "$status"
}
trap cleanup EXIT HUP INT TERM

fail() {
    local message="$1"
    ERRORS+=("$message")
    if [[ "$STATE_RECEIPT_READY" == 1 && "$STATE_LAST_STATE" != failed && "$STATE_LAST_STATE" != completed ]]; then
        if ! write_state failed "$message"; then
            ERRORS+=("state receipt update failed")
        fi
    fi
    emit_json false "rejected_or_failed"
    exit "${2:-$RC_ARGS}"
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --account) [[ "$#" -ge 2 ]] || return 1; ACCOUNT="${2,,}"; shift 2 ;;
            --remote-root) [[ "$#" -ge 2 ]] || return 1; REMOTE_ROOT="$2"; shift 2 ;;
            --staging-root) [[ "$#" -ge 2 ]] || return 1; STAGING_ROOT="$2"; shift 2 ;;
            --mode) [[ "$#" -ge 2 ]] || return 1; MODE="$2"; shift 2 ;;
            --url) [[ "$#" -ge 2 ]] || return 1; URL="$2"; shift 2 ;;
            --ref) [[ "$#" -ge 2 ]] || return 1; REF="$2"; shift 2 ;;
            --expected-head) [[ "$#" -ge 2 ]] || return 1; EXPECTED_HEAD="$2"; shift 2 ;;
            --expected-branch) [[ "$#" -ge 2 ]] || return 1; EXPECTED_BRANCH="$2"; shift 2 ;;
            --proxy-env) [[ "$#" -ge 2 ]] || return 1; PROXY_ENV="$2"; shift 2 ;;
            --target) [[ "$#" -ge 2 ]] || return 1; TARGET="$2"; shift 2 ;;
            --bundle) [[ "$#" -ge 2 ]] || return 1; BUNDLE="$2"; shift 2 ;;
            --bundle-sha) [[ "$#" -ge 2 ]] || return 1; BUNDLE_SHA="$2"; shift 2 ;;
            --operation-id) [[ "$#" -ge 2 ]] || return 1; OPERATION_ID="$2"; shift 2 ;;
            --state-file) [[ "$#" -ge 2 ]] || return 1; STATE_FILE="$2"; shift 2 ;;
            --timeout-seconds) [[ "$#" -ge 2 ]] || return 1; TIMEOUT_SECONDS="$2"; shift 2 ;;
            --dry-run) DRY_RUN=1; shift ;;
            --inspect-state) INSPECT_STATE=1; shift ;;
            *) return 1 ;;
        esac
    done
    if [[ "$INSPECT_STATE" == 1 ]]; then
        [[ -n "$STATE_FILE" && -z "$ACCOUNT" && -z "$REMOTE_ROOT" && -z "$STAGING_ROOT" &&
           -z "$MODE" && -z "$URL" && -z "$REF" && -z "$EXPECTED_HEAD" &&
           -z "$EXPECTED_BRANCH" && -z "$PROXY_ENV" && -z "$TARGET" &&
           -z "$BUNDLE" && -z "$BUNDLE_SHA" && -z "$OPERATION_ID" &&
           "$DRY_RUN" == 0 ]]
        return $?
    fi
    [[ "$MODE" == clone || "$MODE" == fetch ]] || return 1
    [[ -n "$ACCOUNT" && -n "$REMOTE_ROOT" && -n "$STAGING_ROOT" && -n "$URL" &&
       -n "$REF" && -n "$EXPECTED_HEAD" && -n "$EXPECTED_BRANCH" &&
       -n "$PROXY_ENV" && -n "$TARGET" && -n "$OPERATION_ID" && -n "$STATE_FILE" ]]
}

reject_unsafe_path_text() {
    local value="$1"
    [[ "$value" == /* ]] || return 1
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* && "$value" != *$'\t'* ]] || return 1
    case "/$value/" in */../*|*/./*) return 1 ;; esac
    return 0
}

require_within() {
    local child="$1" root="$2"
    [[ "$child" == "$root"/* && "$child" != "$root" ]]
}

validate_account_roots() {
    local expected root_real staging_real
    case "$ACCOUNT" in
        scvi806) expected="/data/run01/scvi806" ;;
        t0s000727) expected="/publicfs01/fs1-t/home/t0s000727" ;;
        *) fail "unsupported account" "$RC_TARGET" ;;
    esac
    reject_unsafe_path_text "$REMOTE_ROOT" || fail "remote root is not a safe absolute path" "$RC_TARGET"
    [[ "$REMOTE_ROOT" == "$expected" || "$REMOTE_ROOT" == "$expected"/* ]] || fail "remote root does not match account" "$RC_TARGET"
    command -v realpath >/dev/null 2>&1 || fail "realpath command is required" "$RC_TARGET"
    [[ -d "$REMOTE_ROOT" && ! -L "$REMOTE_ROOT" ]] || fail "remote root is not a regular directory" "$RC_TARGET"
    root_real="$(realpath -e -- "$REMOTE_ROOT" 2>/dev/null)" || fail "remote root cannot be resolved" "$RC_TARGET"
    [[ "$root_real" == "$REMOTE_ROOT" ]] || fail "remote root must not traverse a symlink" "$RC_TARGET"

    reject_unsafe_path_text "$STAGING_ROOT" || fail "staging root is not a safe absolute path" "$RC_TARGET"
    [[ -d "$STAGING_ROOT" && ! -L "$STAGING_ROOT" ]] || fail "staging root is not a regular directory" "$RC_TARGET"
    staging_real="$(realpath -e -- "$STAGING_ROOT" 2>/dev/null)" || fail "staging root cannot be resolved" "$RC_TARGET"
    [[ "$staging_real" == "$STAGING_ROOT" ]] || fail "staging root must not traverse a symlink" "$RC_TARGET"
    require_within "$staging_real" "$root_real" || fail "staging root is outside remote root" "$RC_TARGET"
    REMOTE_ROOT="$root_real"
    STAGING_ROOT="$staging_real"
}

validate_target() {
    local target_real parent parent_real base
    reject_unsafe_path_text "$TARGET" || fail "target is not a safe absolute path" "$RC_TARGET"
    if [[ "$MODE" == clone ]]; then
        [[ ! -e "$TARGET" && ! -L "$TARGET" ]] || fail "clone target must not already exist" "$RC_TARGET"
        parent="$(dirname -- "$TARGET")"
        base="$(basename -- "$TARGET")"
        [[ -d "$parent" && ! -L "$parent" && "$base" != . && "$base" != .. ]] || fail "clone target parent is invalid" "$RC_TARGET"
        parent_real="$(realpath -e -- "$parent" 2>/dev/null)" || fail "clone target parent cannot be resolved" "$RC_TARGET"
        target_real="$parent_real/$base"
    else
        [[ -d "$TARGET" && ! -L "$TARGET" ]] || fail "fetch target is not a regular directory" "$RC_TARGET"
        target_real="$(realpath -e -- "$TARGET" 2>/dev/null)" || fail "fetch target cannot be resolved" "$RC_TARGET"
        git -C "$target_real" rev-parse --git-dir >/dev/null 2>&1 || fail "fetch target is not a Git worktree" "$RC_TARGET"
    fi
    require_within "$target_real" "$STAGING_ROOT" || fail "target is outside staging root" "$RC_TARGET"
    TARGET="$target_real"
}

validate_state_file() {
    local state_parent state_base state_parent_real
    reject_unsafe_path_text "$STATE_FILE" || fail "state file is not a safe absolute path" "$RC_TARGET"
    state_parent="$(dirname -- "$STATE_FILE")"
    state_base="$(basename -- "$STATE_FILE")"
    [[ -n "$state_base" && "$state_base" != . && "$state_base" != .. ]] || fail "state file name is invalid" "$RC_TARGET"
    [[ -d "$state_parent" && ! -L "$state_parent" ]] || fail "state file parent is invalid" "$RC_TARGET"
    state_parent_real="$(realpath -e -- "$state_parent" 2>/dev/null)" || fail "state file parent cannot be resolved" "$RC_TARGET"
    [[ "$state_parent_real" == "$state_parent" ]] || fail "state file parent must not traverse a symlink" "$RC_TARGET"
    [[ "$state_parent_real" == "$STAGING_ROOT" || "$state_parent_real" == "$STAGING_ROOT"/* ]] || fail "state file is outside staging root" "$RC_TARGET"
    STATE_FILE="$state_parent_real/$state_base"
    [[ "$STATE_FILE" != "$TARGET" && "$STATE_FILE" != "$TARGET"/* ]] || fail "state file is inside checkout target" "$RC_TARGET"
    if [[ -L "$STATE_FILE" ]]; then
        fail "state file must not be a symlink" "$RC_TARGET"
    fi
    [[ ! -e "$STATE_FILE" || -f "$STATE_FILE" ]] || fail "state file is not a regular file" "$RC_TARGET"
}

validate_inputs() {
    local expected_ref
    validate_account_roots
    validate_target
    validate_state_file
    [[ "$URL" != *$'\n'* && "$URL" != *$'\r'* && "$URL" != *$'\t'* ]] || fail "GitHub URL contains a control character" "$RC_URL"
    [[ "$URL" =~ ^https://github\.com/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(\.git)?$ ]] || fail "GitHub URL is not in the fixed safe form" "$RC_URL"
    [[ "$REF" =~ ^refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]*$ && "$REF" != *".."* && "$REF" != *"//"* && "$REF" != */ ]] || fail "Git ref is not a safe branch ref" "$RC_URL"
    [[ "$EXPECTED_BRANCH" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "$EXPECTED_BRANCH" != *".."* && "$EXPECTED_BRANCH" != *"//"* && "$EXPECTED_BRANCH" != */ ]] || fail "expected branch is not safe" "$RC_ARGS"
    expected_ref="refs/heads/$EXPECTED_BRANCH"
    [[ "$REF" == "$expected_ref" ]] || fail "Git ref and expected branch do not match" "$RC_ARGS"
    [[ "$EXPECTED_HEAD" =~ ^[0-9a-fA-F]+$ ]] || fail "expected head is not hexadecimal" "$RC_ARGS"
    (( ${#EXPECTED_HEAD} == 40 || ${#EXPECTED_HEAD} == 64 )) || fail "expected head must be a full SHA-1 or SHA-256" "$RC_ARGS"
    [[ "$OPERATION_ID" =~ ^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{32})$ ]] || fail "operation id is not a UUID" "$RC_ARGS"
    [[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "timeout-seconds must be a positive integer" "$RC_ARGS"
    (( TIMEOUT_SECONDS >= 1 && TIMEOUT_SECONDS <= 300 )) || fail "timeout-seconds must be between 1 and 300" "$RC_ARGS"
    reject_unsafe_path_text "$PROXY_ENV" || fail "proxy env path is not safe" "$RC_TARGET"
    [[ "$PROXY_ENV" == "$REMOTE_ROOT"/* ]] || fail "proxy env is outside remote root" "$RC_TARGET"
    if [[ -n "$BUNDLE" ]]; then
        reject_unsafe_path_text "$BUNDLE" || fail "bundle path is not safe" "$RC_BUNDLE"
        [[ "$BUNDLE" == "$REMOTE_ROOT"/* ]] || fail "bundle is outside remote root" "$RC_BUNDLE"
    fi
}

snapshot_checkout() {
    local status
    TARGET_HEAD="$(git -C "$TARGET" rev-parse HEAD 2>/dev/null)" || fail "target HEAD could not be read" "$RC_GIT"
    TARGET_BRANCH="$(git -C "$TARGET" symbolic-ref --short -q HEAD 2>/dev/null)" || fail "target is not on a named branch" "$RC_GIT"
    status="$(git -C "$TARGET" status --porcelain=v1 --untracked-files=all 2>/dev/null)" || fail "target status could not be read" "$RC_GIT"
    [[ "$TARGET_HEAD" == "$EXPECTED_HEAD" ]] || fail "target HEAD does not match expected head" "$RC_GIT"
    [[ "$TARGET_BRANCH" == "$EXPECTED_BRANCH" ]] || fail "target branch does not match expected branch" "$RC_GIT"
    [[ -z "$status" ]] || fail "target worktree is not clean" "$RC_GIT"
}

prepare_stable_bundle() {
    local bundle_real actual_sha heads
    BUNDLE_ERROR=""
    if [[ -z "$BUNDLE" || -z "$BUNDLE_SHA" ]]; then
        BUNDLE_ERROR="bundle and bundle SHA256 are required for fallback"
        return 1
    fi
    if [[ "$BUNDLE" != *.verified ]]; then
        BUNDLE_ERROR="only .verified bundles are accepted; .part bundles are not acquisition sources"
        return 1
    fi
    if [[ ! "$BUNDLE_SHA" =~ ^[0-9a-fA-F]{64}$ ]]; then
        BUNDLE_ERROR="bundle SHA256 is not a 64-character hexadecimal digest"
        return 1
    fi
    if [[ ! -f "$BUNDLE" || -L "$BUNDLE" ]]; then
        BUNDLE_ERROR="verified bundle is not a regular file"
        return 1
    fi
    bundle_real="$(realpath -e -- "$BUNDLE" 2>/dev/null)" || {
        BUNDLE_ERROR="verified bundle cannot be resolved"
        return 1
    }
    if [[ "$bundle_real" != "$BUNDLE" ]]; then
        BUNDLE_ERROR="verified bundle must not traverse a symlink"
        return 1
    fi
    STABLE_BUNDLE="$(mktemp "$STAGING_ROOT/.verified-bundle.XXXXXX")" || {
        BUNDLE_ERROR="could not create private verified bundle snapshot"
        return 1
    }
    if ! chmod 600 -- "$STABLE_BUNDLE" || ! cp -- "$bundle_real" "$STABLE_BUNDLE"; then
        BUNDLE_ERROR="could not snapshot verified bundle"
        return 1
    fi
    actual_sha="$(sha256sum -- "$STABLE_BUNDLE" 2>/dev/null | awk '{print $1}' || true)"
    if [[ "${actual_sha,,}" != "${BUNDLE_SHA,,}" ]]; then
        BUNDLE_ERROR="verified bundle SHA256 mismatch"
        return 1
    fi
    VERIFY_ROOT="$(mktemp -d "$STAGING_ROOT/.bundle-verify.XXXXXX")" || {
        BUNDLE_ERROR="could not create private bundle verification directory"
        return 1
    }
    if ! chmod 700 -- "$VERIFY_ROOT" || ! git -C "$VERIFY_ROOT" init --bare >/dev/null 2>&1; then
        BUNDLE_ERROR="could not initialize bundle verification directory"
        return 1
    fi
    if ! git -C "$VERIFY_ROOT" bundle verify "$STABLE_BUNDLE" >/dev/null 2>&1; then
        BUNDLE_ERROR="git bundle verify failed"
        return 1
    fi
    heads="$(git bundle list-heads "$STABLE_BUNDLE" 2>/dev/null || true)"
    if ! printf '%s\n' "$heads" | awk -v h="$EXPECTED_HEAD" -v r="$REF" '$1 == h && $2 == r { found=1 } END { exit(found ? 0 : 1) }'; then
        BUNDLE_ERROR="verified bundle refs do not contain expected ref and HEAD"
        return 1
    fi
    return 0
}

run_proxy_probe() {
    hpc_proxy_load "$PROXY_ENV" >/dev/null 2>&1 || return 1
    hpc_proxy_git_ls_remote "$URL" "$REF" "$EXPECTED_HEAD" "$TIMEOUT_SECONDS"
}

clone_from_source() {
    local source="$1" temporary_target status
    WORK_ROOT="$(mktemp -d "$STAGING_ROOT/.git-source.XXXXXX")" || return 1
    chmod 700 -- "$WORK_ROOT" || return 1
    temporary_target="$WORK_ROOT/repo"
    git clone --no-checkout --no-tags "$source" "$temporary_target" >/dev/null 2>&1 || return 1
    git -C "$temporary_target" fetch --no-tags --no-prune "$source" "$REF" >/dev/null 2>&1 || return 1
    FETCH_HEAD_VALUE="$(git -C "$temporary_target" rev-parse FETCH_HEAD 2>/dev/null)" || return 1
    [[ "$FETCH_HEAD_VALUE" == "$EXPECTED_HEAD" ]] || return 1
    git -C "$temporary_target" checkout --force -B "$EXPECTED_BRANCH" "$EXPECTED_HEAD" >/dev/null 2>&1 || return 1
    TARGET_HEAD="$(git -C "$temporary_target" rev-parse HEAD 2>/dev/null)" || return 1
    TARGET_BRANCH="$(git -C "$temporary_target" symbolic-ref --short -q HEAD 2>/dev/null)" || return 1
    status="$(git -C "$temporary_target" status --porcelain=v1 --untracked-files=all 2>/dev/null)" || return 1
    [[ "$TARGET_HEAD" == "$EXPECTED_HEAD" && "$TARGET_BRANCH" == "$EXPECTED_BRANCH" && -z "$status" ]] || return 1
    write_state checkout_verified || return 1
    # -T keeps a race-created destination from turning the rename into a
    # nested move; the validated checkout appears as one atomic target.
    mv -T -- "$temporary_target" "$TARGET" || return 1
    return 0
}

fetch_from_source() {
    local source="$1" before_head before_branch after_status
    snapshot_checkout
    before_head="$TARGET_HEAD"
    before_branch="$TARGET_BRANCH"
    git -C "$TARGET" fetch --no-tags --no-prune "$source" "$REF" >/dev/null 2>&1 || return 1
    FETCH_HEAD_VALUE="$(git -C "$TARGET" rev-parse FETCH_HEAD 2>/dev/null)" || return 1
    [[ "$FETCH_HEAD_VALUE" == "$EXPECTED_HEAD" ]] || return 1
    TARGET_HEAD="$(git -C "$TARGET" rev-parse HEAD 2>/dev/null)" || return 1
    TARGET_BRANCH="$(git -C "$TARGET" symbolic-ref --short -q HEAD 2>/dev/null)" || return 1
    after_status="$(git -C "$TARGET" status --porcelain=v1 --untracked-files=all 2>/dev/null)" || return 1
    [[ "$TARGET_HEAD" == "$before_head" && "$TARGET_BRANCH" == "$before_branch" && -z "$after_status" ]]
}

if ! parse_args "$@"; then
    ERRORS+=("required arguments are missing or invalid")
    emit_json false "rejected_or_failed"
    exit "$RC_ARGS"
fi

if [[ "$INSPECT_STATE" == 1 ]]; then
    if ! reject_unsafe_path_text "$STATE_FILE"; then
        printf '{"schema":"filament.hpc_git_acquisition.v2","state":"unknown_no_receipt","error":"state file path is unsafe"}\n'
        exit "$RC_TARGET"
    fi
    if [[ ! -f "$STATE_FILE" || -L "$STATE_FILE" ]]; then
        printf '{"schema":"filament.hpc_git_acquisition.v2","state":"unknown_no_receipt","state_file":"%s","error":"state receipt not found"}\n' "$(json_escape "$STATE_FILE")"
        exit "$RC_OUTPUT"
    fi
    inspect_real="$(realpath -e -- "$STATE_FILE" 2>/dev/null)" || {
        printf '{"schema":"filament.hpc_git_acquisition.v2","state":"unknown_no_receipt","error":"state receipt cannot be resolved"}\n'
        exit "$RC_OUTPUT"
    }
    if [[ "$inspect_real" != "$STATE_FILE" ]]; then
        printf '{"schema":"filament.hpc_git_acquisition.v2","state":"unknown_no_receipt","error":"state receipt must not traverse a symlink"}\n'
        exit "$RC_TARGET"
    fi
    cat -- "$STATE_FILE"
    exit $?
fi

validate_inputs

if [[ "$DRY_RUN" == 1 ]]; then
    SOURCE_CLASS="not_executed"
    emit_json true "dry_run"
    exit 0
fi

STATE_RECEIPT_READY=1
write_state started || fail "state receipt could not be initialized" "$RC_OUTPUT"

if [[ "$MODE" == fetch ]]; then
    snapshot_checkout
fi
write_state acquiring || fail "state receipt could not record acquisition" "$RC_OUTPUT"
if [[ -n "$BUNDLE" && "$BUNDLE" != *.verified ]]; then
    fail "only .verified bundles are accepted; .part bundles are not acquisition sources" "$RC_BUNDLE"
fi

if run_proxy_probe; then
    SOURCE_CLASS="strict_remote_verified"
    write_state acquiring || fail "state receipt could not record remote source" "$RC_OUTPUT"
    if [[ "$MODE" == clone ]]; then
        clone_from_source "$URL" || fail "strict proxy clone failed" "$RC_GIT"
        snapshot_checkout
    else
        fetch_from_source "$URL" || fail "strict proxy fetch failed" "$RC_GIT"
        write_state checkout_verified || fail "state receipt could not verify checkout" "$RC_OUTPUT"
    fi
    write_state completed || fail "state receipt could not record completion" "$RC_OUTPUT"
    emit_json true "completed"
    exit 0
fi

prepare_stable_bundle || fail "${BUNDLE_ERROR:-verified bundle fallback failed}" "$RC_BUNDLE"
SOURCE_CLASS="verified_bundle_non_strict"
write_state acquiring || fail "state receipt could not record bundle source" "$RC_OUTPUT"
if [[ "$MODE" == clone ]]; then
    clone_from_source "$STABLE_BUNDLE" || fail "bundle clone failed" "$RC_GIT"
    snapshot_checkout
else
    fetch_from_source "$STABLE_BUNDLE" || fail "bundle fetch failed" "$RC_GIT"
    write_state checkout_verified || fail "state receipt could not verify checkout" "$RC_OUTPUT"
fi
write_state completed || fail "state receipt could not record completion" "$RC_OUTPUT"
emit_json true "completed"
