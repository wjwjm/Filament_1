#!/usr/bin/env bash
# Load and probe the HPC GitHub proxy without evaluating the env file.
#
# This file is safe to source from a trusted shell because it only defines
# hpc_proxy_load and hpc_proxy_git_ls_remote.  The secret file itself is read
# line by line; it is never sourced, eval'd, or printed.

set -o pipefail

HPC_PROXY_ENV_RC_INVALID_ARGS=64
HPC_PROXY_ENV_RC_FILE=65
HPC_PROXY_ENV_RC_SYNTAX=66
HPC_PROXY_ENV_RC_URL=67
HPC_PROXY_ENV_RC_PROBE=68

hpc_proxy_fail() {
    HPC_PROXY_ERROR="$1"
    return "${2:-$HPC_PROXY_ENV_RC_SYNTAX}"
}

hpc_proxy_validate_url() {
    local value="$1" remainder
    case "$value" in
        http://*|https://*) ;;
        *) return 1 ;;
    esac
    remainder="${value#*://}"
    [[ -n "$remainder" && "$remainder" != /* ]] || return 1
    # A proxy URL is data, not shell source.  Reject control and shell syntax.
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || return 1
    [[ "$value" != *';'* && "$value" != *'`'* ]] || return 1
    [[ "$value" != *'$('* && "$value" != *')'* ]] || return 1
    [[ "$value" != *'<'* && "$value" != *'>'* && "$value" != *'|'* ]] || return 1
    [[ "$value" != *'&'* ]] || return 1
    [[ "$value" != *$'\t'* && "$value" != *' '* ]] || return 1
    return 0
}

hpc_proxy_file_security() {
    local file="$1" mode owner uid
    [[ -f "$file" && ! -L "$file" ]] || return "$HPC_PROXY_ENV_RC_FILE"
    mode="$(stat -c '%a' -- "$file" 2>/dev/null || stat -f '%Lp' -- "$file" 2>/dev/null || true)"
    [[ "$mode" == "600" ]] || return "$HPC_PROXY_ENV_RC_FILE"
    if owner="$(stat -c '%u' -- "$file" 2>/dev/null)"; then
        :
    elif owner="$(stat -f '%u' -- "$file" 2>/dev/null)"; then
        :
    else
        return "$HPC_PROXY_ENV_RC_FILE"
    fi
    uid="$(id -u)"
    [[ -n "$owner" && "$owner" == "$uid" ]] || return "$HPC_PROXY_ENV_RC_FILE"
    return 0
}

hpc_proxy_load() {
    local file="${1:-}" line key value trimmed
    local http_value="" https_value=""
    local saw_http=0 saw_https=0
    HPC_PROXY_ERROR=""
    if [[ -z "$file" ]]; then
        hpc_proxy_fail "proxy env path is required" "$HPC_PROXY_ENV_RC_INVALID_ARGS"
        return "$HPC_PROXY_ENV_RC_INVALID_ARGS"
    fi
    if ! hpc_proxy_file_security "$file"; then
        hpc_proxy_fail "proxy env file must be a user-owned regular file with mode 600" "$HPC_PROXY_ENV_RC_FILE"
        return "$HPC_PROXY_ENV_RC_FILE"
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        trimmed="${line#${line%%[![:space:]]*}}"
        [[ -z "$trimmed" || "$trimmed" == \#* ]] && continue
        if [[ "$trimmed" == *$'\r'* || "$trimmed" == *$'\n'* ]]; then
            hpc_proxy_fail "proxy env contains a control character" "$HPC_PROXY_ENV_RC_SYNTAX"
            return "$HPC_PROXY_ENV_RC_SYNTAX"
        fi
        if [[ "$trimmed" == *';'* || "$trimmed" == *'`'* || "$trimmed" == *'$('* ]]; then
            hpc_proxy_fail "proxy env contains shell syntax" "$HPC_PROXY_ENV_RC_SYNTAX"
            return "$HPC_PROXY_ENV_RC_SYNTAX"
        fi
        if [[ "$trimmed" =~ ^(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            key="${BASH_REMATCH[2]}"
            value="${BASH_REMATCH[3]}"
        else
            hpc_proxy_fail "proxy env has an invalid assignment" "$HPC_PROXY_ENV_RC_SYNTAX"
            return "$HPC_PROXY_ENV_RC_SYNTAX"
        fi
        if [[ "$value" == \"*\" && "$value" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        case "$key" in
            http_proxy|HTTP_PROXY)
                if [[ "$saw_http" != 0 ]]; then
                    hpc_proxy_fail "duplicate http_proxy assignment" "$HPC_PROXY_ENV_RC_SYNTAX"
                    return "$HPC_PROXY_ENV_RC_SYNTAX"
                fi
                http_value="$value"; saw_http=1 ;;
            https_proxy|HTTPS_PROXY)
                if [[ "$saw_https" != 0 ]]; then
                    hpc_proxy_fail "duplicate https_proxy assignment" "$HPC_PROXY_ENV_RC_SYNTAX"
                    return "$HPC_PROXY_ENV_RC_SYNTAX"
                fi
                https_value="$value"; saw_https=1 ;;
            *)
                hpc_proxy_fail "proxy env contains an unsupported key" "$HPC_PROXY_ENV_RC_SYNTAX"
                return "$HPC_PROXY_ENV_RC_SYNTAX" ;;
        esac
    done < "$file"

    if [[ "$saw_http" != 1 || "$saw_https" != 1 ]]; then
        hpc_proxy_fail "both http_proxy and https_proxy are required" "$HPC_PROXY_ENV_RC_SYNTAX"
        return "$HPC_PROXY_ENV_RC_SYNTAX"
    fi
    if ! hpc_proxy_validate_url "$http_value"; then
        hpc_proxy_fail "http_proxy is not a valid HTTP(S) URL" "$HPC_PROXY_ENV_RC_URL"
        return "$HPC_PROXY_ENV_RC_URL"
    fi
    if ! hpc_proxy_validate_url "$https_value"; then
        hpc_proxy_fail "https_proxy is not a valid HTTP(S) URL" "$HPC_PROXY_ENV_RC_URL"
        return "$HPC_PROXY_ENV_RC_URL"
    fi

    # These are exported only into the current shell/process.  Values are
    # intentionally never included in status output.
    export http_proxy="$http_value" https_proxy="$https_value"
    export HTTP_PROXY="$http_value" HTTPS_PROXY="$https_value"
    export GIT_TERMINAL_PROMPT=0
    return 0
}

hpc_proxy_git_ls_remote() {
    local url="${1:-}" ref="${2:-}" expected_head="${3:-}"
    local timeout_seconds="${4:-}"
    local err_file rc remote_output line line_count=0 match_count=0 sha got_ref extra
    [[ -n "$url" && -n "$ref" && -n "$expected_head" ]] || return "$HPC_PROXY_ENV_RC_INVALID_ARGS"
    [[ "$expected_head" =~ ^[0-9a-fA-F]+$ ]] || return "$HPC_PROXY_ENV_RC_INVALID_ARGS"
    (( ${#expected_head} == 40 || ${#expected_head} == 64 )) || return "$HPC_PROXY_ENV_RC_INVALID_ARGS"
    [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || return "$HPC_PROXY_ENV_RC_INVALID_ARGS"
    (( timeout_seconds >= 1 && timeout_seconds <= 300 )) || return "$HPC_PROXY_ENV_RC_INVALID_ARGS"
    command -v timeout >/dev/null 2>&1 || return "$HPC_PROXY_ENV_RC_PROBE"
    export GIT_TERMINAL_PROMPT=0
    err_file="$(mktemp)" || return "$HPC_PROXY_ENV_RC_PROBE"
    remote_output="$(timeout --signal=TERM "${timeout_seconds}s" git -c credential.interactive=never ls-remote --exit-code "$url" "$ref" 2>"$err_file")"
    rc=$?
    rm -f -- "$err_file"
    [[ "$rc" == 0 ]] || return "$HPC_PROXY_ENV_RC_PROBE"
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        line_count=$((line_count + 1))
        IFS=$'\t' read -r sha got_ref extra <<< "$line"
        if [[ -z "${extra:-}" && "$sha" == "$expected_head" && "$got_ref" == "$ref" ]]; then
            match_count=$((match_count + 1))
        fi
    done <<< "$remote_output"
    [[ "$line_count" == 1 && "$match_count" == 1 ]] || return "$HPC_PROXY_ENV_RC_PROBE"
    return 0
}

hpc_proxy_json_status() {
    local loaded="$1" probe="$2"
    printf '{"schema":"filament.hpc_proxy_env.v1","loaded":%s,"probe":"%s"}\n' "$loaded" "$probe"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "$#" -lt 1 || "$#" -gt 6 ]]; then
        hpc_proxy_json_status false "invalid_args"
        exit "$HPC_PROXY_ENV_RC_INVALID_ARGS"
    fi
    hpc_proxy_load "$1"
    load_rc=$?
    if [[ "$load_rc" != 0 ]]; then
        hpc_proxy_json_status false "not_run"
        exit "$load_rc"
    fi
    if [[ ("$#" == 5 || "$#" == 6) && "$2" == "--probe" ]]; then
        if hpc_proxy_git_ls_remote "$3" "$4" "$5" "${6:-${HPC_PROXY_GIT_TIMEOUT_SECONDS:-30}}"; then
            hpc_proxy_json_status true "passed"
            exit 0
        fi
        hpc_proxy_json_status true "failed"
        exit "$HPC_PROXY_ENV_RC_PROBE"
    fi
    if [[ "$#" != 1 ]]; then
        hpc_proxy_json_status false "invalid_args"
        exit "$HPC_PROXY_ENV_RC_INVALID_ARGS"
    fi
    hpc_proxy_json_status true "not_requested"
fi
