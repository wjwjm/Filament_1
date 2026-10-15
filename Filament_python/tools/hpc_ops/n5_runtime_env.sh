#!/usr/bin/env bash
# Source this file after RUN_ROOT is set and before any project Python import.
# It binds every coordinator, worker, health child, and probe to one private,
# project-storage-backed runtime cache/temp contract.
n5_runtime_env_fail() {
  printf 'N5 runtime environment setup failed: %s\n' "$1" >&2
  return 1
}

if [[ -z "$RUN_ROOT" || "$RUN_ROOT" != /* ]]; then
  n5_runtime_env_fail 'RUN_ROOT must be an absolute path'
  return 1
fi
if [[ ! -d "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
  n5_runtime_env_fail 'RUN_ROOT must be an existing non-symlink directory'
  return 1
fi
_n5_run_real="$(realpath -e -- "$RUN_ROOT")" || {
  n5_runtime_env_fail 'RUN_ROOT cannot be resolved'
  return 1
}
if [[ "$_n5_run_real" != "$RUN_ROOT" ]]; then
  n5_runtime_env_fail 'RUN_ROOT must already be canonical'
  return 1
fi

_n5_entry_root="$RUN_ROOT""_entry"
if [[ -L "$_n5_entry_root" ]]; then
  n5_runtime_env_fail 'entry root may not be a symlink'
  return 1
fi
if [[ ! -e "$_n5_entry_root" ]]; then
  mkdir -m 700 -- "$_n5_entry_root" || {
    n5_runtime_env_fail 'could not create private entry root'
    return 1
  }
fi
if [[ ! -d "$_n5_entry_root" || "$(realpath -e -- "$_n5_entry_root")" != "$_n5_entry_root" ]]; then
  n5_runtime_env_fail 'entry root must resolve to its canonical directory'
  return 1
fi
_n5_uid="$(id -u)"
_n5_entry_owner="$(stat -c '%u' -- "$_n5_entry_root")"
_n5_entry_mode="$(stat -c '%a' -- "$_n5_entry_root")"
_n5_run_dev="$(stat -c '%d' -- "$RUN_ROOT")"
_n5_entry_dev="$(stat -c '%d' -- "$_n5_entry_root")"
if [[ "$_n5_entry_owner" != "$_n5_uid" || "$_n5_entry_mode" != '700' || "$_n5_entry_dev" != "$_n5_run_dev" ]]; then
  n5_runtime_env_fail 'entry root must be owned, mode 700, and on RUN_ROOT storage'
  return 1
fi

N5_ENTRY_ROOT="$_n5_entry_root"
N5_RUNTIME_ENV_ROOT="$N5_ENTRY_ROOT/runtime_env"
if [[ -L "$N5_RUNTIME_ENV_ROOT" ]]; then
  n5_runtime_env_fail 'runtime environment root may not be a symlink'
  return 1
fi
if [[ ! -e "$N5_RUNTIME_ENV_ROOT" ]]; then
  mkdir -m 700 -- "$N5_RUNTIME_ENV_ROOT" || {
    n5_runtime_env_fail 'could not create private runtime environment root'
    return 1
  }
fi
if [[ ! -d "$N5_RUNTIME_ENV_ROOT" || "$(realpath -e -- "$N5_RUNTIME_ENV_ROOT")" != "$N5_RUNTIME_ENV_ROOT" ]]; then
  n5_runtime_env_fail 'runtime environment root must resolve canonically'
  return 1
fi

_n5_uid="$(id -u)"
_n5_owner="$(stat -c '%u' -- "$N5_RUNTIME_ENV_ROOT")"
_n5_mode="$(stat -c '%a' -- "$N5_RUNTIME_ENV_ROOT")"
_n5_env_dev="$(stat -c '%d' -- "$N5_RUNTIME_ENV_ROOT")"
if [[ "$_n5_owner" != "$_n5_uid" || "$_n5_mode" != '700' || "$_n5_env_dev" != "$_n5_run_dev" ]]; then
  n5_runtime_env_fail 'runtime environment root must be owned, mode 700, and on RUN_ROOT storage'
  return 1
fi

for _n5_name in home xdg_cache cupy_cache tmp; do
  _n5_dir="$N5_RUNTIME_ENV_ROOT/$_n5_name"
  if [[ -L "$_n5_dir" ]]; then
    n5_runtime_env_fail "runtime directory may not be a symlink: $_n5_name"
    return 1
  fi
  if [[ ! -e "$_n5_dir" ]]; then
    mkdir -m 700 -- "$_n5_dir" || {
      n5_runtime_env_fail "could not create runtime directory: $_n5_name"
      return 1
    }
  fi
  if [[ ! -d "$_n5_dir" || "$(realpath -e -- "$_n5_dir")" != "$_n5_dir" ]]; then
    n5_runtime_env_fail "runtime directory must resolve canonically: $_n5_name"
    return 1
  fi
  _n5_owner="$(stat -c '%u' -- "$_n5_dir")"
  _n5_mode="$(stat -c '%a' -- "$_n5_dir")"
  _n5_dev="$(stat -c '%d' -- "$_n5_dir")"
  if [[ "$_n5_owner" != "$_n5_uid" || "$_n5_mode" != '700' || "$_n5_dev" != "$_n5_run_dev" ]]; then
    n5_runtime_env_fail "runtime directory must be owned, mode 700, and on RUN_ROOT storage: $_n5_name"
    return 1
  fi
done

export N5_ENTRY_ROOT
export N5_RUNTIME_ENV_ROOT
export HOME="$N5_RUNTIME_ENV_ROOT/home"
export XDG_CACHE_HOME="$N5_RUNTIME_ENV_ROOT/xdg_cache"
export CUPY_CACHE_DIR="$N5_RUNTIME_ENV_ROOT/cupy_cache"
export TMPDIR="$N5_RUNTIME_ENV_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"