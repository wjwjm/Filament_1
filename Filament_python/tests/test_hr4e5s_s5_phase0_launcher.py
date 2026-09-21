"""Real-Bash regression for the S5-FINAL Phase-0 submission entry.

Only Slurm command boundaries are local adapters. The tracked shell launcher
and Python actor run unchanged and the adapters create, signal, and wait for
actual local processes.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_SHA = "204c8edc0ecfe781a9c98afbab08078114f2e9dd"


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    assert drive and resolved.is_absolute(), f"Windows absolute path required: {resolved}"
    return "/mnt/" + drive + "/" + "/".join(resolved.parts[1:])


def _run_bash(script: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="s5-phase0-") as temporary:
        path = Path(temporary) / "run.sh"
        path.write_text(script, encoding="utf-8", newline="\n")
        return subprocess.run(
            ["wsl.exe", "--", "bash", _wsl_path(path)],
            text=True, capture_output=True, check=False, timeout=90,
        )


def _fixture_prefix() -> str:
    return r"""set -euo pipefail
work="$(mktemp -d /tmp/s5-phase0-local.XXXXXX)"
trap 'rm -rf "$work"' EXIT
mock="$work/mock"
mkdir -p "$mock" "$work/run"
registry="$work/registry"
trace="$work/trace"
touch "$registry"
export MOCK_REGISTRY="$registry" MOCK_TRACE="$trace"
cat >"$mock/srun" <<'SRUN'
#!/usr/bin/env bash
set -euo pipefail
{ printf 'srun'; printf ' <%s>' "$@"; printf '\n'; } >>"$MOCK_TRACE"
while (($#)); do case "$1" in --exact|--exclusive|--ntasks=*|--cpus-per-task=*|--gres=none|--gpus-per-task=*) shift ;; *) break ;; esac; done
exec 9>"$MOCK_REGISTRY.lock"
flock 9
step="$(($(wc -l <"$MOCK_REGISTRY")+1))"
(
  export SLURM_STEP_ID="$step" SLURMD_NODENAME="$(hostname)"
  export SLURM_STEP_GPUS="${S5_MOCK_STEP_GPUS:-}" SLURM_STEP_GRES="${S5_MOCK_STEP_GRES:-}"
  exec "$@"
) &
child="$!"
printf '%s.%s %s\n' "$SLURM_JOB_ID" "$step" "$child" >>"$MOCK_REGISTRY"
flock -u 9
exec 9>&-
term() { kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 143; }
trap term TERM INT
wait "$child"
SRUN
cat >"$mock/scontrol" <<'SCONTROL'
#!/usr/bin/env bash
set -euo pipefail
test "$1" = listpids
query="$2"
while read -r step pid; do
  [[ -n "\${pid:-}" ]] || continue
  if [[ "$query" = "$SLURM_JOB_ID" || "$query" = "$step" ]] && kill -0 "$pid" 2>/dev/null; then
    printf '%s %s\n' "$pid" "$step"
  fi
done <"$MOCK_REGISTRY"
SCONTROL
cat >"$mock/scancel" <<'SCANCEL'
#!/usr/bin/env bash
set -euo pipefail
test "$1" = --signal=TERM
step="$2"
printf 'scancel <%s> <%s>\n' "$1" "$2" >>"$MOCK_TRACE"
pid="$(awk -v wanted="$step" '$1==wanted {print $2; exit}' "$MOCK_REGISTRY")"
test -n "$pid"
kill -TERM "$pid"
SCANCEL
chmod 700 "$mock/srun" "$mock/scontrol" "$mock/scancel"
export PATH="$mock:$PATH" PYTHON=/usr/bin/python3 SLURM_JOB_ID=700
export SLURMD_NODENAME="$(hostname)" S5_LOCAL_TEST=1
"""


def test_old_submission_fragment_reproduces_real_nested_quote_failure_in_bash():
    repo = _wsl_path(ROOT)
    script = _fixture_prefix() + f"""
repo={repo!r}
git -C "$repo" show {OLD_SHA}:Filament_python/tools/hr4e5s_s5_final_single_allocation.sbatch |
  sed -n '/^launch_phase0_probe() {{/,/^capture_listpids() {{/p' | sed '$d' >"$work/old_function.sh"
source "$work/old_function.sh"
mkdir -p "$work/run/identities" "$work/run/heartbeats"
set +e
launch_phase0_probe probe_target "$work/run/identities/target.json" "$work/run/heartbeats/target" 2>"$work/old.err"
pid="$!"
wait "$pid"
rc="$?"
set -e
test "$rc" -ne 0
test ! -e "$work/run/identities/target.json"
grep -F 'os.path.basename(out)+.' "$work/old.err"
printf '{{"old_bug_reproduced":true,"exit_code":%s}}\n' "$rc"
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["old_bug_reproduced"] is True


def test_tracked_phase0_launcher_runs_real_actors_and_records_direct_argv():
    phase0 = _wsl_path(ROOT / "tools" / "hr4e5s_s5_phase0.sh")
    actor = _wsl_path(ROOT / "tools" / "hr4e5s_s5_probe_actor.py")
    script = _fixture_prefix() + f"""
phase0={phase0!r}
actor={actor!r}
export RUN_ROOT="$work/run"
set +e
bash "$phase0"
phase0_rc="$?"
set -e
if [[ "$phase0_rc" -ne 0 ]]; then
  find "$RUN_ROOT" -type f -name '*.json' -exec sh -c 'echo "--- $1" >&2; cat "$1" >&2' sh {{}} \\; || true
  exit "$phase0_rc"
fi
/usr/bin/python3 - "$RUN_ROOT/phase0/phase0_observability_result.json" "$MOCK_TRACE" "$actor" <<'PY'
import json, sys
result=json.load(open(sys.argv[1], encoding='utf-8'))
trace=open(sys.argv[2], encoding='utf-8').read()
assert result['status']=='PASS' and result['signal_count']==1
assert ' <bash>' not in trace and sys.argv[3] in trace
assert '--actor> <probe_target>' in trace and '--actor> <probe_survivor>' in trace, trace
assert '--exact>' in trace and '--gres=none>' in trace and '--gpus-per-task=0>' not in trace, trace
assert 'scancel <--signal=TERM> <700.2>' in trace
print(json.dumps({{'actual_launcher_phase0_local_pass': True}}))
PY
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["actual_launcher_phase0_local_pass"] is True


def test_phase0_rejects_contaminated_intent_before_science_start():
    phase0 = _wsl_path(ROOT / "tools" / "hr4e5s_s5_phase0.sh")
    script = _fixture_prefix() + f"""
phase0={phase0!r}
export RUN_ROOT="$work/run"
mkdir "$RUN_ROOT/phase0"
touch "$RUN_ROOT/phase0/signal_intent.json"
set +e
bash "$phase0"
rc="$?"
set -e
test "$rc" -ne 0
test ! -e "$RUN_ROOT/scenario"
/usr/bin/python3 - "$RUN_ROOT/phase0_preflight_defect.json" "$RUN_ROOT/final_decision.json" <<'PY'
import json, sys
phase, final=(json.load(open(item, encoding='utf-8')) for item in sys.argv[1:])
assert phase['status']=='FAIL' and final['status']=='READY_FOR_S5_FINAL_DEFECT_REVIEW'
print(json.dumps({{'phase0_fail_blocks_science': True}}))
PY
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["phase0_fail_blocks_science"] is True


def test_phase0_rejects_probe_step_gpu_gres_before_signal_or_science():
    phase0 = _wsl_path(ROOT / "tools" / "hr4e5s_s5_phase0.sh")
    script = _fixture_prefix() + f"""
phase0={phase0!r}
export RUN_ROOT="$work/run" S5_MOCK_STEP_GRES='gpu:1'
set +e
bash "$phase0"
rc="$?"
set -e
test "$rc" -eq 70
test ! -e "$RUN_ROOT/phase0/signal_intent.json"
test ! -e "$RUN_ROOT/scenario"
/usr/bin/python3 - "$RUN_ROOT/phase0/phase0_observability_result.json" <<'PY'
import json, sys
result=json.load(open(sys.argv[1], encoding='utf-8'))
assert result['status']=='FAIL' and result['reason']=='PROBE_GPU_GRES_PRESENT'
print(json.dumps({{'probe_gres_hard_gate_pass': True}}))
PY
"""
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip())["probe_gres_hard_gate_pass"] is True
