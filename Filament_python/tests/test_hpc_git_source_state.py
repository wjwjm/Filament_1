"""Focused state-receipt tests for protected Git acquisition."""

from __future__ import annotations

import json
from pathlib import Path

from test_hpc_execution_guardrails import _run_posix_stdin


ROOT = Path(__file__).resolve().parents[2]
HPC_OPS = ROOT / "Filament_python" / "tools" / "hpc_ops"


def test_git_source_state_receipt_and_verified_bundle_contract():
    fixture = r'''
set -euo pipefail
source_helper="$1"
proxy_helper="$2"
root=$(mktemp -d)
trap 'rm -rf -- "$root"' EXIT
account_root="$root/account"
remote_root="$account_root/user_Wangjimin"
staging="$remote_root/staging"
mkdir -p -- "$staging"

# Keep the production account mapping intact in the repository file while
# making this isolated fixture runnable without privileged /data access.
mkdir -- "$root/tool"
sed "s#/data/run01/scvi806#$account_root#g" "$source_helper" > "$root/tool/hpc_git_source.sh"
cp -- "$proxy_helper" "$root/tool/hpc_proxy_env.sh"
chmod 700 -- "$root/tool/hpc_git_source.sh"

repo="$remote_root/source"
mkdir -- "$repo"
git -C "$repo" init -b main >/dev/null
git -C "$repo" config user.email guardrail@example.invalid
git -C "$repo" config user.name Guardrail
printf 'fixture\n' > "$repo/tracked.txt"
git -C "$repo" add -- tracked.txt
git -C "$repo" commit -m fixture >/dev/null
head=$(git -C "$repo" rev-parse HEAD)

operation_id=11111111-1111-1111-1111-111111111111
bundle_part="$staging/source.bundle.part.$operation_id"
bundle="$staging/source.bundle.verified"
git -C "$repo" bundle create "$bundle_part" main >/dev/null
bundle_sha=$(sha256sum -- "$bundle_part" | awk '{print $1}')
# This models the uploader's successful SHA/ref verification and promotion.
mv -- "$bundle_part" "$bundle"

proxy="$remote_root/proxy.env"
printf '%s\n' 'http_proxy=http://proxy.example.invalid:8080' 'https_proxy=https://proxy.example.invalid:8443' > "$proxy"
chmod 600 -- "$proxy"

fakebin="$root/bin"
mkdir -- "$fakebin"
real_git=$(command -v git)
real_mv=$(command -v mv)
cat > "$fakebin/git" <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
    if [[ "$arg" == ls-remote ]]; then exit 1; fi
done
exec "$REAL_GIT" "$@"
EOF
cat > "$fakebin/mv" <<'EOF'
#!/usr/bin/env bash
destination=""
for arg in "$@"; do destination="$arg"; done
if [[ "$destination" == "$CHECK_TARGET" ]]; then
    grep -q '"state":"checkout_verified"' -- "$CHECK_STATE"
    test ! -e "$CHECK_TARGET"
fi
exec "$REAL_MV" "$@"
EOF
chmod 700 -- "$fakebin/git" "$fakebin/mv"
export REAL_GIT="$real_git" REAL_MV="$real_mv"
target="$staging/checkout"
state="$staging/acquisition.json"
export CHECK_TARGET="$target" CHECK_STATE="$state"
PATH="$fakebin:$PATH"

clone_json=$("$root/tool/hpc_git_source.sh" \
    --account scvi806 --remote-root "$remote_root" --staging-root "$staging" \
    --mode clone --url https://github.com/example/repo.git \
    --ref refs/heads/main --expected-head "$head" --expected-branch main \
    --proxy-env "$proxy" --target "$target" --bundle "$bundle" \
    --bundle-sha "$bundle_sha" --operation-id "$operation_id" \
    --state-file "$state" --timeout-seconds 1)
python3 - "$clone_json" "$state" "$head" "$operation_id" "$target" <<'PY'
import json, pathlib, sys

report = json.loads(sys.argv[1])
receipt = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert report["ok"] is True
assert report["source_class"] == "verified_bundle_non_strict"
assert report["operation_id"] == sys.argv[4]
assert receipt["schema"] == "filament.hpc_git_acquisition.v2"
assert receipt["state"] == "completed"
assert receipt["source_class"] == "verified_bundle_non_strict"
assert receipt["target"] == sys.argv[5]
assert receipt["target_head"] == sys.argv[3]
assert receipt["target_branch"] == "main"
assert receipt["fetch_head"] == sys.argv[3]
PY
test -z "$(git -C "$target" status --porcelain=v1 --untracked-files=all)"
if compgen -G "$state.tmp.*" >/dev/null; then exit 31; fi

before=$(stat -c '%Y' -- "$state")
inspected=$("$root/tool/hpc_git_source.sh" --inspect-state --state-file "$state")
after=$(stat -c '%Y' -- "$state")
test "$inspected" = "$(cat -- "$state")"
test "$before" = "$after"

# A .part bundle is never an acquisition source and remains untouched.
part_operation=22222222-2222-2222-2222-222222222222
part="$staging/source.bundle.part.$part_operation"
cp -- "$bundle" "$part"
part_before=$(sha256sum -- "$part" | awk '{print $1}')
part_target="$staging/part-checkout"
part_state="$staging/part.json"
set +e
"$root/tool/hpc_git_source.sh" \
    --account scvi806 --remote-root "$remote_root" --staging-root "$staging" \
    --mode clone --url https://github.com/example/repo.git \
    --ref refs/heads/main --expected-head "$head" --expected-branch main \
    --proxy-env "$proxy" --target "$part_target" --bundle "$part" \
    --bundle-sha "$bundle_sha" --operation-id "$part_operation" \
    --state-file "$part_state" --timeout-seconds 1 >/dev/null
part_rc=$?
set -e
test "$part_rc" -ne 0
test ! -e "$part_target"
test "$part_before" = "$(sha256sum -- "$part" | awk '{print $1}')"
python3 - "$part_state" <<'PY'
import json, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert receipt["state"] == "failed"
PY

# Missing state is an explicit unknown and must not trigger acquisition.
missing="$staging/missing.json"
missing_target="$staging/missing-checkout"
set +e
missing_output=$("$root/tool/hpc_git_source.sh" --inspect-state --state-file "$missing")
missing_rc=$?
set -e
test "$missing_rc" -ne 0
test ! -e "$missing"
test ! -e "$missing_target"
printf '%s\n' "$missing_output" | grep -q '"state":"unknown_no_receipt"'
printf 'git-source-state-fixture-ok\n'
'''
    result = _run_posix_stdin(
        fixture,
        HPC_OPS / "hpc_git_source.sh",
        HPC_OPS / "hpc_proxy_env.sh",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "git-source-state-fixture-ok"
