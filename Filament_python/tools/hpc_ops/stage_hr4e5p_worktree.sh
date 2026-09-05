#!/usr/bin/env bash
set -euo pipefail
readonly BASE_REPO="$1" BUNDLE="$2" EXPECTED_SHA="$3" NEW_REPO="$4" NEW_BRANCH="$5"
test -d "$BASE_REPO" && test -f "$BUNDLE" && test ! -e "$NEW_REPO"
test "$(git -C "$BASE_REPO" rev-parse HEAD)" = "17431cf09c127a657d47dd9729f0a1d819dcd2fb"
test -z "$(git -C "$BASE_REPO" status --porcelain=v1 --untracked-files=all)"
git -C "$BASE_REPO" bundle verify "$BUNDLE" >/dev/null
git -C "$BASE_REPO" fetch "$BUNDLE" "refs/heads/HR-4E:refs/heads/$NEW_BRANCH" >/dev/null
test "$(git -C "$BASE_REPO" rev-parse "refs/heads/$NEW_BRANCH")" = "$EXPECTED_SHA"
git -C "$BASE_REPO" worktree add "$NEW_REPO" "$NEW_BRANCH" >/dev/null
test "$(git -C "$NEW_REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$NEW_REPO" status --porcelain=v1 --untracked-files=all)"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","repo":"%s","sha":"%s"}\n' "$NEW_REPO" "$EXPECTED_SHA"
