#!/usr/bin/env bash
# Create one clean non-overwriting E3 worktree from a verified incremental bundle.
set -euo pipefail
readonly REPO="$1" BUNDLE="$2" EXPECTED_SHA="$3" WORKTREE="$4" BRANCH="$5"
test -d "$REPO" && test -f "$BUNDLE" && test ! -e "$WORKTREE"
case "$BRANCH" in
  *[!A-Za-z0-9._/-]*|*..*|*//*|*/) exit 64 ;;
esac
git -C "$REPO" fetch "$BUNDLE" refs/heads/HR-4E:refs/heads/HR-4E-e3-source >/dev/null
test "$(git -C "$REPO" rev-parse refs/heads/HR-4E-e3-source)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" show-ref --verify --hash "refs/heads/$BRANCH" || true)"
git -C "$REPO" worktree add -b "$BRANCH" "$WORKTREE" "$EXPECTED_SHA" >/dev/null
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test "$(git -C "$WORKTREE" symbolic-ref --short HEAD)" = "$BRANCH"
test -z "$(git -C "$WORKTREE" status --porcelain=v1 --untracked-files=all)"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","worktree":"%s","branch":"%s"}\n' "$WORKTREE" "$BRANCH"
