# E5-1A unrelated targeted-failure disposition

Date: 2026-09-20

The starting HEAD `46e56e8e19dfd9328b9cccdb669192556386beb1` records the
full targeted outcome as 381 passed, 3 skipped and 4 failed. This task does not
modify any implicated implementation or test file. The committed
`E5_1A_R_TEST_RESULTS.json` at that starting HEAD is the available direct
pre-task evidence; no separate clean-base rerun was performed.

| Test | Failure type | Available pre-existing evidence | E5-1A changed implicated files? | Classification | Closeout disposition |
|---|---|---|---|---|---|
| `test_preflight_bundle_fallback_via_wsl_fixture` | WSL/Git bundle fixture returned 69 because the expected bundle ref/head was not recognized | Same failure is recorded in the starting-HEAD targeted receipt | No changes to `test_hpc_execution_guardrails.py` or `tools/hpc_ops` from repair base through starting HEAD | `UNRELATED_WSL_GIT_FIXTURE_ENVIRONMENT` | Retained; not fixed in E5-1A |
| `test_git_source_bundle_clone_fetch_and_wrong_head_cleanup` | WSL launch failed with `WSL_E_USER_NOT_FOUND` | Same failure is recorded in the starting-HEAD targeted receipt | No changes to the implicated guardrail test or Git-source helpers | `UNRELATED_WSL_GIT_FIXTURE_ENVIRONMENT` | Retained; requires environment/fixture work outside E5-1A |
| `test_git_source_state_receipt_and_verified_bundle_contract` | WSL launch failed with `WSL_E_USER_NOT_FOUND` | Same failure is recorded in the starting-HEAD targeted receipt | No changes to `test_hpc_git_source_state.py` or Git-source helpers | `UNRELATED_WSL_GIT_FIXTURE_ENVIRONMENT` | Retained; requires environment/fixture work outside E5-1A |
| `test_region_energy_splits_intervals_conservatively_at_focus_boundaries` | Strict equality expected `3.0`; implementation returned `3.0000000000000004` | Same failure is recorded in the starting-HEAD targeted receipt | No changes to the test or HR2E localization implementation | `UNRELATED_STRICT_FLOAT_EQUALITY` | Retained; do not alter numerical semantics in this closeout |

These failures keep the full targeted command at nonzero exit status. They do
not invalidate the focused E5-1A production, recovery, overlap, Streaming/HR4D,
backend, sanity or compileall checks. “Pre-existing” here means present in the
committed starting-HEAD evidence; it is not a claim based on a newly executed
historical-base comparison.
