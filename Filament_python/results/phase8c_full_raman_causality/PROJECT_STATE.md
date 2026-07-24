# Phase 8C-B — full Eq.27 Raman ON/OFF causal test

Prepared from `e591a2b89f9859169ea7c49ef24be3bb08006844` on branch `codex/phase8c-full-raman-causality`.

- Operational baseline: `f70c5f48dd11e6db2376604751c8b13afdc1cd2f`
- Frozen physical baseline: `e11d13f103c484953c0f733aa9b410bff385b2b5`
- Numerical admission evidence: A3 Job `180573`, completed `0:0`.
- Scheduler evidence: the `gpu` partition reported `MaxTime=UNLIMITED`; Test A requests `15:00:00` directly, with no checkpoint/restart.
- Authorized production jobs: exactly two — full Eq.27 Raman feedback ON and OFF.

The causal variable is only `propagation.use_raman_full_operator`. No performance optimization, profiling smoke, grid/step change, or non-Raman ablation is in scope.
