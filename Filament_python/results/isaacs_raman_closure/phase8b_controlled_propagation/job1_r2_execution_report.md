# Phase 8B-R Task R2 — Job 1 execution audit

- Case: `120fs_talebpour_isaacs_full_operator_on`
- Locked source SHA: `aad67a100fba612789dba2e41e39fadf04219d63`
- Slurm Job 1: `179706`, `COMPLETED 0:0`, elapsed `03:46:38`
- Backend: CuPy on NVIDIA GeForce RTX 5090; total device memory was `33,668,988,928` bytes.
- Final archive: `z=1.2999999523162842 m`, 15,000 records, no NaN/Inf.
- Raw NPZ/MAT/LUT files were not copied to or committed in Git.

## Passed checks

- Full `full_isaacs_eq27` feedback was enabled; legacy Raman phase and absorption were disabled.
- `legacy alpha_R = 0` exactly.
- Strang used two Raman operator substeps and four Raman convolutions per propagation step.
- Raman per-step closure p99 was `1.990885304985568e-4`, below `1e-3`.
- Raman cumulative closure was `2.245331188532873e-6`, below `5e-3`.
- Near-focus total-energy residual was `1.2044126022869913%`, below `2%`.

## Failed contract and stop condition

The post-run diagnostic audit is **failed**.  The final total-energy residual is
`1.4176516743495496%`, exceeding the required `<1%` contract.  This is a
physical/accounting acceptance failure and blocks Phase 8B-R from proceeding.

The audit also reports two coordinate checks as failed: the float32 archive has
`|z_final - 1.3 m| = 4.768371586472142e-8 m` and a reconstructed-axis residual
of `9.23209881875664e-8 m`, while the current audit tolerances are `2e-12 m` or
tighter.  These are recorded as an audit-tolerance mismatch distinct from the
total-energy failure; they do not erase the total-energy failure.

No feedback-OFF Job 2 was prepared or submitted.  Phase 8B-R is stopped pending
a resolution of the failed Job 1 audit and separate subsequent authorization.
