# Current-observability Popruzhenko 120 fs baseline

This directory archives the lightweight, post-run evidence for the one authorized
`120fs_popruzhenko_full_model_current_observability` run.  It intentionally does
not contain the raw NPZ, MAT, ionization-LUT cache, or source archive.

## Execution identity

- Slurm job: `175050` (`COMPLETED`, exit code `0:0`, 2026-07-17 16:36:28 to
  19:18:36 cluster-local time).
- Execution Git SHA: `8dcd01ee38adf2167a2fd6083ae4785e94de89a0`.
- Configuration snapshot SHA256:
  `9c2ce3786eff4ff30ee201f939a5a3430b7d1d6517b27be68af898b2367d1f37`.
- Raw NPZ: retained only on the authorized remote run root; SHA256
  `2bd1606c455bef7c8439b6b05aca043a52f655f15c17e4eaee5dce311778d61f`
  (27,440,996 bytes).

## Formal baseline gate

`baseline_reaudit.json` reports `formal_baseline_gate = passed`.

- The propagation reached `z = 1.2999999523 m` in 15,000 records.
- Every required z-history is finite and aligned to `z_axis`.
- The N2 and O2 peak-density histories are non-negative and remain below their
  neutral-density bounds (`2.0e25` and `5.0e24 m^-3`, respectively).
- Raw/applied plasma phase and ionization-absorption histories are both present;
  their maximum absolute differences are zero for this baseline.
- Deposited energy and input-energy loss are non-decreasing.  The final relative
  pulse-energy change is `-5.61297%`; the diagnostic plot found no energy growth.
- Actual `dz_used_z` ranges from `5e-5` to `1e-4 m`; cumulative rejection and
  safety-mode trigger counters remain zero, as recorded by the live propagator.

The CSV provides the complete z-resolved lightweight diagnostic history in the
fixed coordinate convention `x_focus_cm = 100 * (z_m - 0.95)`.

## Contents

- `baseline_reaudit.json`, `baseline_reaudit_report.md`, and
  `baseline_axial_diagnostics.csv`: remote validator outputs.
- `run_metadata.json` and the configuration snapshot: execution provenance.
- `figures/`: remotely generated diagnostic PNGs and summary JSON.
- `slurm_log_summary.md`: concise, hash-linked Slurm evidence; the complete
  stdout/stderr remains only in the authorized remote run root.
- `archive_manifest.json`: hashes of all committed lightweight artifacts.
