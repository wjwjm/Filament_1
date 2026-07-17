# Phase 5 baseline audit: formal gate failed

## Decision

`120fs_talebpour_full_model` was **not generated or submitted**. The existing FT90 Popruzhenko 120 fs run completed normally, but it cannot serve as the formal Phase-5 causal baseline.

## What passed

- The 120 fs candidate is job `170697`; its 40 fs companion is job `170913`.
- Both NPZ files reach `z_max = 1.3 m`, have 15,000 strictly increasing axial samples, and contain no numeric NaN/Inf values.
- Both preserve the required 17 GW, 0.95 m, FT90 `(R=1.979 mm, edge=0.9R)` and Popruzhenko-LUT physical settings.
- Their existing diagnostic summaries report successful legacy quality gates.

## Blocking evidence

The legacy NPZ schema does not contain the Phase-5-required shared diagnostics:

- `E_dep_cumulative_z`, `U_rel_change_z`;
- raw and applied plasma-phase histories;
- raw and applied ionization-absorption histories;
- N2/O2 per-species axial density summaries and O2 fraction at the total-density maximum;
- actual accepted axial-step history, adaptive rejection/shrink counts, and safety-mode event summary.

The historical `run_metadata.json` also does not record the code Git SHA used to create the files. The currently observed remote repository SHA cannot establish the historical execution revision. Although the old null switch fields resolve to the legacy full-model defaults, this does not cure the missing observability or provenance.

## Required next decision

Task 0 requires stopping before a Talebpour comparison when the Popruzhenko baseline cannot pass the formal audit. To resume, authorize exactly one current-observability Popruzhenko 120 fs baseline rerun, or explicitly relax the mandatory Phase-5 comparison diagnostics. No Talebpour, 40 fs, or O2-off Slurm job has been submitted.
