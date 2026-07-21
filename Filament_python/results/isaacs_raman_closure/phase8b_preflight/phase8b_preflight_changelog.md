# Phase 8B-P changelog

- Task P1: copied the Phase 6 production baseline, introduced the explicit full-operator switch, and generated single-factor config diffs.
- Task P2: wired full-operator Raman diagnostics and reused each Heun stage convolution for RHS and Eq.10 energy accounting.
- Task P3: implemented opt-in nonlinear split ordering, validated Strang composition, and selected `strang` for the formal configs.
- Task P4: added performance instrumentation and ran two strictly serial 20-step full-grid Slurm smokes; no full propagation was run.
- Task P5: defined the machine-readable production diagnostic and energy contract plus the completed-run auditor.
- Task P6: regenerated all preflight gates and reports and required the complete local pytest result for full-job submission admission.

No production Raman parameters, non-Raman physics, PyCAP data, or Phase 5-8A.1 historical results were changed. No raw NPZ/MAT/LUT file was committed.
