# Phase 6 postprocess correction changelog

Supersedes the Phase-6 postprocessing interpretation in the parent result directory while preserving all prior artifacts unchanged.

- Corrected `E_dep_cumulative/U0`: the old code used a dimensionless expression derived from `U_rel_change_z`; the corrected analysis uses each case's own `diagnostic_summary.json.metrics.U0_J`.
- Corrected peak collapse direction: it now tests `off_peak < 0.5 * full_peak` and records the ratio.
- Corrected tail criterion: it compares each tail's absolute and relative error to PyCAP rather than interpreting the full/off ratio as a paper-agreement criterion.
- Corrected missing PyCAP crossings: unavailable paper crossings are `null` with `not_available_in_pycap`, while full/off causal shifts remain available.
- Added the threshold, peak/width, feedback, numerical-path, config-diff, input-audit, and corrected decision artifacts required for formal closure.
- Classification remains `raman_phase_partially_supported`; the physical conclusion is unchanged in category but is now supported by corrected quantitative evidence.
