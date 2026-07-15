# Transverse-profile validation results

This directory is a lightweight, versioned archive of the completed 120 fs and 40 fs Gaussian-versus-FT90 controlled comparisons.

Each run directory contains only:

- the stage report (`reports/`);
- comparison PNG/CSV/JSON artifacts (`comparison/`);
- per-case diagnostic PNGs and JSON summaries (`cases/*/figures/`).

The large raw NPZ and MAT arrays are deliberately excluded. They remain reproducible from the tracked stage configurations and are retained on the supercomputer under the corresponding `outputs/` run directory.

| Run | Pulse width | Simulation jobs | Postprocess job |
| --- | ---: | --- | --- |
| `profile_validation_20260715_004` | 120 fs | 170696, 170697 | 170698 |
| `profile_validation_40fs_20260715_001` | 40 fs | 170912, 170913 | 170914 |
