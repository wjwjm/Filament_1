# Complete Isaacs Eq. (27) candidate postprocess

Gate: **passed**.

Candidate staging provenance class: `verified_bundle_non_strict`. This verified-bundle source does not establish a direct GitHub remote push/fetch verification.

The candidate is required to use `full_isaacs_eq27_complete`, with the full complex Eq. (27) electronic and rotational RHS, no legacy Raman absorption, and fixed x_focus_cm = 100 * (z_m - 0.95).

## Checks

- Complete operator mode and semantic strings passed.
- Operator-applied, energy, adaptive-step, safety, dz, finite-value, and Raman-closure checks passed.
- Raw NPZ remains outside the repository; only CSV and audit artifacts are written here.
