# Historical f_R-mixture Raman static audit

Phase semantics and operator path across the four reference points:

| point | phase semantics | kernel | IIR | f_R in phase | n_R in phase | self-steepening order | absorption w_R |
|---|---|---|---|---|---|---|---|
| c34c3a | historical_fr_mixture | area_normalized | S=rS+cI[n]; IR=Im(kS) | True | False | shock(I_nl) then *n2 | n0/c0 |
| 4c330ac | historical_fr_mixture | area_normalized | S=rS+cI[n]; IR=Im(kS) | True | False | shock(I_nl) then *n2 | n0/c0 |
| 037ead0 | explicit_n2_elec_I_plus_n_R_IR | analytic_prefactor | S=rS+cI[n]; IR=Im(kS) | False | True | shock(summed delta_n) | n_R/c0 |
| e11d13f | explicit_n2_elec_I_plus_n_R_IR | analytic_prefactor | S=rS+cI[n]; IR=Im(kS) | False | True | shock(summed delta_n) | n_R/c0 |

## Assertions

- `legacy_split_is_explicit_nR_IR_not_fr_mixture`: **True**
- `4c330ac_is_fr_mixture`: **True**
- `037ead0_is_boundary`: **True**
- `historical_absorption_not_to_restore`: **True**

## Boundary

- Current `legacy_split` is the explicit two-coefficient form `Δn = n2_elec*I + n_R*I_R`; it is NOT the pre-April f_R mixture.
- The pre-April f_R mixture (4c330ac) uses `I_nl=(1-f_R)I + f_R*I_R`, `Δn=n2*I_nl`, with `f_R=0.15, T2=80ps, T_R=8.4ps, method=iir`.
- 037ead0 (2026-04-02) removed the mixture and introduced `Δn=n2_elec*I + n_R*I_R`.
- Self-steepening: historical applies `shock_intensity` to `I_nl` then multiplies by n2; frozen applies it to the summed `delta_n`. Both are equivalent for the linear tdiff/fft shock operator.
- Raman absorption is intentionally NOT restored: historical conv_deriv uses `(n0/c0)*n_R*IR*dI/dt`; frozen uses `(n_R/c0)*IR*dI/dt`.
