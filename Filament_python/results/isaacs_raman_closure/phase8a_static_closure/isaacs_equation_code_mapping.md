# Isaacs Raman equation-to-code mapping

This Phase 8A record maps the static implementation against J. Isaacs et al.,
*Optics Express* 30, 22306-22320 (2022), Eqs. (7)-(12) and (27).  It does not
alter any Phase 5-7 propagation result.

| Paper quantity | Definition | Repository quantity/path | Status |
| --- | --- | --- | --- |
| `A` | complex electric-field envelope | `E`, `runner.py`, `propagate.py` | verified envelope state |
| `I` | local intensity | `diagnostics.intensity(E, n0)` | verified SI intensity |
| `n2` | instantaneous electronic Kerr index | `beam.n2_air` | independent electronic coefficient |
| `n_R` | rotational Raman index | `raman.n_R` | independent rotational coefficient |
| `Omega(tau)` | Eq. (9) causal rotational kernel | `raman.make_raman_kernel` | explicit sin-exp kernel |
| `I_R` | `Omega * I` | `raman.raman_convolve_intensity` | IIR and linear FFT paths |
| `p_rot` | delayed Raman polarization | no explicit complex polarization object | closure gap audited in Task 6 |
| Eq. (10) | signed Raman fluence change | legacy `w_R/Q_rot_vol` | legacy path clips per time; not Eq. (10) |
| Eq. (27) | product derivative on full `p_NL` | `shock_intensity` split source | approximation audited in Task 6 |

## Fixed Isaacs boundary

- `n2 = 7.8e-24 m^2/W`: electronic Kerr coefficient.
- `n_R = 2.3e-23 m^2/W`: rotational coefficient.
- `omega_R = 1.6e13 s^-1`, `Gamma_R = 1.3e13 s^-1`.
- `f_R` is not an Isaacs Eq. (7)-(12) parameter.
- `T_R/T2` are not the Isaacs production parameterization.
- `n_R/n2 = 2.9487179487` is a coefficient ratio, not an error by itself.
- Double weighting is unsupported: delayed response must be represented once.

## Convention audit

The repository time axis is the retarded pulse-frame coordinate used by the
solver.  The causal convolution samples `Omega(s >= 0) I(t-s)` and requires a
linear, not circular, FFT convolution.  Eq. (27) is applied to the complete
nonlinear polarization in the paper; the repository's split Raman source is
therefore tested separately rather than assumed equivalent.
