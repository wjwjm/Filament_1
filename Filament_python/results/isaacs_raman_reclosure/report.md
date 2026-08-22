# Isaacs Raman continuous-to-discrete independent reclosure

## Scope

- Source authority: Isaacs et al. 2022 Eqs. (7)-(12) and Eq. (27), transcribed directly from the supplied PDF (PDF pages 4 and 9).
- No parameter fitting, no PyCAP fitting, and no full propagation or Slurm job.
- Audited configuration: `120fs_talebpour_isaacs_full_operator_on.json`.

## Paper-to-SI derivation

With `W(tau)=-1`, define `I_R(tau)=integral Omega(tau-tau') I(tau') d tau'`. Eq. (8) becomes

`Q = n_R n0 I_R / (2 pi chi_L)`, and therefore `p_rot = chi_L Q A = n0 n_R I_R A / (2 pi)`.

Isolating the rotational part of Eq. (27), using `k0=n0 omega0/c`, gives

`dA/dz = i (omega0/c) n_R I_R A - (n_R/c) d(I_R A)/d tau`.

Thus the field prefactor is the vacuum wave number `omega0/c`; `n_R` appears once, and the derivative must act on the complete product `I_R A`.

`omega_R=1.6e13 s^-1` is the angular rate appearing inside `sin(omega_R tau)`. Its ordinary frequency is `omega_R/(2 pi)=2.546e12 Hz`, giving a `392.699 fs` period. `Gamma_R=1.3e13 s^-1` is a damping rate with `76.923 fs` decay time; neither value receives another `2 pi` factor.

## Continuous and discrete closure

- Eq. (9) kernel analytic integral: 1; the first 8192 exact-PWL IIR weights sum to 1, while the analytic infinite sum is 1.
- Production time step: 2.500 fs. Gaussian fp64 maximum absolute response error: 1.202e-04 of peak response.
- In the physically visible region `I_R >= 1e-3 peak(I_R)`, maximum pointwise relative error is 5.142e-03. At the extreme `1e-6` tail it reaches 1.236e-01, but the absolute response there is negligible.
- SciPy adaptive quadrature agrees with independent 60-decimal mpmath checkpoints to 3.057e-16 of peak response.
- The IIR interval coefficients already contain the time integration. There is no missing or duplicated external `dt`; the equivalent discrete kernel weights sum to one.
- Eq. (10) signed integral: -1.195509e+04 J/m^3; deposited energy density is 1.195509e+04 J/m^3.
- Unit chain: `(m^2/W)/(m/s) * (W/m^2) * (W/m^2/s) * s = J/m^3`.
- The strict Isaacs path uses `n2*I` and `n_R*I_R` once each. It does not use `f_R`, does not enable split Raman phase, and does not enable legacy Raman absorption.
- Rotational delta_n reaches 1%, 5%, and 10% of the electronic peak at -128.852, -99.143, and -84.280 fs.
- For this fixed `I_peak=5.000e+17 W/m^2` test, peak `I_R/I_peak=0.837366` and peak rotational index is 2.469156 times the electronic peak. This large response follows from the paper values `n_R/n2=2.948718`, not from IIR amplification.

## Eq. (27) operator audit

- Current rotational RHS vs direct `D[I_R A]`: relative L2=1.839e-16.
- Omitting `I_R*dA/dt` would produce relative L2 error 1.209e-01; the current Raman code does not omit it.
- Current scalar electronic Kerr/shock RHS vs full `D[I A]`: relative L2=6.754e-02.
- Heun one-step errors versus a 128-substep RK4 reference are 4.087e-07, 5.109e-08, and 6.387e-09 as `dz` halves, showing the expected approximately eightfold local-error reduction.
- The production-scale complex64 probe applies a global energy projection with scale 1.00000000327; its single-step field-level difference from pure Heun is 1.858e-08. It is not the unmodified Eq. (27) Heun map; this local test does not infer its cumulative propagation impact.
- Therefore the Raman `full_isaacs_eq27` suboperator is mathematically equivalent to the paper's full rotational polarization derivative, while the overall nonlinear step is not a monolithic Eq. (27) implementation of every `p_NL` term.
- The inactive FFT compatibility path was not used in the production comparison; its centered-time kernel interface remains a separate implementation risk, but it cannot explain the current IIR result.

## Decision

**B. The continuous Raman formula and the audited strict IIR/full-rotational code path close, but the overall operator called `full_isaacs_eq27` is not a complete mathematical equivalent of Eq. (27) because electronic Kerr remains scalar and complex64 may apply a global energy projection.**

The next step must be operator-only and local: implement or explicitly separate the full electronic `D[I A]` term, quantify/remove the complex64 projection boundary, and rerun the small-array closure. Only after that closes should one design a single full propagation. The Raman parameters must remain fixed; no new Raman-ON propagation is justified by this audit alone.
