# Phase 7 Raman static audit

## Overall decision

Overall static audit status: **not_ready_parameter_conflict**. No new Raman-parameter propagation is admissible. This is a static CPU audit only; no Slurm job or full propagation was run.

## Required physical conclusions

1. Phase 6 proves that **the Raman phase implemented and applied by this code** has a strong causal effect in the 120 fs simulation. It does not by itself prove that the present parameterization is a source-verified air Raman model.
2. The actual production response is `omega_R=1.6e+13 rad/s`, `Gamma_R=1.3e+13 s^-1`, giving period `392.699 fs` and dephasing time `76.923 fs`.
3. Yes: with explicit `omega_R/Gamma_R`, `T_R/T2` are silently shadowed in the `rot_sinexp` production path. The configured alternatives imply `omega=7.47998e+11`, `Gamma=1.25e+10`; ratios are `21.390` and `1040.0`.
4. `n_R/n2_air=2.948718`, and `(n2_air+n_R)/n2_air=3.948718`; the rotational coefficient can therefore exceed the electronic term when `I_R/I` is order one.
5. `f_R=0.15` is read but does not scale the `rot_sinexp` phase or absorption calculation.
6. No primary-source evidence was found in this audit that establishes whether `n2_air` includes delayed response.
7. There is a documented **risk**, not a proven conclusion, of Kerr/Raman double weighting: the coefficient semantic provenance is unresolved. For reference only, total-Kerr assumption A gives electronic/delayed `6.630e-24/1.170e-24`, while pure-electronic assumption B gives delayed `1.376e-24 m^2/W`.
8. Current production IIR agrees with direct causal convolution to `0.0254` (40 fs) and `0.0126` (120 fs); the IIR gate passes at the static tolerance.
9. The FFT path fails static reference comparison by about `4e+14`; it lacks the convolution `dt` factor and shows circular wrap behavior. Production Phase-6 used IIR, so this is a latent nonproduction defect, not an explanation retroactively assigned to Phase 6.
10. In `conv_deriv`, signed static exchanges are `-172/-478 J m^-3` for 40/120 fs while positive-clipped values are `15.7/90.5 J m^-3`; clipping is not signed net exchange and the absorption-energy gate fails.
11. Field usage identifies `f_R` as read-but-unused, `T_R/T2` as shadowed, `Omega_R` and `tau2` as unused, and several geometry/tau compatibility fields as read-but-unused. Documentation also conflicts with `absorption_model` defaults.
12. New Raman parameter propagation is **not allowed** until parameter provenance, coefficient semantics, and absorption energy closure are resolved.

## Scope preservation

> 第六阶段的因果消融结果本身仍然有效：它证明当前代码中被应用的 Raman phase 对 120 fs 传播具有显著影响。第七阶段审计的是该 Raman 实现及参数是否具有正确、可追溯的物理归一化；静态审计不得追溯性篡改第六阶段原始结果。

Recommended next propagation cases: **none until provenance is resolved**. If and only if those gates are closed later, any next case must remain a single-factor, separately authorized test.
