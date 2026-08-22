# Isaacs complete Eq.27 C1 operator report

Overall: **PASS**.

Base SHA: `c9d9b952c4c23d6839374bdc5de184f0cd389eb3`; current HEAD: `c9d9b952c4c23d6839374bdc5de184f0cd389eb3`.
Git dirty: **True**; implementation diff hash (sha256): `c08056a474cd0fc53d37a9fdb5afd14055a8aa2dc3feb0ae1ca70a9e25f244b7`.
Implementation identity is the base/current SHA pair plus the dirty-worktree diff hash; HEAD alone is not sufficient.
Changed paths:
  - `Filament_python/KHz_filament/Config_explain.md`
  - `Filament_python/KHz_filament/README.md`
  - `Filament_python/KHz_filament/config.py`
  - `Filament_python/KHz_filament/config_normalize.py`
  - `Filament_python/KHz_filament/diagnostics.py`
  - `Filament_python/KHz_filament/propagate.py`
  - `Filament_python/KHz_filament/raman.py`
  - `Filament_python/results/isaacs_complete_eq27/PROJECT_STATE.md`
  - `Filament_python/results/isaacs_complete_eq27/c1_closure_summary.json`
  - `Filament_python/results/isaacs_complete_eq27/c1_operator_report.md`
  - `Filament_python/tests/test_isaacs_complete_eq27.py`
  - `Filament_python/tools/audit_isaacs_complete_eq27.py`
No propagation or Slurm job was run.

## Gates

- `electronic_D_IA_closure`: **PASS**
- `rotational_D_IRA_closure`: **PASS**
- `combined_Eq27_closure`: **PASS**
- `coefficient_single_count_audit`: **PASS**
- `Heun_convergence`: **PASS**
- `production_default_unchanged`: **PASS**
- `no_Raman_parameter_change`: **PASS**
- `no_unrelated_physics_modification`: **PASS**

## Closure metrics

- electronic D[I A] relative L2: `1.416747e-17`
- rotational D[I_R A] relative L2: `3.369796e-17`
- combined Eq.27 relative L2: `1.541003e-16`
- combined-minus-components relative L2: `0.000000e+00`
- Heun dz-halving error ratio: `4.000021e+00`
- vacuum-prefactor/sign relative L2: `1.541003e-16`
- wrong n0*omega0 prefactor separation: `2.772169e-04`
- 960 fs edge amplitude ratio: `2.931718e-10`
- pure complex128 field-vs-Eq.10 energy residual: `3.361503e-08`
- The old 128 fs window had edge amplitude ratio `0.678`; its truncated-tail comparison showed `8.1%` spurious flux.

## complex64 projection

- scale: `1.0000000012`
- single-step field relative difference: `1.280228e-08`
- single-step energy relative difference: `3.494231e-09`
- status: **not_primary_for_C2**

## Scope

- Raman parameters are fixed at the C1 audit values; no f_R or historical mixture is used.
- Existing `full_isaacs_eq27` remains rotational-only plus scalar electronic; only the new complete mode uses combined D[(n2 I+n_R I_R)A].
- Ionization, plasma, BK-NEE, self-steepening coefficients, defaults, and production results were not changed.
- Complete-mode scalar `dphi_kerr` is not applicable; self-steepening is represented by the full product derivative.
- A failing gate means STOP and no C2 submission.
