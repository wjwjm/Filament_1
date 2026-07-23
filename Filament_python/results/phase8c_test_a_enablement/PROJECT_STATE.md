# Phase 8C-A Test A enablement — project state

Recorded on 2026-07-23 before any Phase 8C-A numerical-path work.

## Source and frozen baselines

| Item | SHA / status |
| --- | --- |
| Test A branch | `codex/phase8c-test-a-enablement` |
| User-authorized current remote-main baseline | `f70c5f48dd11e6db2376604751c8b13afdc1cd2f` |
| Frozen physical baseline | `e11d13f103c484953c0f733aa9b410bff385b2b5` |
| Branch base relationship | `f70c5f4` is an ancestor of this branch |
| Current branch HEAD at record | `d76b5863b45ea170359a4afc27acbb7f84667206` |

`d76b586` is a separate non-physical retention commit for the three Phase 8C PDF redraw artifacts explicitly approved on 2026-07-23. It does not alter the Test A physics, numerical contract, configuration, grid, or step contract.

## Audit of `e11d13f..f70c5f4`

The audited diff contains only:

- Phase 8C read-only inventory, ledger, report, and figure artifacts;
- downloaded Phase 6 Raman-phase-off PNG diagnostics;
- `Filament_python/tools/build_filament_effect_ledger.py` and its test.

It contains no changes under `Filament_python/KHz_filament/` and no production configuration changes. Therefore the frozen physical baseline remains `e11d13f`.

## Working-tree exceptions preserved without staging

The following pre-existing untracked paths remain deliberately untouched:

- `Filament_python/results/isaacs_raman_closure/phase8b_controlled_propagation/job1_179706.err`
- `Filament_python/results/isaacs_raman_closure/phase8b_controlled_propagation/job1_179706.out`
- `phase8b_r_job1_audit_e724cd66.bundle`
- `tmp/`

## Authorized Phase 8C-A scope

- BK-NEE mixed-precision internal implementation, conversion placement, allocation/workspace reuse, fixed-kernel caching, FFT scheduling, and opt-in profiling/memory diagnostics.
- Test A configuration preregistration, configuration-difference audit, smoke/report tooling, and related tests.
- The narrowly scoped Phase 8C ledger corrections enumerated in the Test A task.

## Frozen / prohibited items

The 800 nm, 17 GW, 120 fs, FT90, grid, z-step, ionization, Kerr, plasma, ionization-loss, self-steepening, and full Isaacs Eq.27 Raman parameters are frozen exactly as recorded in the Task A specification. No filters, cropping, energy projection, artificial compensation, legacy Raman phase/absorption, grid/step relaxation, or full propagation is authorized.

Full Test A ON/OFF 15,000-step jobs are **not authorized**. At most the restricted smoke-job budget defined by the task may be used after its prior gates are satisfied.
