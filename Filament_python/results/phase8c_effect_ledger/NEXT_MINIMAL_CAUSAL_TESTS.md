# Next minimal causal tests (proposal only)

No Slurm configuration is generated here, and none of these tests is authorized by Phase 8C-0. Every future run requires new authorization and a numerically admitted production path first.

## Test A — full Eq.27 Raman feedback ON/OFF

- **Goal:** Determine whether complete Eq.27 feedback, rather than the legacy split phase, produces a reproducible signed shift in the `1e22 m^-3` onset.
- **H1 / H0:** H1: enabling only full Eq.27 feedback advances or delays onset by more than `epsilon_x=0.10 cm`; H0: its shift is within `±0.10 cm`.
- **Only changed setting:** full Eq.27 Raman feedback ON versus OFF; all other fixed production parameters, diagnostics, numerical precision strategy, and energy contract identical.
- **Jobs:** two matched production-length jobs after authorization.
- **Discriminating metrics:** all four crossings, especially `1e22 m^-3`; peak position; FWHM; tail integrals; energy and Raman closure gates.
- **Success / stop:** accept only if both jobs pass energy admission and contain complete local `rho_max_z` archives. Stop if an admission failure, missing provenance, or non-identical non-Raman configuration appears.
- **15,000 steps:** yes, because onset and tail metrics are the question.
- **Why ahead of R6 optimization:** it directly answers whether the presently unmeasured full operator changes the physical onset. Any narrow performance work is only a prerequisite to this authorized causal pair, not an independent science objective.

## Test B — electronic Kerr causal ablation

- **Goal:** Measure the independent electronic-Kerr contribution to early onset after the Raman question is bounded.
- **H1 / H0:** H1: switching electronic Kerr off changes `1e22 m^-3` crossing by more than `0.10 cm`; H0: the change is within tolerance.
- **Only changed setting:** `use_electronic_kerr`.
- **Jobs:** two matched production-length jobs (ON and OFF).
- **Discriminating metrics:** crossings, peak position/density, rise width, and energy diagnostics.
- **Success / stop:** both archives and energy contracts must pass; stop on any non-Kerr config difference or if the OFF run makes the prescribed threshold unavailable.
- **15,000 steps:** yes.
- **Why ahead of R6 optimization:** it is the shortest strict test of a leading unquantified nonlinear focusing term.

## Test C — plasma phase / ionization-loss staged split

- **Goal:** Separate dispersive plasma feedback from ionization loss in the remaining coupled trajectory.
- **H1 / H0:** H1: at least one switch causes a crossing shift larger than `0.10 cm`; H0: neither does.
- **Only changed setting:** stage 1 toggles `use_plasma_phase`; stage 2 toggles `use_ionization_loss`, each against the same admitted baseline.
- **Jobs:** three production-length jobs if the admitted baseline is retained as the common control, otherwise four.
- **Discriminating metrics:** threshold crossings, peak density, tail metrics, `dphi_plasma`, ionization-loss channels, and energy closure.
- **Success / stop:** do not combine the switches in one job; stop if either causal pair is not single-delta or if energy admission fails.
- **15,000 steps:** yes.
- **Why ahead of R6 optimization:** it transforms an untestable coupled claim into two explicit physical questions.

Priority follows the ledger: A first because legacy Raman is the only existing strict onset pair but is not full Eq.27; B second because electronic Kerr remains a plausible unisolated focusing term; C third because it needs more jobs and is inherently staged.
