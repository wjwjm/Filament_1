# Phase 8C-B Test A final report

Date: 2026-07-24 (Asia/Shanghai)

## Decision

The strict full Eq. (27) Raman ON/OFF comparison classifies rotational Raman feedback as a **major contributor** to the simulated filament onset and density response.

- At the `1e22 m^-3` density threshold, Raman ON crosses at `x_focus = -16.380 cm` and Raman OFF at `-14.088 cm`: ON is earlier by `2.292 cm`.
- The peak density is `6.387e22 m^-3` with Raman ON and `2.524e22 m^-3` with Raman OFF (`2.53x` larger with feedback).
- The density peak occurs at `-14.350 cm` with Raman ON and `-10.370 cm` with Raman OFF: ON peaks `3.980 cm` earlier.
- The onset shift increases with density threshold: `+0.091 cm` at `1e19`, `-0.368 cm` at `1e20`, `-1.180 cm` at `1e21`, and `-2.292 cm` at `1e22 m^-3` (negative means Raman ON is earlier).

The comparison uses `x_focus_cm = 100 * (z_m - 0.95)` without shifting, smoothing, or renormalization.

## Job and provenance validation

Slurm jobs `180748` (ON) and `180749` (OFF) both completed with exit code `0:0` on node `g0609`. Runtime was `04:21:05` for ON and `02:59:13` for OFF. Slurm used its default combined stdout/stderr stream, so each job has one `slurm-*.out` file and no separate stderr file.

Both metadata records validate:

- expected Git SHA = actual Git SHA = `f0a7b5d5ac103546bd693378e8f8efb4f07c6c27`;
- source worktree clean at execution;
- GPU model `NVIDIA GeForce RTX 5090`, total memory `33,668,988,928` bytes;
- 8 CPU threads;
- case IDs and Slurm IDs match the authorized pair.

The remote raw config SHA-256 values (`aafec917...` ON and `1c141594...` OFF) match the submitted guards. Local raw hashes differ because Git checked out CRLF while the remote staging copy used LF. Parsed/canonical JSON hashes match exactly between local and remote (`778ff58c...` ON and `95099020...` OFF). The only semantic ON/OFF difference is `propagation.use_raman_full_operator: true -> false`.

The staging path contains the suffix `_c4edd8d399f2_proxy` because a local-bundle fallback was used, but the job guard and metadata establish that both runs actually executed the same clean `f0a7b5d5...` commit. Therefore, **the fallback provenance does not invalidate or reduce the admissibility of this paired causal comparison**. It only affects how the exact source snapshot was transported to the cluster; independent retrieval depends on retaining the local bundle or publishing the SHA to GitHub.

## Diagnostic and sanity validation

Both repository diagnostic reports pass with 15,000 finite z records and no warnings. The Phase 8C-B audits also pass: raw Raman response is nonzero in both cases, the legacy Raman attenuation channel is zero, the full Eq. (27) RHS/operator is applied only for ON, and it is identically inactive for OFF.

The Slurm logs contain no fatal error, traceback, NaN, OOM, cancellation, failure, or warning match. The sanity envelope is satisfied:

- `U_z` has no unexplained growth: ON ends `4.984%` below its initial energy; OFF ends `1.682%` below, with maximum positive drift only `1.32e-5` relative.
- Maximum adjacent-step `I_max_z` ratio is about `1.0074` in both runs, with no order-of-magnitude jump.
- Maximum on-axis density is below the neutral-density alarm scale (`1e25 m^-3`).
- `w_mom_z` contracts smoothly from about `1.881 mm` to minima near `0.397 mm` (ON) and `0.395 mm` (OFF), then expands.
- Plasma, fluence, and temporal FWHM diagnostics are finite and positive throughout.
- ON final cumulative Raman closure residual is `1.69e-6` (maximum absolute `4.21e-5`); OFF is exactly zero as required.

## Result assets

- `test_a_effect_summary.json`: classification, threshold shifts, and shape metrics.
- `test_a_crossing_shifts.csv`, `test_a_metrics.csv`, `test_a_pycap_comparison.csv`: tabular comparison outputs.
- `test_a_on_diagnostic_audit.json`, `test_a_off_diagnostic_audit.json`: switch and diagnostic audit results.
- `rho_max_on_off_pycap.png`, `i_max_on_off.png`, `crossing_shift_vs_threshold.png`, `raman_intensity_density_effect_chain.png`: final figures.
- `test_a_on_job_metadata.json`, `test_a_off_job_metadata.json` and the two diagnostic reports: execution provenance and runtime validation.

Raw NPZ files and Slurm streams are retained locally for reproducibility but are not intended for Git commit.
