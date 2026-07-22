# R5.1 BK-NEE precision strategy matrix

| Candidate | Implementation | Physics/operator change | Production default | Evaluation status |
|---|---|---|---|---|
| A `baseline_complex64` | Current complex64 FFT/multiply path | None | Preserved legacy default | Baseline |
| B `orthonormal_fft` | Matched `norm="ortho"` on every FFT/IFFT pair | None | Opt-in | Candidate only |
| C `mixed_precision` | complex64 storage, complex128 inside a linear half step, one output cast | None | Opt-in | Candidate only |
| D `unitary_projection` | Float64 global energy scale after an otherwise unchanged pure-phase half step | No physical term added; prohibited for a non-unitary/absorptive linear path | Opt-in | Candidate only |

Candidates are tested separately.  No candidate is combined with another during
R5 screening.  Candidate D is never an energy-deposition channel and must not
be selected merely because it enforces a scalar norm.
