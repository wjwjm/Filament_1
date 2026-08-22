# C2 numerical interpretation limits

The comparison script's evidence gate passed and its predefined mechanical classification is **`electronic_eq27_operator_not_supported`**: the complete candidate moves the `1e22 m^-3` onset by only `+0.032105 cm`, below the `0.1 cm` threshold.

This classification is conditional rather than an unconditional physical rejection of electronic `D[I A]`:

- Candidate energy loss from the input is `1.369891688e-4 J`, while cumulative reported deposition is `1.061128714e-4 J`. The remaining `3.087629739e-5 J` is not closed by the current energy ledger. The candidate Raman cumulative closure residual is only about `1.55e-7`, so the larger difference remains unresolved by the available diagnostics.
- Jobs `180748/180749` remain `fallback_verified_non_strict` and used `mixed_precision`; candidate job `221822` uses the locked baseline precision and has `verified_bundle_non_strict` source provenance.
- The candidate changes the complete combined Eq.27 path: derivative algebra, electronic stage placement, and electronic-rotational Heun coupling are not separately identified.
- The reported density RMSE is evaluated on the simulation/PyCAP overlap interval, not on portions of either axis outside their common support.
- Zero recorded adaptive rejections and safety triggers establish only the recorded execution path; they are not an independent adaptive-stability proof. Zero GPU allocation/reservation diagnostics reflect disabled profiling rather than zero GPU use.

Accordingly, the supported conclusion is:

> Under the non-strict comparator provenance and complete combined Eq.27 definition, the candidate does not produce a centimetre-scale onset correction. The unresolved candidate energy-ledger residual and combined-operator coupling prevent a stronger claim that the isolated electronic derivative algebra has been physically disproven.
