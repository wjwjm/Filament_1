# CPU-S4-5C: private Eq10 pointwise experiment

Only `EQ10_DERIVATIVE_POINTWISE_SAME_ORDER` is implemented. K1 materializes
the original frequency coefficient times the original FFT output; K2
materializes the original IFFT real view divided by the current scale.
The coefficient, FFT/IFFT, reductions, stable loss, G3/R8/L1 mathematics,
factor cache and immutable POST source remain on their original paths.

The new kernels use fixed contiguous output ranges of `Nt*32` elements,
read actual input strides, allocate only the two replaced native outputs,
and retain no scientific arrays. Normal HPC execution uses W48 and restores
the outer W24 mask. W1 and explicit W2 diagnostic execution are local only.
Serial and parallel functions have distinct code names and cache identities.

The common adapter inserts one complete interval-body scalar clock before
the original G2 clone guard is captured. Removing it restores the original
AST. There is no S45B fine PROFILE. The candidate stage transformation has
an exact inverse that restores the original R8 transformed AST.

The task plan, committed source binding and original comparators are
mandatory. The full-size entry additionally requires the same-SHA small-grid
promotion receipt. No kernel-only timing grants that permission.

Default remains I0. This namespace is experimental, not production-qualified.
Qualification and performance status are recorded in the task report, never
inferred from the presence of this code or a passing software test.
