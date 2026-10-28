# CPU-S4-5D saved-fluence reuse

This namespace contains the bounded `SAVED_FLUENCE_TRAPEZOID_REUSE_SAME_ORDER`
candidate.  `L1_REF_5D` binds the accepted 4A/4B L1 factory and retains the
original 3-D `second_moment_radius` call.  `D_FLUENCE_REUSE_V1` changes only a
private, source-bound clone of the saved branch: the original `F2D` trapezoid
is kept verbatim and its result is passed to
`fluence.second_moment_radius_from_f2d`.

The private helper preserves the original 3-D post-integration sequence,
including `nan_to_num`, non-negative clipping, the `count_nonzero(csum <
target) + 1` threshold rule, coordinate-origin moments, reductions, and
square-root order.  A source-bound AST proof compares that post-integration
body with the original diagnostics function and records equal structural
hashes; the existing public 2-D helper is intentionally not used.

Both roles use the same exact-eight-row body clock and source-bound scalar
qualification counters.  Counters report saved calls, outer and inner
integrals, radius calls, and the fluence FWHM call; the second plasma FWHM
call remains untouched and is only structurally known.  Inner-integral counts
are explicitly source-bound inference rather than a backend trapezoid hook;
no field arrays are captured or retained.  `compile_interceptor.static_receipt`
records the baseline/stripped AST hashes, post-processing hashes, exact
inverse proof, and sealed private clone-global helper slots.  The frozen
`propagate.py` and `diagnostics.py` blob/canonical identities are checked
before a role is built.

This is engineering/qualification support only.  It does not modify the
production propagation modules, defaults, physical parameters, or frozen
results; `DEFAULT=I0` and scientific acceptance remain outside this namespace.
