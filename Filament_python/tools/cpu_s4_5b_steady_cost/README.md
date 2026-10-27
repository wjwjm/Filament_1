# CPU-S4-5B private steady-cost instrumentation

This namespace adds the bounded scalar A8 propagation scopes and B8 Raman
Eq10 scopes required by CPU-S4-5B.  It does not select or implement a physics
optimization, alter the L1 factor cache, or qualify a production result.

The role entry point is `candidate.build_role`.  It returns the existing G2
six-tuple `(wrapped, dispatch, clock, binding, audit, lease)`.  The role is
selected by an explicit boolean/profile pair:

- `L1_S45B_OFF`: `profile=False`; the new fine scopes and zero marker are
  disabled, while `APPLICATION`, `SOURCE_VALIDATION`, `L1_FACTORY`, and
  `INTERVAL_BODY` common shells remain enabled with the same definitions.
- `L1_S45B_PROFILE`: `profile=True`; the same common shells plus the fixed A8
  and B8 scalar scopes are enabled.

`candidate.make_recorder(profile)` returns a `RecorderPair`.  Its `b` field is
the B-owned `TimingRecorder`; its `s45a` field is an exact instance of the
accepted S4-5A recorder.  The pair must not be replaced with a subclass or
merged into one raw event tree.  The default `make_recorder_pair()` budget is
8,192 combined raw scopes/markers (A is capped at 4,096 and B receives the
remainder); `RecorderPair.snapshot()` preserves both raw trees and adds a
separate merged scalar tree.  The merged topology uses strict wall-clock
containment, non-strict CPU compatibility (Windows CPU-clock ties are valid),
and recomputes direct-child and exclusive integer nanoseconds.

In PROFILE, B's Eq10 helpers read only the active exact S4-5A scalar frame for
phase/stage while retaining B's original zero-based interval ordinal.  The
bridge rejects missing companion frames, owner/profile changes, and arbitrary
callbacks.  Every generated wrapper also carries a bounded scalar
`source_site`; no arrays, arguments, results, or exception frames are kept.

The propagation AST seam is `adapter.make_g2_compile_interceptor`, used by
`adapter.attach_with_b` before the G2 clone is compiled and guarded.  Removing
B nodes with `instrument.strip_generated_nodes` must restore the pre-B G2
timing/boundary tree.  `instrument.instrument_eq10_stage` similarly restores
the exact input stage AST after removing its scopes, three fixed transparent
call shims, and the scalar zero marker.

The pair reserves a conservative 64 MiB metadata budget for raw plus merged
records.  Capacity exhaustion is an explicit exception and never a dropped
event.  New source-file identities use the repository `provenance_v2` raw /
canonical helpers; the fixed backend registry additionally seals FFT/IFFT/sum
callable identity, Python implementation, globals, closure, and bound-method
components.

The two test modules are AST/recorder checks only.  They are not a substitute
for the parent Agent's official Windows/Linux transparency gates or HPC
measurement.  No scientific qualification, speedup, long-range steady-state,
or production authorization follows from these files.
