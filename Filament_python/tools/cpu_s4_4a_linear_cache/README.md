# CPU-S4-4A linear-factor reuse

This private experimental role replaces only the factor getter inside the
source-bound BK-NEE function. It uses the accepted G3 factory and unchanged
scientific operators from baseline `41d0957eff04280b74fef839dc6a8d4676e2316a`.
The science tree remains `821fec5bdacd832b0347e1f2e84285cb54172e57`.

The core AST still computes every coefficient and `phase_xy` in its original
frequency order. Its `transfer_kernel_preparation` callback now calls a private
cache getter with the unchanged two-dimensional exp/astype expression as a
lazy builder. Complete readonly plane tuples become reusable only after the
entire linear half-step succeeds. Key identity uses the converted small grid
parameters and exact scalar bits, precision/norm/backend and source/AST version.
It never scans E or a factor table for a new content hash.

Each factory/case owns at most two tables and a 256 MiB conservative persistent
budget including key/container reserve. At a new-half safe boundary, capacity
pressure evicts complete LRU entries before a new build. Eviction remains an
eviction if that later build fails; revisiting its old key rebuilds it exactly.
No resource outside this cache is removed. The cooperative single-controller
scope does not promise protection against arbitrary native/reflection writes.

The frozen producer remains the original G guard. The new cached execution
producer has a separate source, AST/code, namespace and owner binding. Each
call verifies both identities. Original four-return, pre/post lifetime,
nonlinear in-place, alias, FFT and scientific comparison rules remain active.
`G3_REF` uses the old factory without importing any cache implementation.

`entry` runs either one reference or candidate window. The common application
clock starts before source/factory/cache initialization and ends after original
normal output persistence. This includes factory costs previously outside the
old SG application clock. Research comparisons/archive service is separate.
The first-boundary comparison loads both actual receipt paths after collector
retirement and streaming-view close. New counters are separate JSON sidecars;
the original profile scope measures logical factor acquisition requests.

Local validation uses `Filament_python/tools/run_local_tests.ps1` with the
frozen explicit interpreter, external caches/outputs and worker=1. Enable
`CPU_S4_4A_RUN_REAL_LINEAR=1` for the actual J2 four-form test. Real local entry
tests use Nt384 micro space and continuous two/eight intervals. Their source
binding files are task-private artifacts, never synthesized Git identities.

The authorized `hpc` entry serializes QUAL_G3_REF, QUAL_L1 and the three fixed
G3_REF/L1 paired batches. Every case starts a new process and empty cache.
It requires a clean immutable source binding, actual one-node 96-physical-core
allocation, no GPU, 32 GiB, twenty-minute wall limit and actual cgroup guardian.
No login-node propagation or full-size cache is admitted. Only original exact
qualification followed by median application gain >=3% and >=2/3 positive
batches meets the experimental gain gate.

DEFAULT=I0. FULLSIZE_NEW_CACHE=NOT_RUN. PRODUCTION_QUALIFIED=false.
The final stop remains MANUAL_REVIEW_PENDING, including a successful experiment.

Array writeability follows the supported NumPy interface described in the
[NumPy 2.2 setflags documentation](https://numpy.org/doc/2.2/reference/generated/numpy.ndarray.setflags.html).
