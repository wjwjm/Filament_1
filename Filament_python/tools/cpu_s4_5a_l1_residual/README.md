# S45A small-grid L1 residual instrumentation

This private measurement namespace calls the accepted S44A L1 factory and its
unchanged G3 ownership, FFT, five mathematical kernels and 256 MiB factor cache.
L1_OFF disables the added scalar clocks. L1_PROFILE adds bounded scalar scopes
in factory-private Raman and propagation helpers. It does not implement a new
numerical algorithm or replace a scientific expression.

The fixed HPC domain is complex128 native C `[384,117,129]`, eight continuous
intervals, Raman W48, outer mask24, temporal FFT48, spatial FFT24 and BLAS1.
Local engineering uses `[384,17,19]`, Nt384 and explicitly one worker. Local
checks are not W48 qualification. Every case starts a new process/factory and
an empty two-table cache; no propagation warmup is added.

The inherited public strict paths and hook refusal remain active. Only fixed
source-bound wrappers and AST timing statements are installed in private
namespaces; removing timing statements restores the existing scientific AST.
The existing original linear/nonlinear guard objects and code are retained.
No new E/factor content hash, scientific array copy, reduction or response reuse
is performed for measurement. The recorder retains scalar timing records only.

Raman stage1 and fresh predictor stage2, pre/post, coefficient spans, complete
IIR loops, energy work and existing math helpers are distinguished. Parent
exclusive time is its inclusive time minus its own direct children. Residual
self time remains unattributed. The original ionization subclocks are exported
separately and overlap with the new tree; their CPU time is unobserved.

Formal measurement is three fixed OFF/PROFILE pairs after two qualifications.
Both application and propagation ratio medians must be at most1.05 and every
pair at most1.10, with original full exact/layout/alias/lifetime/cache gates.
Ratios below1 are not optimization gains. Scope shares are PROFILE coverage,
not OFF savings. Empty timing control is recorded and never subtracted.

HPC maximum: two jobs, 20 minutes/job,96 physical CPUs,32 GiB,zero GPU,
64 allocated core-hours total,one task job in flight,no requeue. References
are checked completely for raw file identity on the compute node before any
propagation. Source provenance is checked before submission locks.

DEFAULT=I0. New mathematical candidate IMPLEMENTED=false,
PERFORMANCE_QUALIFIED=false,PRODUCTION_QUALIFIED=false.
Stop: MANUAL_REVIEW_PENDING.
