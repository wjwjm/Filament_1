# CPU-S4-5D: same-save fluence reuse

The frozen 14-window ADDA/DAAD small-grid study completed as
`NO_WORTHWHILE_GAIN`. The scientific execution commit is
`f07e7170c1e1966931b59322b014efca910a0e8d`; this document is a later report tip.
The Filament_python tree and frozen science remain unchanged by this document.

The candidate preserves the first saved-point fluence trapezoid and the
original 3D radius post-processing. It removes only the radius's duplicate
time integral: 16 to 8 calls in the specified eight-save loop, with radius and
fluence FWHM outputs retained. Inner counts are source-bound execution
inferences backed by actual save/return events, not independent backend hooks.
The extra profile probe was rejected by the unchanged observer guard and was
not used as a performance sample. Neither A nor D loads the C Eq10 package.

| Measured screening result | Gain, positive means D faster |
|---|---:|
| Balanced later complete body | -0.013706% |
| ADDA body block | -1.663135% |
| DAAD body block | +1.608961% |
| Balanced median POST gap | -0.116969% |
| Balanced full application | -2.028294% |

Both block directions and the original 3% practical gate were not satisfied.
The body result did not exceed the observed A/A log reference. No full-size
job, extra block, second candidate, or production default change followed.

Job 8853929 completed `0:0`: 96 physical CPU cores, 32 GiB, 1536 allocation
seconds, 40.96 allocated core-hours. Batch/extern steps are not double billed.
The new budget is independent of prior stages, and no task job remains active.

Windows and native Linux each passed the current 76 unit/control checks and
22 actual local windows. The HPC study retained 14 continuous eight-interval
windows, 18 original comparison logs, 14,644 numeric exact rows, 1,260 metadata
exact rows, and 64 original timing exclusions. Each qualification boundary
retains the original 108 exact plus 4 timing rule. Source, native layout,
alias, lifetime, cache and W48 route contracts were retained.

The offline review preserves the original bytes and negative/failed evidence.
Only a non-gating `expm1(log_noise)` display scalar differs by one ULP on
Windows; its exact log input and every gate value/decision match. This does
not change the scientific comparator, tolerances or promotion thresholds.
The portable review package and external ZIP receipt are delivered separately.

BAL14 was closed by append-only limited acceptance and a separate portability
sidecar, without rerunning its HPC work or promoting C. Its offline repair is
not an optical speedup.

`DEFAULT=I0 / MANUAL_REVIEW_PENDING / PRODUCTION_QUALIFIED=false`.
