# E5-1A production path completion report

Date: 2026-09-20  
Branch: `codex/hr4e5-e5-1a-streaming-implementation`  
Repair base: `493109fbe0bba41638bed709f2b04b9458bc609c`  
Code/test commit: `dd4b2c0019ea7d78c5ad06b3fa6710806a327356`

## Disposition

`E5_1A_IMPLEMENTATION_READY_FOR_REVIEW`

This is a local implementation-review gate, not E5-1A CLOSED. It does not
authorize formal execution, HPC/Slurm, GPU work, S5 changes, E5-1B or E5-1C.
The executed evidence level is `LOCAL_ORCHESTRATION_QUALIFICATION`; it is not
formal-input equivalence or scientific qualification.

## Production entry and admission

The only supported production creator/resumer is
`open_e5_1a_production_campaign(E5AProductionSpec)`. Its facade exposes only
`run()`, `pause()`, `close()` and `report()`. Public construction and the
private driver-opening helper both reject arbitrary runners; formal mode cannot
accept a caller runner, callback PASS, optical double, caller terminal roles or
bypass.

The factory rebuilds and binds config raw hash, normalized runtime parameters,
analytic-source canonical array hash, LUT manifest (including config-bound
`DISABLED_BY_CONFIG`), schedule/grid/PRE0, N/K, block8, queue16, budget including
safety margin, retention and execution level. Create and resume revalidate the
same identity before scientific writes. Future `FORMAL_SITE_ADMISSION` uses the
same runner and additionally requires qualified input/provenance/resource
evidence; it does not select another scientific implementation.

## Writer, reservation and takeover closure

All production creation sites follow reservation → intent → epoch-bound writer
→ write → ownership completion → reservation reconciliation → writer close.
The durable storage ledger is the single registry for reservations, intents,
writers, artifacts and GC plans.

Windows identities use PID plus process creation time and an active exit-code
check; Linux uses PID plus `/proc` start token and boot-qualified host identity.
Takeover behavior is fixed:

- matching live ACTIVE coordinator/writer: reject without mutation;
- unknown host or unverifiable identity: reject fail-closed;
- verified dead/PID-reused writer in a stale epoch: atomically mark
  `INTERRUPTED_STALE`, preserve residual files and ACTIVE reservation charging,
  and emit a registry-backed takeover receipt;
- only then create the new coordinator epoch; it may rebind the same interrupted
  intent/reservation, while the old epoch cannot close, commit or reclaim.

GC quiescence receipts are generated from the registry. Formal GC binds and
rechecks trajectory, pulse, attempt, admission, exact evidence, READY,
dependency evidence and target identity before unlink and again on verification.

## Executed production chains

The N3/K8 qualification uses `Nx=Ny=Nt=8` and real CPU
`propagate_one_pulse` on both independent trajectories. R uses HR4C batch state
plus the original CPU hydro evolution; C uses existing Streaming claims,
queue16, block8, barrier and promotion. The order is R → C → exact → R
successor/GC → C successor/GC → pair commit; the final pulse is POST-only.

The first process completed pair 0 and exited normally. A fresh process opened
the same campaign and completed pairs 1 and 2. A separate uninterrupted run
matched retained array exact signatures and the normalized exact, lineage, GC,
terminal inventory, writer-role/status and reservation structures. A third
read-only process reopened the terminal Candidate CURRENT+POST; no p3/PRE4/NEXT4
was created.

Observed N3 counts:

- 336 screen exact rows;
- 27 ledger arrays;
- three final optical arrays;
- 96 NEXT→successor CURRENT binding rows.

The N2/K16 supplement used the same factory/runner/admission and real CPU
optical path. It covered two complete block8 groups, queue16 POST consumption,
barrier/promotion, one successor/GC and the second-pulse terminal inventory. It
does not duplicate the N3 recovery or negative matrix.

## Terminal inventory and resource model

Terminal roles are generated from the production contract and owned artifacts.
Required role minima, attempt zero, complete exact/successor/GC evidence, no
ACTIVE reservation, no ACTIVE or INTERRUPTED intent, no ACTIVE writer, and no
unregistered NPY/NPZ/MAT/HDF5 payload are mandatory. Successor archive receipts
are owned and charged by the same successor intent. Candidate final CURRENT+POST
and `POST_FINAL_READY` are retained.

The scalar K8048 model remains:

- campaign cap: 322122547200 bytes (300 GiB);
- modeled peak: 299676844288 bytes;
- peak headroom: 22445702912 bytes;
- final-output cap: 68719476736 bytes (64 GiB);
- modeled final retention: 47504725760 bytes;
- final headroom: 21214750976 bytes.

The same planner is cross-checked against the N3/K8 durable ledger. Local runs
record actual filesystem free space and the 300 GiB policy cap as
`LOCAL_POLICY_CAP_NOT_SITE_QUOTA`. Site quota/free space/QoS, allocated blocks,
RSS/VRAM/I/O, GPU/CUDA and K8048 materialization move to E5-1B.

## Qualification update

| Item | Result |
|---|---|
| A05 | `PASS_LOCAL_PRODUCTION_PATH` |
| A07 | `PASS_LOCAL_PRODUCTION_RECOVERY` |
| A09 | `PASS_LOCAL_PRODUCTION_PATH` |
| A14 | `PASS_LOCAL_PRODUCTION_PATH` |
| A15 | `PASS_LOCAL_PRODUCTION_PATH` |
| A17 | `PASS_LOCAL_RESOURCE_MODEL` |

Scientific propagation, nonlinear, ionization and Raman operators and frozen
production config/LUT were not changed. One narrow Streaming infrastructure
change retries Windows `os.replace` only for transient `PermissionError`
(20 × 10 ms maximum); it does not alter claim, barrier, promotion or numerical
semantics. The formal optical entry gained a narrow deferred-intent settlement
option so concurrent C hydro NEXT files are owned by the same production intent.

Final verification and Git/ZIP receipts are recorded in
`E5_1A_R_TEST_RESULTS.json` and the web-review bundle index.
