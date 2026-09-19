# E5-1A change and qualification matrix

Base: `131daa541ac37e54d43b29cd3bfc12ba83c3b91d`.
This is local engineering qualification. The authoritative executed results are
in the historical [test results](E5_1A_TEST_RESULTS.json) and the additive
[E5-1A-R results](E5_1A_R_TEST_RESULTS.json). The current disposition is in the
[E5-1A-R completion report](E5_1A_R_COMPLETION_REPORT_20260918.md).

| Boundary | Reused authority | Increment requiring its own evidence |
|---|---|---|
| Optical physics | Original field construction, lens, `propagate_one_pulse`, HR-3 state/sinks | New CURRENT read view, complete schedule injection and fresh-source call |
| Candidate slow state | Existing Streaming CURRENT/POST/NEXT, claims, queue16/block8, barrier/promotion | Per-pulse roots, independent successor copy and receipt validation |
| Hydro | Existing single-screen worker and original interpulse step schedule | Nonzero PRE velocity inheritance, exact supported duration, explicit remainder refusal |
| Reference | Isolated serial/Batch tools and same frozen scientific operators | Independent states and snapshots; never reuse candidate answers |
| Exact | Existing `sha256_array` plus actual shape/dtype/array equality/finite predicates | Parameterized PRE/POST/NEXT/sinks/ledger/optical inventory and complete counts |
| Final pulse | Existing durable per-screen POST artifacts | No enqueue/hydro/NEXT; complete and partial terminal recovery |
| Lineage | Streaming remains state authority | Self-contained READY, parent binding, archived-parent refusal, partial-write windows |
| Storage | No historical data is eligible | One 300 GiB ledger, reservations, finite final outputs, safe per-file GC transaction |
| Recovery | Historical S5 fault evidence retains its original scope | Local fresh-process persistence after parent arrays are reclaimed |

Existing S3/S4R/S5 evidence does not automatically qualify N=3, K=8048, new
successor/terminal wrappers, automatic reclamation or new-process continuation.
The original 48-screen guards remain applicable to their original entries.
The early S5 job-state snapshot is not refreshed or treated as current PASS.

The local baseline check ran the repository test wrapper with `UPPE_USE_GPU=0`:
backend and import passed; sanity was 1 passed; existing Streaming and HR4D
pulse-lifecycle selection was 34 passed, 304 deselected. This baseline is not
new-function qualification and does not establish GPU numerical equivalence.

Protected scientific files, configs/LUTs, old E5-0 documents and original
entries must remain unchanged. Any necessary Streaming interface extension
must be separately identified with its reason, diff and targeted regression;
no such extension may silently change nonfinal recovery or physics.

Formal input materialization, HPC exact, actual GPU overlap, site quota/QoS,
RSS/VRAM, full-z endurance, cross-node/allocation/power-loss survival and B/C
execution remain outside A. Byte models and metadata prototypes are distinct
from measured campaign storage and filesystem performance.

## Current delivery disposition

`E5_1A_PARTIAL / BLOCKED_BY_FORMAL_DRIVER_AND_RECOVERY_QUALIFICATION`.
The report lists code-level A gaps independently of B site gates. In particular,
small-array end-to-end fixture evidence is not a validated formal callback
binding, global creation-admission path, or complete interrupted-handoff recovery.
A07/A09/A14/A15/A17 must retain the applicable partial scope; successful lower-level
tests do not upgrade them. A01-A18 final command outcomes and coverage limitations
are recorded individually in E5_1A_TEST_RESULTS.json.

No narrow change to the original Streaming core was made. The original
Streaming/HR4D regression selection remains the inherited-compatibility check.

## E5-1A-R additive qualification (2026-09-19)

| Item | Added gate/evidence | Current disposition |
|---|---|---|
| A05 | immutable admission, complete paired exact and formal callback-path refusal | PARTIAL: no unique production R/C runner and ordinary R/C receipt ownership is incomplete |
| A07 | active-epoch fencing and hash-bound stale coordinator takeover in independent subprocesses | PARTIAL: full production pointer/index/handoff chain is not instantiated |
| A09 | GC plan is checked before unlink and at commit against trajectory/pulse/attempt/admission plus writer receipt; cross-track reuse is refused | PARTIAL: writer lifetime is not unified with a production coordinator and the positive chain is local synthetic/fixture evidence |
| A14 | formal PRE0/successor/optical writes require admission and StorageBudget intent | PARTIAL: arbitrary production runner output creation is not fully closed |
| A15 | successor READY and complete-campaign-root terminal inventory are mandatory | PARTIAL: no complete positive production role inventory |
| A17 | unchanged 300 GiB/64 GiB formula is checked by runtime planner | PARTIAL: no full production lifecycle or site measurement |

Latest code/test commit: `217f74587edcdf739127602afd73b2b830fc77bb`.
Latest local results: 13 focused E5-1A-R, 43 total E5-1A, 34 existing
Streaming/HR4D, sanity 1, backend and compileall all passed. These results do
not change the final `E5_1A_PARTIAL / BLOCKED_BY_FORMAL_DRIVER_AND_RECOVERY_QUALIFICATION`
disposition or authorize B/C.
