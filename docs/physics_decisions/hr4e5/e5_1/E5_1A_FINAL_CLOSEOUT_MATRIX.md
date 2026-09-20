# E5-1A final closeout matrix

Date: 2026-09-20

Authority: this table consolidates the historical A01–A18 evidence with the
sealed production-path qualification through starting HEAD `46e56e8` and the
final overlap test. It supersedes earlier current-state `PARTIAL` labels but
does not rewrite their historical evidence.

| ID | Requirement | Final status | Evidence file/test | Scope or limitation | Future stage |
|---|---|---|---|---|---|
| A01 | Frozen implementation boundary and protected inputs | `PASS_LOCAL_PRODUCTION_PATH` | `E5_1A_CONTRACT.json`; protected-file audit | Scientific operators/config/LUT/S5 unchanged; one documented Windows filesystem retry only | E5-1B rechecks formal input provenance |
| A02 | Complete independent R/C paired execution | `PASS_LOCAL_PRODUCTION_PATH` | `test_production_n3_k8_split_process_matches_uninterrupted` | Real CPU N3/K8 local qualification, not K8048/GPU | E5-1B inputs/site; E5-1C scientific canary |
| A03 | PRE/current velocity inheritance and three-field closure | `PASS_LOCAL_PRODUCTION_PATH` | N3/K8 exact receipts; `E5_1A_R_TEST_RESULTS.json` | Exact local small-grid arrays | E5-1C scientific interpretation |
| A04 | Same-pulse optical–hydro overlap | `PASS_PRODUCTION_OVERLAP` | `E5_1A_PRODUCTION_OVERLAP_EVIDENCE.json`; `test_production_n2_k16_multiblock_overlap_happy_path` | Candidate p0 proves block 0 hydro start before optical complete on real CPU production path | GPU/site overlap remains E5-1B |
| A05 | Authentic production entry and immutable admission | `PASS_LOCAL_PRODUCTION_PATH` | production factory/refusal tests; `E5_1A_PRODUCTION_PATH_COMPLETION_REPORT_20260920.md` | Local admission level is not formal-input equivalence | E5-1B formal admission |
| A06 | Final-pulse POST-only behavior and replay refusal | `PASS_LOCAL_PRODUCTION_RECOVERY` | focused E5-1A tests; terminal reopen in N3/K8 | No node/power-loss claim | E5-1B site recovery |
| A07 | Epoch fencing, stale takeover and new-process recovery | `PASS_LOCAL_PRODUCTION_RECOVERY` | live/dead/unknown writer tests; N3/K8 split process | Local host process identity only; cross-host verifier excluded | E5-1B site verifier |
| A08 | Complete exact object inventory | `PASS_LOCAL_PRODUCTION_PATH` | N3/K8: 336 screen rows, 27 ledgers, 3 optical arrays | Small local physical configuration | E5-1C scientific acceptance |
| A09 | Registry-backed, identity-bound safe GC | `PASS_LOCAL_PRODUCTION_PATH` | GC negative tests and N3/K8 production chain | Local filesystem semantics | E5-1B site filesystem |
| A10 | Self-contained successor and retained terminal state | `PASS_LOCAL_PRODUCTION_PATH` | successor READY/lineage tests; terminal Candidate CURRENT+POST reopen | Does not authorize formal-data deletion | E5-1B formal retention authority |
| A11 | Interrupted reclaim recovery and unknown-missing refusal | `PASS_LOCAL_PRODUCTION_RECOVERY` | focused storage/GC recovery tests | Local subprocess/filesystem only | E5-1B site recovery |
| A12 | Traversal/link/replacement safety refusals | `PASS_FIXTURE` | `E5_1A_TEST_RESULTS.json`; storage refusal tests | Platform-specific link cases include controlled fixtures | E5-1B site filesystem validation |
| A13 | 300 GiB hard cap and oversize refusal | `PASS_LOCAL_RESOURCE_MODEL` | `E5_1A_RESOURCE_BUDGET.csv`; planner tests | Policy cap, not measured site quota | E5-1B quota/QoS |
| A14 | Reservation→intent→writer ownership for production creation | `PASS_LOCAL_PRODUCTION_PATH` | production writer/storage tests; terminal quiescence test | No alternate production writer path accepted | E5-1B site allocation |
| A15 | Automatic terminal role inventory and final-output cap | `PASS_LOCAL_PRODUCTION_PATH` | terminal inventory tests; 64 GiB model | Local terminal inventory; K8048 not materialized | E5-1B materialization/site retention |
| A16 | Fresh-process persistent continuation equals uninterrupted run | `PASS_LOCAL_PRODUCTION_RECOVERY` | N3/K8 split/uninterrupted comparison | Normal process exit plus separately tested stale-writer takeover | E5-1B allocation/node recovery |
| A17 | Allocation/retention resource model | `PASS_LOCAL_RESOURCE_MODEL` | `E5_1A_RESOURCE_BUDGET.csv`; actual small ledger | `LOCAL_POLICY_CAP_NOT_SITE_QUOTA`; no RSS/VRAM/I/O measurement | E5-1B resource measurement |
| A18 | Existing Streaming/HR4D compatibility | `PASS_LOCAL_PRODUCTION_PATH` | 34-test Streaming/HR4D regression; Windows retry coverage | Scientific and lifecycle semantics unchanged | E5-1B Linux/GPU environment check |

There is no unresolved E5-1A functional `PARTIAL`. Deferred rows identify site,
formal-input, GPU, or scientific-interpretation authority that belongs to
E5-1B/E5-1C; they are not silent deferrals of an A-stage failure.

Human review accepted this evidence on 2026-09-20. The final classification is
`E5-1A = CLOSED / LOCAL_PRODUCTION_IMPLEMENTATION_QUALIFIED`. E5-1B is
`READY FOR DESIGN / NOT STARTED`; E5-1C is `NOT STARTED / NOT AUTHORIZED`.
Formal execution remains unauthorized and the resource gate is not released.
