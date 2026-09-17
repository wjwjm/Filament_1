# E5-1A storage/retention contract

Issued 2026-09-16; local implementation work 2026-09-17. This document applies only
to the scoped A implementation. E5-0 remains CLOSED. B/C scientific execution,
HPC data deletion, and S5 operations are not authorized.

The statements below are acceptance requirements. Implementation and executed
qualification are reported separately in the implementation report and test
results; this contract alone does not certify that every requirement passed.

## Authority and limits

The candidate uses the existing Streaming CURRENT/POST/NEXT authority, block 8,
queue 16, original claims and promotion. Reference is isolated serial/Batch.
The outer pair index records progress and references; it is not physical state.

`max_campaign_live_bytes = 322122547200` (300 GiB) is a single campaign cap across
both trajectories and every created path, including attempts, temporary files,
handoff copies, diagnostics, metadata and residues. Overrides may only reduce it.
`final_output_budget_bytes = 68719476736` (64 GiB) is a second, included limit.
Neither is a site availability or quota assertion. Protected pre-existing source
and LUT files cannot be reclaimed; newly copied versions count against the cap.

Every prospective large write must reserve its maximum incremental live bytes
under a process-safe lock. Reservations represent unmaterialized bytes, not an
extra count of materialized files. Recheck actual logical/allocated bytes,
unmaterialized reservations, positive safety allowance, filesystem free bytes
and verified quota independently. Unknown production quota fails admission;
explicit mock quota is confined to CPU fixtures. Sparse allocation is not a
license to ignore eventual logical payload. Preserve capacity for receipts,
failure summary and orderly stopping. Never credit a rename as freed storage.

## Paired lifetime

| Phase | Required state/evidence | Advance/release gate |
|---|---|---|
| Initialize | Independent R PRE and C CURRENT, common immutable input identity | Complete self-contained READY, no partially initialized root consumed |
| R_p | R reads its own PRE, produces its own POST/NEXT and six sink arrays | R completion; no candidate answer substitution |
| C_p | Actual CURRENT read; durable POST immediately eligible for existing hydro | Nonfinal existing barrier/promotion; final POST-only completion |
| Exact_p | PRE/POST, nonfinal NEXT, six sinks, nine ledgers and final optical | Every declared object compared for shape/dtype/hash/array equality and finite values |
| R handoff | Parent NEXT plus separately copied child PRE coexist | Array-exact binding, durable child READY, parent summary, no active writers |
| R archive/reclaim | Immutable parent manifest/pointer and comparison records retained | Only owned, whitelisted, no-longer-dependent payload files |
| C handoff | Parent NEXT plus separately copied child CURRENT coexist | Same independent binding and READY requirements; never borrow R state |
| C archive/reclaim | Parent archived, child fully reopenable without parent arrays | Completed exact and durable plan; all scientific/state-holder processes quiescent |
| New process | All first-pair scientific state holders exited | New process uses persisted child READY, inputs and pair progress only |
| Final | Candidate complete CURRENT+POST, no NEXT, bounded outputs | POST_FINAL_READY; no fictitious fourth PRE or redundant checkpoint |

Handoffs are sequential so two new generations are not simultaneously copied
before reclamation. Failed comparison stops before any next pair/reclamation.
Partial initialization and failed attempts remain charged and cannot be cleaned
by the successful-data reclaimer. No unbounded retries.

## Exact before reclaim

Each nonfinal pair compares 15K screen arrays; the final pair compares 12K.
N=3 therefore gives 42K = 338016 at K=8048, plus 27 ledger arrays and three final
optical arrays. Each trajectory independently validates 6K successor binding
fields across its two handoffs. Counts are derived from actual object manifests.
Canonical array identity uses existing `hr4e_timestep.sha256_array`; scientific
equality additionally requires actual `np.array_equal` on loaded arrays. Raw file
hashes serve transport/ownership and need not match across NPZ metadata layouts.

Preserve both trajectories' per-pulse final optical fields, ledgers/summaries,
complete comparison records, input/config/LUT/schedule identities, lineage,
READY/archive/reclaim records, event/provenance and resource summaries. Candidate
final complete CURRENT+POST is the last restart point. Intermediate sinks and
reference slots/large states may be reclaimed only after all gates. Once reclaimed,
the historical actual-array comparison is an executed result, not a promise that
hashes can reproduce the deleted arrays or rerun all historical comparisons.

Optional diagnostic arrays/checkpoints must be declared and budgeted in advance;
they are not enabled by default. Output growth is N-dependent even when the active
large-state window is bounded.

## File ownership and reclaim transaction

Creation registration binds campaign, trajectory, pulse, attempt, role, relative
path and expected file identity/content. A path merely located under the root is
not owned. Protect external sources, old campaigns, unknown files and unrelated
user data. Reject traversal, path-prefix collisions, symbolic links, Windows
reparse/junction points, unknown hardlinks and changed file identity. Recheck at
execution. Delete individual approved files only; do not recursively delete a
root or scan by age.

Only one coordinator plans/applies recovery-aware deletion. Preconditions include
successful complete exact, durable self-contained successor READY (or final
POST_FINAL_READY), terminated writer identities with PID-reuse/epoch distinction,
and zero future recovery/acceptance dependence on the payload. The new checkpoint
contains its own arrays and the durable parent-to-child exact receipt; its reopen
does not require deleted parent arrays.

Archive parent manifest/pointer before deletion and mark the old root
`ARCHIVED_AFTER_EXACT / NOT_RESTARTABLE` outside its original manifest. Active
reopen must refuse it. Never modify a historical manifest to disguise missing
payload. Keep plan/preconditions and per-file progress durable. A missing file is
success only when explained by that original reclaim transaction; unknown missing
files fail closed. Resume never expands the original authorized list. Released
bytes become available only after deletion is confirmed. Failed/unknown safety
evidence leaves files intact and stops the next pair.

## Evidence limits

A validates controlled CPU fixtures, explicit mock capacity providers and fresh
local subprocess persistence. It does not establish real GPU overlap, Slurm
allocation/node/power-loss survival, actual quota/RSS/VRAM/I/O or formal materialized
PRE0. B must close environment, inputs and resource admission. C executes the
separately authorized scientific canary. Actual cross-allocation endurance belongs
to later authorized execution, not the A subprocess label.

## Implementation disposition

The current delivery is E5_1A_PARTIAL. The rules above are not a claim of full
formal-run qualification. Low-level storage/GC refusal and recovery tests plus
the controlled integrated fixture cover only their explicit call paths. Formal
PRE0/successor creation admission, arbitrary caller ownership registration and
the combined interrupted pair/handoff/reclaim coordinator require the remaining
outer-entry work identified in the implementation report. Do not use this
partial delivery to authorize formal data reclamation.
