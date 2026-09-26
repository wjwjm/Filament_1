# E5-1B WP2 recovery redesign contract

Status: R0 design freeze for local implementation. Scope: WP2 recovery control plane only. Baseline: `079deb38f92bd5d25b952561440f85ac15efb34c`; copied multiprocess snapshot: `d75aac68c3c59fc2742569ad0d20f623a1fff49a`. This document does not authorize formal K8048, GPU preflight, or E5-1C.

## Authorities and stable identities

- `StorageBudget` owns a **single aggregate Candidate reservation** for a pulse's optical output and Streaming state. Its aggregate creation intent proves allocation, campaign, admission, trajectory, pulse, logical attempt and generation. It is not a child-process permission. Overlap and cap checks stay fail-closed.
- `StreamingLifecycle` remains the sole CURRENT/POST/NEXT, claim, queue, barrier and promotion authority. WP2 invokes its existing `reconstruct_queue()` only after old writers are quiescent. It does not create another recovery queue or NEXT generation.
- Formal `coordinator_epoch` is execution provenance. It never changes a scientific generation or logical scientific attempt. Existing `attempt=0` is the logical scientific attempt for this campaign. Child launch/replay attempts are execution attempts recorded separately by grant/output record.
- `scientific_pair_id` is SHA256 of canonical JSON with schema, `campaign_id`, admission `identity_sha256`, pulse index, logical attempt 0, pair role `R+C`, the immutable source `current_generation` values of both trajectory lifecycle roots, and the formal config SHA already bound by admission. It is persisted before R/C execution and reused after coordinator restart; a changed input/generation/config/pulse produces a different ID. The original source generation is read from the manifest field, not its promoted authority pointer.
- `implementation_identity` is a versioned manifest of every production `.py` file under `Filament_python/KHz_filament`, including dirty tracked and formerly untracked child modules, plus the hash helper `Filament_python/tools/hpc_ops/provenance_v2.py`. Per-file canonical-LF SHA256 uses that repository helper; the aggregate is SHA256 of canonical JSON containing the sorted relative paths and per-file hashes. The manifest also records Python and relevant dependency versions. Each process recomputes and compares it before a child grant binds or a trajectory is reused. Git branch/HEAD/time/PID do not substitute for source bytes. A source mismatch is `REUSE_REJECTED / IMPLEMENTATION_IDENTITY_MISMATCH`.
- Admission `identity_sha256` already binds config, source, LUT, schedule, grid, precision, topology request and budget. It includes operational fields, so it is an input to the pair ID rather than the pair ID itself. Cross-epoch reuse additionally checks implementation identity, scientific pair identity, trajectory, generation, allocation paths/bytes, object schema, report/object-set hashes and writer quiescence.

## Writer grants and handoff

The aggregate Candidate intent has one reconciler and bounded child grants: exactly one `OPTICAL` and up to the admitted H indexed `HYDRO` grants per execution wave. A grant records nonce/id, role/index, pair/admission/implementation identity, trajectory/pulse/generation/epoch, reservation/intent, permitted operation/domain and paths, state, actual PID/start token, bind time and close/fence evidence. `RECONCILER` is a separate manager role and cannot claim blocks or create optical science output.

Parent creates `PENDING` under the storage lock and passes the grant id/nonce to one child. Child reads its own process identity and atomically binds `PENDING → BOUND_ACTIVE` before any science write or hydro claim; wrong role/scope/epoch, duplicate bind or different implementation is rejected. Parent confirms bound identity. `PENDING` can become `ABANDONED` only with launch/death evidence; `BOUND_ACTIVE` becomes `CLOSED`, `INTERRUPTED`, or `INTERRUPTED_STALE` only with matching owner/epoch or verified DEAD evidence. LIVE/UNKNOWN always block takeover. A coordinator crash in the handoff interval leaves PENDING inspectable and recoverable only after its launch candidate is proved absent/dead.

Optical grant authorizes optical output and POST writes; HYDRO grant authorizes the indexed consumer's NEXT commit operation through StreamingLifecycle, not an unrestricted copy of `[output,state]`. The shared manifest is serialized by Streaming's lock. Writer grant metadata cannot create another scientific allocation. Completion/consume require child grants closed or fenced; the manager closes before aggregate intent completion.

## Coordinator outcome and takeover

Coordinator outcomes distinguish `COMPLETED`, `PAUSED`, `FAILED_RECOVERABLE`, and OS-stale (`ACTIVE` with DEAD process). The fresh epoch's unified recovery admission inspects prior outcome, coordinator identity, every grant, intent/reservation, Streaming state, output locator and saved R/C receipts. It fences verified DEAD grants, rejects LIVE/UNKNOWN, and records a durable decision. `active_coordinator=None` after clean failure never bypasses this scan. An old ACTIVE coordinator that is LIVE blocks a second coordinator.

| Prior coordinator | Prior grant | Decision |
| --- | --- | --- |
| ACTIVE and LIVE | any | reject takeover |
| FAILED_RECOVERABLE / PAUSED / ACTIVE but DEAD | LIVE or UNKNOWN | reject takeover |
| FAILED_RECOVERABLE / ACTIVE but DEAD | BOUND_ACTIVE with verified DEAD | fence; retain reservation and evidence |
| Failed or stale | PENDING with child candidate LIVE/UNKNOWN | reject; no speculative abandonment |
| Failed or stale | all CLOSED, ABANDONED with proof, or DEAD/fenced | admit matching-scope rebound and Streaming reconstruction |
| COMPLETE, all quiescent | none | report already complete; no replay |

The Candidate reservation/intent is rebound only when campaign, pair, admission, implementation, pulse, logical attempt, generation, paths, expected bytes, dtype/shape and schema match exactly. It keeps the same reservation ID and charged bytes; no second overlapping reservation or cap increase is permitted.

## Output attempts and pair transaction

Original and replay optical outputs are immutable execution-attempt directories. A durable output-attempt record binds logical trajectory, pair id, pulse/generation, logical attempt, producer epoch/grant, actual path, admission/implementation identity and validated object set/hash. A single `ACTIVE_OUTPUT` locator points to exactly one **qualified** attempt and retains prior locator hash/history. Ambiguous qualified candidates or changed bytes reject reuse. `_object_descriptor()` resolves through this locator; it never assumes the fixed first optical directory after recovery. Prior failed output stays in place.

R and C independently become `COMMITTED` only after their own durable report, output locator and object set pass validation. Peer failure does not roll back the successful side. A crash between science durability and formal commit requires a hash-bound reconciliation path, not a directory-presence shortcut. Once both sides qualify, the coordinator writes an immutable `PAIR_LINEAGE_RECEIPT` with shared pair/admission/implementation/source/config/LUT/schedule identities and each side's epoch, grant/output locator, object set and receipt hash. `cross_epoch=true` is legal when all shared identities and implementation hashes match; R(epoch0)+C(epoch1) and the symmetric case are both allowed. The lineage receipt is validated before strict exact. Exact PASS alone then gates successor/terminal/reclaim as before.

After old writers are quiescent and the reservation rebound is durable, WP2 calls `StreamingLifecycle.reconstruct_queue()` for POST-without-NEXT, launches bound replacement hydro children, and requires unique NEXT, empty queue/backlog, barrier and promotion. Repeated restart preserves the same pair id and reservation and cannot create duplicate ACTIVE_OUTPUT or NEXT.

## Stop conditions

Stop local implementation and report `RECOVERY_REDESIGN_BLOCKED` if any path requires physics or Streaming scientific-semantic change, H copies of full-z reservation, disabling overlap protection, relaxed exact, deletion/overwrite of old scientific output, or an ambiguous pair/output authority. Local tests and review evidence do not release K8048, physical-GPU preflight or E5-1C.
