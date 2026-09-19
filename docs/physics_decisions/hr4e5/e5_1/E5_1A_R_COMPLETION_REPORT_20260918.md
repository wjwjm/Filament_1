# E5-1A-R formal-entry, admission and recovery closeout

Issued 2026-09-18; implementation and final local validation completed 2026-09-19.

- Branch: `codex/hr4e5-e5-1a-streaming-implementation`
- Repair base: `b9b5f31d0875ee92b9ea98d23a0659a00e18fdde`
- Code/test commit: `217f74587edcdf739127602afd73b2b830fc77bb`
- Original E5-1A base: `131daa541ac37e54d43b29cd3bfc12ba83c3b91d`
- Disposition: **`E5_1A_PARTIAL / BLOCKED_BY_FORMAL_DRIVER_AND_RECOVERY_QUALIFICATION`**

This delivery closes several concrete bypasses but does not satisfy the task's
successful `READY_FOR_REVIEW` exit. The stopping condition is local and
specific: a unique production R/C runner, ordinary R/C receipt ownership, and a
complete non-fixture production GC/terminal positive path are still absent.
No site quota, S5, GPU or HPC uncertainty is being used to hide that code-level
boundary.

## Scope and protected authorities

The existing four-module design remains in place:

| Module | E5-1A-R responsibility |
|---|---|
| `hr4e5_formal_entry.py` | immutable admission identity; PRE0, successor and optical write admission |
| `hr4e5_paired_campaign.py` | durable formal step order, coordinator epoch fencing, receipt revalidation |
| `hr4e5_evidence.py` | complete paired object inventory, actual-array exact and GC-safe metadata validation |
| `hr4e5_storage.py` | one campaign budget, creation intents and scoped GC/terminal inventory |

`hr4e5s_streaming.py`, propagation, nonlinear, ionization, Raman, LUTs,
scientific configuration and frozen evidence are unchanged relative to the
repair base. Runtime orchestration changed=yes; scientific operators changed=no;
Slurm-GPU-S5 actions=no.

## Six requested gaps: before, after and remaining limit

| Acceptance | Before | E5-1A-R result | Evidence level / remaining limit |
|---|---|---|---|
| A05 calling authenticity | Arbitrary callback PASS and incomplete input binding | Formal mode rejects legacy `run_pair/run`; immutable admission identity and complete paired exact are required; the same driver reaches the real tiny CPU optical path | **PARTIAL**: the real call is `TEST_FIXTURE_ONLY`; there is no unique default production R/C runner and ordinary R/C reports are not fully ownership-bound |
| A07 lineage/READY | incomplete epoch and takeover fencing | current active epoch is required; stale coordinator takeover requires a hash-bound exit receipt; successor READY is revalidated against parent generation and admission identity | **PARTIAL**: local subprocess crash/takeover is covered, but the full production pointer/index/handoff path is not instantiated |
| A09 reclamation | generic durable receipts could stand in for GC semantics | formal GC requires completed `gc_plan_id`; plan, writer and verification bind trajectory/pulse/attempt/admission; the binding is checked before unlink and cross-track plan reuse is refused; commit rechecks the plan | **PARTIAL**: writer lifetime is not unified with a production coordinator/takeover registry; positive evidence uses owned local synthetic arrays |
| A14 competition/overage | PRE0/successor could bypass campaign reservation | formal PRE0, successor and optical writes require admission, reservation and intent context; pre-existing targets cannot be retroactively owned | **PARTIAL**: an arbitrary production runner is not yet prevented from creating all ordinary R/C outputs outside the controlled factory |
| A15 final output | terminal subroot could hide campaign orphans | formal terminal inventory is forced to the complete campaign root and requires explicit expected roles; successor and terminal receipts are revalidated on resume | **PARTIAL**: no unique production runner supplies and passes the complete positive role inventory |
| A17 implementation budget | model matched the fixture lifetime only | runtime planning still uses the same object-size rules and hard caps; new intent/report/GC evidence remains inside the positive metadata allowance | **PARTIAL**: model is not yet exercised by a full production driver or measured on the site |

The remaining limits are the task's stop condition; extending this patch into a
new scheduler or storage architecture would exceed the approved L1 repair.

## Admission and write coverage

| Creation point | Required before scientific payload | Durable ownership result |
|---|---|---|
| PRE0 root and three slow fields | formal admission identity, target validation, quota-aware reservation and creation intent | completed intent records trajectory/pulse/attempt/generation and file identity/hash |
| Successor root and `_next_fields()` materialization | verified parent, unique child target, admission and StorageBudget intent | child payload binding plus self-contained READY; duplicate/fork root rejected |
| Streaming optical output | declared config/source/LUT identity when present, schedule/grid/current generation, admission and reservation | final optical and report remain campaign-accounted |
| Paired exact objects | complete contract-derived R/C set and StorageBudget ownership | shape/dtype/finite/canonical hash plus actual `np.array_equal`; descriptor and report hashes agree |
| GC targets | completed exact, READY, dependency and writer evidence; original plan and whitelist | only owned registered files; plan is bound to R/C, pulse, attempt and admission before unlink and is rechecked at commit |
| Terminal inventory | complete campaign root and explicit expected role map | unregistered NPY/NPZ/sink/orphan or role mismatch refuses completion |

This table is an implemented gate map, not proof that a full production runner
has exercised every row. Ordinary R/C durable reports remain the main ownership
gap.

## Exact to READY to GC evidence chain

The formal paired validator derives the expected screen, sink, ledger and final
optical keys from `N`, `K` and pulse position. While arrays exist, it loads both
tracks and requires matching shape, dtype, finite status, canonical array hash
and `np.array_equal`. After an authorized GC, metadata-only recovery is allowed
only with the same `StorageBudget`, completed creation ownership, unchanged
file descriptors and equal R/C canonical hashes. Missing hashes are rejected.

Successor receipts bind child root, parent root/generation and admission
identity. Formal GC then requires the original completed plan and rechecks the
writer receipt plus trajectory/pulse/attempt/admission identity before unlink;
completion verification re-reads the prerequisite receipt. The focused
test creates separate R and C plans, deletes their owned target sets, rejects
using the R plan as the C plan, and reopens the exact receipt metadata-only.
This is local synthetic-array evidence; it is not a formal scientific GC run.

## Recovery and process evidence

The local crash/takeover test launches two independent subprocesses. The first
exits with `os._exit(17)`; the second may take over only with the persisted,
hash-bound stale-coordinator receipt. Old epochs cannot record or commit, the
admission hash and `next_pair_index` remain stable, and no second successor is
created. The N=3/K=32 integration fixture continues to use the same
`FormalPairedDriver`, original Streaming lifecycle and CPU hydro worker. Its
optical operation remains an explicit deterministic test double.

The fixture exact counts are:

- 1,344 screen arrays (`480 + 480 + 384`);
- 27 ledger arrays;
- 3 final optical arrays.

R01's separate tiny test reaches `run_streaming_optical_pulse` without an
injected propagation double, but retains `TEST_FIXTURE_ONLY` identity. Neither
test is a K8048 or GPU scientific equivalence result.

## Local validation

The authoritative structured record is
[E5_1A_R_TEST_RESULTS.json](E5_1A_R_TEST_RESULTS.json).

| Check | Latest result on code/test commit |
|---|---:|
| backend | PASS |
| focused E5-1A-R | 13 passed, 368 deselected |
| all E5-1A | 43 passed, 338 deselected |
| existing Streaming/HR4D | 34 passed, 347 deselected |
| sanity | 1 passed |
| explicit compileall | PASS |
| `git diff --check` | PASS |

All tests use the repository wrapper and the isolated Windows CPU interpreter.
The retained logs were run at checkout HEAD
`e0ed1c57900fc5ef9d0a4fdaa22d79c7d6c2abcf`; its changes after code/test commit
`217f74587edcdf739127602afd73b2b830fc77bb` are qualification documents only,
so the tested runtime and test files are byte-identical to the code/test commit.
No GPU, Slurm, HPC, S5 or K8048 scientific array was used.

## Budget and retention

The model remains:

- campaign hard cap: `322122547200` bytes (300 GiB);
- modeled peak: `299676844288` bytes;
- modeled headroom: `22445702912` bytes;
- final-output cap: `68719476736` bytes (64 GiB, included in the campaign cap);
- modeled default final retention: `47504725760` bytes;
- modeled final headroom: `21214750976` bytes.

The model includes both tracks, one source-copy allowance, metadata, GC and
failure receipts, one failed-handoff slow generation and positive safety
margin. New E5-1A-R management records stay within the existing 2 GiB metadata
allowance. These are scalar formula results, not allocated-block, quota, RSS,
VRAM or I/O measurements. The known successor `list + np.stack` host transient
and PRE cache limitation remain for B.

## A01-A18 disposition

| Item | Status | Current boundary |
|---|---|---|
| A01 | PASS | fixed branch/base and protected authorities |
| A02 | PASS_FIXTURE | N3/K32 independent tracks |
| A03 | PASS_FIXTURE | per-track velocity inheritance |
| A04 | PASS_FIXTURE | CPU event ordering; not GPU overlap |
| A05 | PARTIAL | no unique production R/C runner |
| A06 | PASS_LOCAL_BOUNDARY | final POST replay/refusal |
| A07 | PARTIAL | local epoch/takeover improved; full production handoff absent |
| A08 | PASS_FIXTURE | actual 1344/27/3 exact in fixture |
| A09 | PARTIAL | scoped GC gates closed; production positive chain absent |
| A10 | PASS_FIXTURE | owned parent reclaim and reopen |
| A11 | PASS_LOCAL_BOUNDARY | independent local process recovery |
| A12 | PASS_LOCAL_BOUNDARY | path/link/identity protection |
| A13 | PASS_LOCAL_BOUNDARY | exact 300 GiB hard cap |
| A14 | PARTIAL | large helper writes gated; arbitrary production runner remains |
| A15 | PARTIAL | full-root gate exists; complete production role inventory absent |
| A16 | PASS_LOCAL_BOUNDARY | fresh-process completed-pair resume |
| A17 | PARTIAL | formula and fixture only, not full production lifecycle/site measurement |
| A18 | PASS_LOCAL_BOUNDARY | 34 existing regressions and local metadata scope |

## Remaining blocker and next gate

The implementation remains blocked by four related local items:

1. no unique default production runner/factory closes R, C, exact, successor,
   GC and terminal through one trusted object model;
2. ordinary R/C receipts are not uniformly bound to completed StorageBudget
   creation intents and the complete contract-derived object set;
3. writer lifetime evidence is not unified with the production coordinator and
   takeover registry, so stale writer semantics remain a formal blocker;
4. there is no complete non-fixture positive production run of GC and terminal
   inventory;
5. config/source/LUT declarations are not mandatory path/hash identities in
   every possible production runner path.

Therefore the final state remains
`E5_1A_PARTIAL / BLOCKED_BY_FORMAL_DRIVER_AND_RECOVERY_QUALIFICATION`.
E5-0 remains `CLOSED / STREAMING_PREPARATION_CONTRACT_ACCEPTED` with the
user-fixed existing Streaming architecture. Formal execution is false and the
resource execution gate is not released. E5-1B/C are not started or authorized.
