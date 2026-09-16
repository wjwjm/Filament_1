# Formal HR-4E-5 E5-1 legal small-case candidate

**Verdict:** `E5_1_SMALL_CASE_REQUIRES_FULL_Z`

**Execution status:** `NOT_AUTHORIZED_TO_RUN`

## Purpose and boundary

E5-1 is an engineering closure/restart canary, not the formal scientific
endpoint. It must prove fresh optical inputs and repeated slow-medium
inheritance without changing frozen operators, precision, source meaning, or
the final `POST_final` rule. Its later execution remains blocked by S5-FINAL
PASS, implementation authorization, and resource authorization.

## Proposed canary

| Item | Candidate | Rationale |
| --- | --- | --- |
| Pulse count | `Npulses=3` | N=2 proves one handoff; N=3 is the minimum that proves two independent rollovers `POST_0 -> PRE_1 -> optical_1` and `POST_1 -> PRE_2 -> optical_2`, then terminal `POST_final`. |
| Domain | complete inherited full-z schedule | the optical field begins at the frozen input plane and the only frozen state coverage is full `[15000,351,301]`. |
| Precision/operators | inherited fp64 and frozen optical/HR-2/HR-3/HR-4 operators | engineering path must not alter scientific behavior. |
| PRE_0 | E1B delta-n source plus deterministic zero vx/vy initialization | complete three-field candidate, described below. |
| Topology | not selected for execution | later choose qualified 1+4 preferred or 1+2 fallback after resource authorization. |

The word *small* describes the N=3 engineering pulse train, not a reduced
state domain. The existing S3/S4R/S5 48-screen record set is explicitly not
this case's state domain.

## Why a reduced domain is not legal yet

The selected 48 records are peak-centred qualification samples. S3 still runs
the complete frozen optical schedule and only commits those selected records.
They cannot initialize the first optical interval or provide slow state for
every interval between the frozen input and an arbitrary endpoint.

The current frozen provenance establishes a complete source only for the full
longitudinal schedule. A prefix, a cropped peak window, or a new input plane
would require a separately derived config/state manifest and a reviewed
equivalence argument. No such artifact exists. Therefore E5-1 must retain
full-z until an independent, authorized reduced continuous input exists.

## PRE_0 and coverage contract

`PRE_0` is all three HR4C fields at the same `[K,Ny,Nx]` float64 layout:

- `delta_n`: inherited E1B source, raw file SHA-256
  `70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467`,
  array SHA-256 `5990da24bec80937bf3be9985777b3797be9adffedb776dd2f2e840b118d8ad9`.
- `vx`, `vy`: deterministic float64 zero initialization by HR4C's existing
  batch initialization rule. The rule has HR4C and 48-screen S3 qualification
  evidence, but the future materialized full-volume files still require their
  own raw hashes in the preflight manifest.

The full z edges, `dx=dy=1e-5 m`, config hash, source-manifest hash,
generation 0, phase PRE and pulse index 0 must be checked before optical p=0.

## Reference and exact-comparison design

The reference path is serial HR4D authority. The integrated candidate shares
exactly the same source, PRE_0, N=3, schedule, config, precision, and frozen
operators, but executes hydro work through the non-authoritative adapter.

Exact comparisons are required for each pulse:

1. fresh source identity and non-aliasing;
2. PRE three-field hashes and HR4C generation/phase/pulse metadata;
3. POST three-field hashes, HR-3 authority flags, and deposition-ledger
   identity;
4. barrier receipt plus adapter queue/completeness/no-duplicate evidence;
5. next PRE three-field hashes and exactly one counter/generation transition.

Global gates require all K intervals once per pulse, no cross-pulse namespace
collision, restart receipts at declared interruption boundaries, two
interpulse evolutions, three POST commits, and terminal `POST_final`. Physical
trend amplitude is diagnostic only, not an engineering PASS gate.

## Candidate storage and runtime evidence

For the historical illustrative full geometry, one three-field generation is
`G=35.422 GiB`. The selected authority has steady HR4C storage `2G=70.845 GiB`.
If all non-authoritative per-screen NEXT staging is retained until a barrier,
the hydro transaction adds `G`, producing 106.267 GiB raw payload before
headers, diagnostics, checkpoints, optical arrays, worker buffers, and run
history. This is neither a quota claim nor a campaign peak bound.

Historical 48-screen timing is not a full-z estimate. No observed CPU RSS,
GPU VRAM peak, I/O, quota, purge policy, GPU type, or QoS numerical limit is
available for sizing the canary allocation.

## Blockers and next gate

The candidate is blocked by: S5-FINAL terminal evidence; a separately reviewed
minimal orchestration implementation; materialized PRE_0 three-field hashes;
resource/retention authorization; and normal-QoS/quota confirmation. It is
not authorized to create a config, run root, staging worktree, or Slurm job.
