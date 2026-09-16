# Formal HR-4E-5 E5-0 authority and cross-pulse interface design

**Status:** `E5_0_AUTHORITY_INTERFACE_READY_FOR_MANUAL_REVIEW`

**Boundary:** design and read-only audit only. `implementation_authorized = false`,
`formal_execution_authorized = false`, and `ENTRY_BLOCKED_BY_S5_FINAL` remains
in force.

## Executive verdict

**Selected single authority:** the `HR4DPulseController` backed by its
`HR4CThreeFieldStore` is the sole canonical slow-medium authority for formal
E5. It owns the authoritative PRE/POST state, pulse index, generation lineage,
and restart metadata. This is not a new scientific model: the class docstring
already defines it as a restart-safe state machine with HR-4C as the sole
authority store.

`StreamingLifecycle` is not selected as an additional store. In a future
formal entry it must be narrowed to a **non-authoritative streaming execution
adapter**: it may schedule screen work, persist restartable *staging* results,
and provide telemetry/barrier evidence, but it must not create a second
CURRENT/POST/NEXT authority or write `authoritative_generation.json`. The
current S3/S4R/S5 `StreamingLifecycle` remains valid qualification evidence;
it is not itself the formal-E5 authority path.

This selection resolves the former dual-authority ambiguity without altering
an optical, HR-2, HR-3, HR-4, or LUT operator. It does not authorize the
orchestration implementation or an HPC run.

## Evidence identity

| Item | Value |
| --- | --- |
| Review source SHA | `075a3f796dc7722fa167cef64b3b0685271382e0` |
| S5 execution baseline SHA | `cd456ff8413cbc041d2d60b9b64007a1554028a1` |
| S5 gate | `244700` was `PENDING (Priority)` at the last read-only snapshot; no action taken here. |
| HR-4D blob | `hr4d_pulse_lifecycle.py`: `8adaf3bdd33093efb9fa0d2f7670dae3adea08b2` |
| HR-4C blob | `hr4c_state.py`: `6f82e2a6291d3ac42deb5a15a56ef5f570ca87d0` |
| Streaming blob | `hr4e5s_streaming.py`: `d6c1341791fdd7fd131136bac7939997e24f94d8` |
| S3 blob | `hr4e5s_s3.py`: `769c9af73bbbdbc817e7680e4648c30c58df83cb` |
| runner blob | `runner.py`: `cfa174197e2aefc3e484b18a33ea05d3efde1a00` |

The inherited qualification config has raw SHA-256
`eaec83ad326a29af95881912db76a7e5c26943dec9e0d6876ca9c9cc2a100c22`.
The E1B delta-n source has raw-byte SHA-256
`70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467`
and array SHA-256 `5990da24bec80937bf3be9985777b3797be9adffedb776dd2f2e840b118d8ad9`.

## Current mechanisms and declared roles

| Component | Present authority behavior | Formal E5 role | Why |
| --- | --- | --- | --- |
| `HR4CThreeFieldStore` | manifest-selected three-field current slot; atomically swaps staging slots | `AUTHORITATIVE_STATE_STORE` and `TRANSACTION_STAGING` | stores `(delta_n,vx,vy)` with one generation lineage and two fixed slots. |
| `HR4DPulseController` | validates phase, pulse index, counters, generation and flow parameters | pulse lifecycle and `PROVENANCE_INDEX` | metadata is tied to the HR4C authoritative generation. |
| `HR4DPulseTransaction` | reads PRE delta-n once and writes POST with unchanged PRE velocity | `TRANSACTION_STAGING` interface to optical/HR-3 | preserves frozen PRE-to-POST velocity semantics. |
| Existing `StreamingLifecycle` | CURRENT/POST/NEXT plus pointer may be authoritative in S3/S4R/S5 | `STREAMING_EXECUTION_ADAPTER` only | no pulse index/rollover; it must not promote an independent generation. |
| `authoritative_generation.json` | identifies Streaming NEXT in qualification roots | `DIAGNOSTIC_ONLY` historical artifact | it must not exist in an HR4C-authoritative formal root. |
| S3 `S3ReadOnlyCurrentState` | read-only delta-n and local POST construction | `READ_ONLY_VIEW` | one-pulse/48-record qualification, not complete E5 authority. |
| runner HR3C path | separate HR-3C lifecycle | non-selected precedent | proves fresh copies but is not HR-4D authority. |

No component may be both a separately advancing authority and an execution
cache. A future adapter artifact must name its HR4C source generation and is
writable only before the corresponding HR4C commit.

## Canonical PRE/POST/next-PRE transaction

```text
HR4C generation 2p, metadata {phase=PRE, pulse_index=p}
  -> FreshOpticalInput(p) = immutable post-lens E_source.copy()
  -> optical reads SlowMediumPRE(p)[k].delta_n exactly once per interval
  -> authoritative HR-3B increment is added to PRE delta_n
  -> HR4DPulseTransaction writes POST_p[k] =
       {PRE.delta_n + increment, PRE.vx, PRE.vy} to HR4C staging
  -> complete all K intervals exactly once
  -> atomic HR4C manifest commit: generation 2p+1, {phase=POST,pulse_index=p}
  -> non-authoritative adapter schedules hydro from this committed POST
  -> per-screen NEXT staging / barrier validates all K outputs
  -> coordinator writes HR4C staging and atomically commits:
       generation 2p+2, {phase=PRE,pulse_index=p+1}
  -> release or retain adapter staging only under a recorded retention policy
```

The first HR4C commit is the sole durable POST authority. The second is the
sole durable `PRE_(p+1)` authority. The adapter may record a barrier receipt
and per-screen hashes, but its NEXT files are not a PRE state until the HR4C
manifest commit succeeds. After that commit, adapter files are historical and
may be released only under a recorded policy.

For the final pulse, the first commit yields `POST_final` with
`run_complete=true`; no hydro transition, pulse-index increment, or extra
interpulse evolution occurs.

## Cross-pulse interface contract

### Optical input

`FreshOpticalInput(p)` is a new copy of the same immutable source captured at
the exact `propagate_one_pulse` input plane, after lens and any permitted
pre-advance. `runner.py` implements `E_source = E` then
`E_pulse = E_source.copy()`; a previous output field is never the next input.

`SlowMediumPRE(p)` is the HR4C manifest-selected three-field generation with
matching `[K, Ny, Nx]`, float64 layout, z edges, `dx`, `dy`, and grid
fingerprint. One interval-centred delta-n view is passed to optical/HR-3 per
schedule; vx/vy remain in the canonical store although the frozen optical
mapping reads only delta-n.

The future entry must validate source/config hashes, z identity, grid
fingerprint, dtype, phase PRE, and pulse index before it opens POST staging.

### Optical/HR-3 POST

`propagate_one_pulse` reads `thermal_slow_state.read_interval(interval.index)`
once, computes authoritative HR-3A/HR-3B output, then calls
`update_interval(interval.index, delta_n_increment)`. The existing
`HR4DPulseTransaction` implements that interface: it writes
`delta_n_post = delta_n_pre + increment` and copies `vx_pre` and `vy_pre`
unchanged into staging. Velocity is therefore **not** updated by optical/HR-3.

Every POST receipt must bind pulse index, source PRE generation/hash, interval
and z identity, config/source hashes, HR-3A/HR-3B authority flags, scalar
deposition-ledger identity, and one-write-per-interval completion. A failure
before complete finalize aborts staging; PRE remains the only authority.

### Hydro and next PRE

The interpulse duration is exactly `1/f_rep`, decomposed by
`build_interpulse_step_schedule(f_rep, dt_hydro)` into full `dt_hydro` steps
plus at most one remainder. The frozen `advance_hr4_single_screen` evolves all
three POST fields. Workers may handle distinct intervals only. A coordinator
verifies screen identity, hashes, completeness, queue emptiness and barrier
receipt before one HR4C staging commit. That commit advances canonical
generation and pulse index exactly once.

## Restart contract

| Restart point | Canonical reopen | Permitted replay / prohibited duplication |
| --- | --- | --- |
| Before optical p | controller constructed with `resume=True`, which calls `HR4CThreeFieldStore.open_existing(...)` | replay fresh optical p; no POST exists. |
| During optical / partial POST staging | `open_existing` detects staging and aborts it | replay the whole optical pulse; no partial POST may become authority. |
| POST complete, before hydro | HR4C `POST,p` generation | do not rerun optical or duplicate POST; begin/recover hydro only. |
| Hydro incomplete | same HR4C `POST,p` plus non-authoritative adapter receipt | validate/reuse completed NEXT staging; replay only missing work; incomplete HR4C staging is discarded/rebuilt. |
| Hydro complete, before promotion | HR4C `POST,p` plus PASS barrier and NEXT staging | rebuild staging if necessary, then make exactly one HR4C commit; never promote adapter pointer. |
| After promotion, before optical p+1 | HR4C `PRE,p+1` generation | start fresh optical p+1; prior adapter files are historical. |
| Between allocations | the same manifest/receipts and run root | resume the state-specific row above; allocation identity is provenance, not a second authority. |

Each reopen must record execution SHA, source/config hashes, HR4C manifest
hash, phase, pulse index, canonical generation, adapter receipt/hash index,
and whether any work was replayed. The actual restart entry is
`HR4DPulseController(..., resume=True) -> HR4CThreeFieldStore.open_existing(...)`;
there is no `HR4DPulseController.open` method.

## PRE_0 three-field candidate contract

| Field | Candidate source / initialization | Layout and interpretation | Provenance / evidence | Classification |
| --- | --- | --- | --- | --- |
| `delta_n` | frozen E1B HR-3B source array | `[K,351,301]`, float64, interval-centred PRE state | raw file and array hashes in Evidence identity; z/grid/source-manifest hashes must be bound | `INHERITED_FROZEN_ARRAY` |
| `vx` | deterministic all-zero float64 initialization during HR4C legacy initialization | same shape/layout; no inherited pulse-induced velocity | `initialize_from_legacy_delta_n` creates a zero batch; 48-screen S3 also uses `zeros_like(selected)` | `FROZEN_ZERO_INITIALIZATION_RULE` |
| `vy` | deterministic all-zero float64 initialization during HR4C legacy initialization | same shape/layout; ambient PRE velocity | same evidence as vx | `FROZEN_ZERO_INITIALIZATION_RULE` |

The candidate is scientifically and lifecycle-consistent with qualified HR-4
initialization, but it is not an executed formal input. A formal PRE_0
preflight must materialize the two HR4C slots non-destructively, calculate and
record raw-byte hashes for all three persisted fields, and bind shape, dtype,
z edges, `dx/dy`, generation `0`, phase `PRE`, and `pulse_index=0`. No missing
physical velocity array is invented; the provenance is the deterministic zero
rule plus the later materialized-file hashes.

## Legal E5-1 engineering canary

**Verdict:** `E5_1_SMALL_CASE_REQUIRES_FULL_Z`.

The legal canary is small in engineering pulse count, not a silently cropped
z domain: `Npulses=3` with the inherited complete frozen schedule and full
slow-state coverage. Two pulses are enough for one handoff, but three are the
minimum to test **two** independent `POST -> PRE_next -> next optical` rollovers
and the `POST_final` terminal rule. This N=3 value is an engineering canary
only; it does not freeze a formal scientific campaign endpoint.

No reduced continuous domain is currently justified. Optical propagation
starts at the frozen input plane, while the only frozen state coverage is the
full `[15000,351,301]` source. The old 48 records are selected qualification
records around a peak, not a propagation/state domain. A prefix or arbitrary
window would require a distinct derived config, source identity, and
equivalence authorization. Therefore the smallest legal currently evidenced
case retains full-z coverage.

The reference is a serial HR4D-authority path with the same full-z source,
PRE_0, schedule, N=3, operators and precision. The candidate integrated path
adds only the non-authoritative streaming adapter. Exact comparison per pulse
must cover fresh-source identity, all PRE/POST three-field hashes, HR-3
deposition ledger, barrier receipt, `PRE_(p+1)` hashes, phase/generation/pulse
counters, no duplicate/omitted intervals, restart receipts, and final
`POST_final`. A physical-trend magnitude is not a canary PASS gate.

## Resource implications of the selected architecture

Let `G = 3*K*Ny*Nx*8`. At the historical illustrative
`K=15000, Ny=351, Nx=301`, `G=35.422 GiB`; this is not a selected formal K.

- **Steady canonical disk state:** HR4C two slots = `2G = 70.845 GiB`.
- **Conditional hydro-transaction disk state:** if adapter NEXT staging is
  retained for per-screen restart until one barrier/HR4C commit, add `G`.
  The resulting `3G=106.267 GiB` is a transaction payload, not a campaign
  peak or retained-data bound.
- **Atomic file transient:** one three-field 2D payload is 2.418 MiB before
  container overhead.
- **PRE_0 caller memory:** legacy delta-n is a disk-backed one-field memmap
  (11.807 GiB addressable payload); zero velocities are allocated as batches,
  not full volumes, by `initialize_from_legacy_delta_n`.
- **Disallowed formal initializer:** current S3 creation materializes
  `selected = np.asarray(source[indices])` and one shared
  `zero = np.zeros_like(selected)` before `StreamingLifecycle.create`. For a
  full volume these are two host-resident fields (23.615 GiB raw payload), in
  addition to the mapped source. The selected architecture must not use it.
- **Optical minimum:** at inherited fp64, one complex `[Nt,Ny,Nx]` field is
  0.605 GiB; immutable source plus working copy is at least 1.209 GiB GPU
  payload before propagation kernels, diagnostics, and transfers.

Checkpoint/history retention determines campaign storage. No full-volume
history retains steady `2G` plus current receipts; selected `C` full
checkpoints add `C*G`; every-pulse checkpointing for N pulses adds `N*G` (or
the expressly named payload) beyond current state. CPU RSS, GPU VRAM, I/O,
quota, QoS limits, and purge policy remain unobserved site-level facts.

## Minimal future implementation request

Only after S5 PASS and explicit authorization, the minimal request is:

| Proposed area | Classification | Required behavior |
| --- | --- | --- |
| new formal entry module | `RUNTIME_ENTRY_ONLY` | construct/reopen HR4D controller, build fresh optical input, pass its transaction to real propagation, and enforce PRE/POST/next-PRE state machine. |
| HR4C-aware streaming adapter | `ORCHESTRATION_ONLY` | create work descriptors and restartable NEXT staging keyed to HR4C POST; no independent CURRENT/POST/NEXT authority or promotion pointer. |
| manifest/receipts | `PROVENANCE_ONLY` | bind PRE_0 three-field hashes, source/config/SHA, phase, generation, pulse and adapter barrier/replay facts. |
| focused tests | `TEST_ONLY` | tiny N=3 fresh-source, exact serial-vs-adapter, all restart boundaries, no-pointer, no-duplicate and `POST_final` checks. |

`propagate_one_pulse`, HR-2, HR-3, `advance_hr4_single_screen`, HR4C
numerics, ionization, Raman, frozen configs, source arrays and LUTs must be
reused unchanged. Any proposal that needs a scientific-operator change is a
blocker, not part of this request.

## Remaining gates and user decisions

S5 terminal PASS and its evidence binding are still required before
implementation/submission. The user must later authorize the formal science
endpoint, source binding, topology (1+4 preferred or 1+2 fallback), retention
and checkpoint policy, resource request after site-limit confirmation, exact
acceptance thresholds, and the minimal implementation task. No decision here
authorizes E5-1 or E5-2 execution.
