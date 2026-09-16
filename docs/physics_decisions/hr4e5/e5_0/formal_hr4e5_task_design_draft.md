# Formal HR-4E-5 task design and authorization map

**Status (2026-09-16):** `E5_0_AUTHORITY_INTERFACE_READY_FOR_MANUAL_REVIEW / ENTRY_BLOCKED_BY_S5_FINAL`

**Boundary:** documentation and read-only audit only. No code, config, source,
LUT or result modification; no run-root creation; no Slurm/GPU action; and no
S5-FINAL intervention is authorized. S5-FINAL job `244700` was `PENDING
(Priority)` at the latest read-only snapshot; that queue state is not an E5
result.

## Current design verdict

Formal E5 uses exactly one canonical slow-medium authority:

```text
HR4DPulseController + HR4CThreeFieldStore
```

The HR4C manifest-selected three-field slot is the only canonical PRE or POST
state. It owns the state generation; HR4D binds phase and pulse-index metadata.
Future Streaming is constrained to a `STREAMING_EXECUTION_ADAPTER`: it can
persist restartable per-screen staging, claims and barrier receipts, but cannot
own a pulse index, promote an authority, or write
`authoritative_generation.json` in the formal root.

```text
PRE_p (canonical HR4C)
 -> fresh E_source.copy()
 -> real optical propagation reads PRE.delta_n
 -> HR-3 POST: {delta_n + increment, vx, vy}
 -> atomic canonical POST_p
 -> non-authoritative hydro staging and barrier
 -> atomic canonical PRE_(p+1), one pulse-index/generation advance
```

The final pulse is `POST_final`; it has no interpulse evolution. The actual
restart constructor is `HR4DPulseController(..., resume=True) ->
HR4CThreeFieldStore.open_existing(...)`, not a nonexistent controller `open`.

## Stage hierarchy

The old D0--D5 ideas are checklists, not a new serial six-stage project.

| Stage | Objective and verdict | Former checklist mapping | Boundary |
| --- | --- | --- | --- |
| **E5-0** | Authority, interface, PRE_0, resource and minimal-glue design. `READY_FOR_MANUAL_REVIEW`. | D0 evidence binding; D1 source/claim binding; D2 read-only audit; D3 planned local/preflight gates. | Design may continue while S5 is pending. Implementation and submission remain blocked. |
| **E5-1** | Legal small multi-pulse engineering closure. `E5_1_SMALL_CASE_REQUIRES_FULL_Z`; proposed `Npulses=3`. | D3 tiny validation; D4 controlled comparison/restart; D5 engineering closeout. | Requires S5 PASS, reviewed glue, PRE_0 hashes, resources/retention and explicit authorization. |
| **E5-2** | Full-z endurance/restart production-scale validation and scientific closeout. `NOT_STARTED`. | D4 production allocation(s); D5 terminal postprocess/evaluation. | Requires E5-1 PASS and a separate science/resource/acceptance authorization. |

No stage silently adopts a scientific pulse count, cropped longitudinal domain,
new input plane, or S3's 48 selected records.

## E5-0 design package

- `formal_hr4e5_e5_0_authority_interface_design_20260916.md` selects the
  HR4D/HR4C-only authority and states PRE/POST/next-PRE/restart rules.
- `formal_hr4e5_e5_0_authority_contract.json` is the implementation-neutral
  contract.
- `formal_hr4e5_e5_0_freeze_candidate.json` records PRE_0, E5-1, resources and
  later authorization requirements.
- `formal_hr4e5_e5_0_resource_budget.csv` separates steady authority state,
  transient staging, caller allocations, GPU fields and policy-based retention.
- `formal_hr4e5_e5_0_minimal_glue_implementation_draft.md` is a future,
  orchestration-only request, not implementation authority.

## PRE_0 and E5-1 decisions

`PRE_0` is a three-field rule, not a delta-n filename:

| Field | Candidate source/rule | Status |
| --- | --- | --- |
| `delta_n` | E1B HR-3B array: raw SHA `70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467`; array SHA `5990da24bec80937bf3be9985777b3797be9adffedb776dd2f2e840b118d8ad9` | `INHERITED_FROZEN_ARRAY` |
| `vx` | deterministic float64 zero batches in existing HR4C initializer | `FROZEN_ZERO_INITIALIZATION_RULE` |
| `vy` | deterministic float64 zero batches in existing HR4C initializer | `FROZEN_ZERO_INITIALIZATION_RULE` |

The common historical illustrative layout is `[15000,351,301]` float64,
`dx=dy=1e-5 m`, full frozen z identity, phase PRE, generation 0 and pulse
index 0. A later preflight must materialize and raw-hash all three fields
non-destructively.

E5-1 is small only in pulse count: N=3 produces two independent cross-pulse
rollovers and `POST_final`. It must retain the full frozen optical schedule and
complete slow-state coverage. The reference is serial HR4D authority; the
candidate uses identical inputs plus the non-authoritative adapter. Per-pulse
exact checks cover fresh-source identity, PRE/POST/next-PRE hashes, deposition
ledger, barrier evidence and phase/generation/pulse counters. Physical-trend
magnitude is diagnostic, not an engineering PASS threshold.

## Resource and site boundary

For illustrative `K=15000, Ny=351, Nx=301`, a three-field generation is
`G=35.422 GiB`. Selected steady HR4C state is `2G=70.845 GiB`; conditional
full adapter NEXT staging makes one transaction `3G=106.267 GiB`. Neither is a
campaign peak or quota claim. The old S3 initializer would materialize full
`selected` and `zero` host arrays (23.615 GiB raw) in addition to its mapped
source, so it is excluded. One fp64 complex optical source plus working copy
requires at least 1.209 GiB GPU payload before kernels and diagnostics.

No full-volume history is `2G` plus receipts/selected diagnostics; `C` selected
full checkpoints add `C*G`; every-pulse checkpoints add `N*G`. Filesystem
overhead, CPU RSS, GPU VRAM, diagnostics, I/O, worker buffers and temporary
files remain extra. Quota, purge policy, numeric normal-QoS limits, GPU
model/VRAM/CUDA, observed RSS/VRAM/I/O and formal retention are unverified.

Historical 48-screen timing is qualification evidence only. Tested topology is
1 optical + 4 hydro GPUs, with 1 + 2 fallback; 1 + 6 remains
`NOT TESTED / RESOURCE_UNAVAILABLE`. These are not full-z estimates or selected
formal resources.

## S5 gate and later authorization

Before any E5-1 implementation or execution, record:

1. S5-FINAL terminal PASS, terminal `sacct`, exact/provenance closeout and
   F01--F06 SHA inheritance.
2. A user-authorized minimal orchestration implementation; a
   `SCIENTIFIC_OPERATOR_CHANGE` is a blocker.
3. Non-overwriting PRE_0 materialization/hashes with z/config/source/metadata.
4. A retention/checkpoint policy and confirmed QoS/quota/resource request.
5. Tiny N=3 serial-versus-adapter exact and restart tests, followed by local,
   batch-entry and remote-provenance gates.
6. Separate user authorization before a run root, allocation or `sbatch`.

S5 PASS releases none of these by itself. E5-2 also needs accepted E5-1
engineering closure and a separately stated scientific endpoint, comparison,
pulse count, diagnostics/retention and acceptance contract.
