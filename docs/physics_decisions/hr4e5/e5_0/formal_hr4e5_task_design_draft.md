# Formal HR-4E-5 task-design draft

## Status and purpose

**Status (2026-09-16): `DESIGN_DRAFT / ENTRY_BLOCKED_BY_S5_FINAL`.**  E5-0
has reached `E5_0_READY_FOR_MANUAL_REVIEW`, but this remains a design draft
and no formal execution is authorized.  This document starts the design work
for the formal HR-4E-5 campaign while
HR-4E-5S S5-FINAL waits for its sole integrated worker-loss qualification.
It does not authorize a code change, a source/configuration change, an HPC
staging action, a Slurm submission, or any physical conclusion.

At this snapshot, initial S5-FINAL job `244700` is `PENDING (Priority)`;
its controller is `WAIT_FOR_DUAL_CLAIMS`, `signal_sent=false`, and it has no
recovery job.  This is a queue state, not an S5 result.

The term **formal HR-4E-5** below means the subsequent multi-pulse and
durable formal scientific design and execution.  It is distinct from
HR-4E-5P (single-screen/block/topology qualification) and HR-4E-5S
(Streaming lifecycle and restart qualification).

## Research position and frozen boundary

Formal E5 is first an engineering and provenance task: demonstrate a real
multi-pulse closure with fresh optical fields and inherited slow medium,
full-z execution, restart, memory/resource stability, and endurance.  Under
one separately predeclared physical condition it will record propagation and
slow-state observables, but formation of a particular physical accumulation
trend is not a necessary engineering PASS condition.

It must use the frozen HR-2/HR-3/HR-4 scientific contract:

```text
PRE_p -> fresh immutable E_source copy -> one optical propagation / POST_p
      -> exactly 1/f_rep of HR-4C evolution -> PRE_(p+1)
...
POST_final (no extra final evolution)
```

`(delta_n, vx, vy)`, PRE/POST/NEXT ownership, barrier and promotion semantics,
the frozen `advance_hr4_single_screen` operator, precision, source definition,
and exact-comparison semantics are not tuning variables for this campaign.
The historical HR-4E-2 velocity-field convergence caveat remains visible; a
formal E5 result cannot silently close it.

The E5-0 evidence packet is
[`formal_hr4e5_e5_0_readonly_audit_20260916.md`](formal_hr4e5_e5_0_readonly_audit_20260916.md).
It records that the current entry is `EXISTING_ENTRY_PARTIAL`: HR-4D and
Streaming lifecycle pieces exist, but no production entry joins real full-z
optical/HR-3 POST, HR-4D authority, and cross-pulse Streaming rollover.  The
missing item is reviewed orchestration glue, not permission to alter a frozen
operator.

The preferred tested execution topology is 1 optical + 4 hydro GPUs, with
1 optical + 2 hydro GPUs as the validated lower-resource fallback.  These are
execution qualifications, not scientific parameter choices or performance
claims for formal E5.  The unexecuted 1+6 point remains
`NOT TESTED / RESOURCE_UNAVAILABLE` and is outside this design.

## Entry gate

Formal E5 may move from `ENTRY_BLOCKED_BY_S5_FINAL` to
`ENTRY_READY_FOR_AUTHORIZATION` only when all of the following are recorded:

1. S5-FINAL controller is `PASS`, and `sacct` confirms terminal evidence for
   the initial and the one permitted recovery job.
2. The integrated test has two distinct hydro actors holding different active
   claims before a targeted external loss; old-job quiescence is proven before
   recovery bootstrap.
3. The frozen interruption inventory and `expected_recovery_effects` predate
   reconstruction; bootstrap precedes all recovery hydro claims; recovery
   finishes 48/48 screens.
4. The complete S5-FINAL exact/provenance audit passes, including the 432
   POST/NEXT/deposition-array comparisons, optical and ledger evidence,
   ownership/barrier/promotion/manifest checks, recovery provenance, and no
   erroneous authoritative temporary/orphan artifact.
5. The S5 closeout identifies the F01--F06 inheritance by SHA rather than
   claiming that all six faults were rerun at one SHA.

If S5-FINAL enters `READY_FOR_S5_FINAL_DEFECT_REVIEW`, this design remains
frozen at the current draft.  No formal E5 workaround, alternate executor, or
new run is implied.

## Inheritance versus new authorization

The E5-0 source matrix separates values already frozen from new decisions.
`INHERITED_FROZEN` values are bound by a reviewed source/config/state manifest;
they must not be retuned or re-authorized one field at a time.  Binding a
different source remains a new authorization.

| Category | Current contents | Required action before an E5 run |
| --- | --- | --- |
| `INHERITED_FROZEN` | wavelength, pulse definition/amplitude, beam/focus, grid/domain, optical stepping, float64, gas/ionization/Raman switches, HR-2/HR-3 selection, hydro constants, `f_rep`, exact interpulse-construction rule, and hashed `PRE_0` source | Bind their existing hashes in the formal manifest; do not treat them as tuning choices. |
| `FORMAL_VALUE_TO_BE_SELECTED` | research question/control, formal endpoint, `Npulses`, formal full-z/source selection, and claim boundary | user scientific authorization. |
| `RESOURCE_POLICY_TO_BE_SELECTED` | 1+4 preferred topology or 1+2 fallback, allocation count, walltime/CPU/memory request, run roots, checkpoint/POST/NEXT retention and cleanup, and HR-4C/Streaming authority design | user execution authorization after site quota/limit confirmation. |
| `ACCEPTANCE_THRESHOLD_TO_BE_SELECTED` | restart/completion evidence, numerical-health thresholds, endurance criteria, diagnostic retention, and manual-review rule | user acceptance-contract authorization. |

The historical `Npulses=1`, 48 selected records, queue depth 16, block size 8,
and `[15000,351,301]` source shape are qualification/source facts, not formal
values to adopt automatically.  S5 PASS does not fill any selected category
and is never authorization to submit a formal E5 GPU job.

## Proposed task sequence

### E5-D0: bind the completed S5 evidence

After S5 reaches a terminal state, append the completed S5 receipt, controller
state, `sacct` terminal rows, exact-audit index, execution SHA(s), and the
F01--F06 cross-SHA inheritance table.  This is an evidence link only; it must
not copy or overwrite the S5, clean-reference, r2, or r3 run roots.

### E5-D1: freeze the formal science and resource contract

Before implementation or submission, create one reviewed input/claim manifest
from the authorization fields above.  It must distinguish the formal case from
any smaller engineering canary and state why every chosen reference is
scientifically comparable.  The manifest must bind canonical-LF Git-tracked
text hashes and raw-byte hashes for external/binary inputs through the existing
`provenance_v2.py` implementation.

### E5-D2: perform a read-only capability and evidence-gap audit

Audit the selected execution entry and current artifact schema against the
approved evidence plan.  In particular, the present runner persists the last
pulse's propagation diagnostics plus pulse-level summary arrays; it must not
be assumed to provide a per-pulse, per-z scientific record for a formal
accumulation claim.  The audit decides whether the existing persisted outputs
already satisfy the approved plan.  Any missing field becomes a separately
reviewed, minimal implementation request; this design does not prescribe a
replacement pipeline.

The 48-screen S3/S4R/S5 window validates the executor and lifecycle only in
its recorded scope.  It is a selected HR-3/Streaming/Hydro qualification
window while the S3 optical path follows the complete frozen longitudinal
schedule; it is neither a 48-step optical propagation nor a shortened z-domain.
It must not be extrapolated to an unlisted full-z or multi-pulse science
schedule, and its 3--4 hour runtime must not be linearly extrapolated to E5.

The E5-0 raw-payload accounting is also deliberately model-specific:
70.845 GiB is only the historical illustrative HR-4C authoritative+staging
model.  A separate Streaming CURRENT+POST+NEXT model is 106.267 GiB for that
same non-formal illustration, and simultaneous retention would be 177.111 GiB
before NPZ overhead, optical outputs, diagnostics, caches, and checkpoints.
A reviewed artifact-retention policy must create the actual formal full-z
budget; no number here selects `K=15000` or proves quota sufficiency.

### E5-D3: qualify the authorized formal payload

Once D1 and D2 are accepted, run the existing local checks for the actual
entry path: compileall, `run_local_tests.ps1 -Mode backend`, then the relevant
sanity/targeted tests.  If the entry, configuration loading, diagnostics, or
resume path changes, add the smallest non-overwriting check that reaches that
boundary.  A passing local check establishes software readiness only.

Before any Slurm side effect, perform the existing batch-entry audit and the
strict remote preflight using the exact execution SHA, inputs, output root,
interpreter, and resource request.  The preflight must precede creation of
execution locks or run directories and must fail closed on a source or
provenance mismatch.

### E5-D4: execute a separately authorized campaign

The execution plan must state whether cases run serially or in parallel and
provide an evidence-based queue/runtime estimate.  Each run receives a new
non-overwriting root and records the source/config/state manifests, topology,
Slurm receipt, and terminal `sacct` evidence.  Raw production data remain
remote unless a later authorization requests transfer.

No postprocessing, rerun, fallback topology, or expansion of the pulse count
is automatic after a failed gate.  A launcher/infrastructure failure before
scientific work is classified separately from a numerical or scientific
result.

### E5-D5: evaluate and close out

Postprocess only after terminal scheduler evidence and only through the
documented environment and existing analysis entry.  The closeout must
separate:

- engineering/lifecycle qualification;
- terminal execution and provenance completeness;
- numerical-health diagnostics (energy, finite fields, spatial boundaries,
  pulse-to-pulse state continuity, and resume counters);
- scientific comparison and any claim supported by the frozen evidence; and
- deferred limitations, including the E2 velocity-field caveat and HR-5
  performance backlog.

The closeout status is determined from the accepted contract and results; this
draft does not predeclare a PASS, a production parameter freeze, or closure of
all HR-4E work.

## S5 terminal supplement checklist

When S5 changes state, append only verified facts to this section:

- [ ] controller terminal state and timestamp;
- [ ] initial and recovery job IDs with terminal `sacct` rows;
- [ ] dual-claim/targeted-signal/quiescence evidence index;
- [ ] frozen inventory and `expected_recovery_effects` hashes;
- [ ] 48/48 completion and exact/provenance audit counts;
- [ ] F01--F06 inheritance table with each execution SHA and reason;
- [ ] S5 final status wording and any remaining limitation.

Until all items are verified, this remains a task design rather than an
execution authorization.
