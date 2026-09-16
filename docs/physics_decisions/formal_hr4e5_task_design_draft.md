# Formal HR-4E-5 task-design draft

## Status and purpose

**Status (2026-09-16): `DESIGN_DRAFT / ENTRY_BLOCKED_BY_S5_FINAL`.**  This
document starts the design work for the formal HR-4E-5 campaign while
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

The formal question is to measure, under one predeclared physical condition,
how the frozen pulse-to-pulse slow-medium state changes the propagation and
slow-state observables over a specified pulse train.  It must use the frozen
HR-2/HR-3/HR-4 scientific contract:

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

## Decisions that require explicit scientific authorization

The following fields are intentionally unfilled.  Filling any of them changes
the scientific campaign rather than merely completing engineering preparation.

| Decision | Required record before an E5 run |
| --- | --- |
| Research comparison | Question, control/reference definition, and the claim that the comparison may support |
| Pulse schedule | `Npulses`, `f_rep`, exact interpulse schedule, and whether the run is a pilot or the formal endpoint |
| Physical input | Immutable source/config/state manifests, input-plane definition, dtype, grid/domain, `dt_hydro`, and all HR-2/HR-3/HR-4 switches |
| Initial condition | Authoritative `PRE_0` source, hash, and its relationship to the selected physical condition |
| Execution scope | Full-z/source-screen selection, topology selected from the already qualified choices, job count, queue/walltime/memory request, and non-overwriting run roots |
| Evidence plan | Per-pulse observables, state checkpoints, field-retention policy, comparison metrics, numerical-health thresholds, and manual-review requirement |
| Acceptance language | What qualifies an engineering pass, a numerical diagnostic, and a limited scientific conclusion; unresolved caveats must remain named |

S5 PASS does not fill these fields and is never authorization to submit a
formal E5 GPU job.

## Proposed task sequence

### E5-D0: bind the completed S5 evidence

After S5 reaches a terminal state, append the completed S5 receipt, controller
state, `sacct` terminal rows, exact-audit index, execution SHA(s), and the
F01--F06 cross-SHA inheritance table.  This is an evidence link only; it must
not copy or overwrite the S5, clean-reference, r2, or r3 run roots.

### E5-D1: freeze the formal science contract

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
its recorded scope.  It must not be extrapolated to an unlisted full-z or
multi-pulse science schedule without a specifically authorized equivalence
check at the formal E5 inputs.

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
