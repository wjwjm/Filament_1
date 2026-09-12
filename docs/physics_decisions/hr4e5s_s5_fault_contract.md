# HR-4E-5S S5 restart and fault qualification contract

## Scope and frozen boundary

S5 qualifies only the durable lifecycle around the existing Streaming executor.
It does not alter the HR-2/HR-3/HR-4 scientific operator, deposition,
thermalization, grid, precision, `dt_hydro`, block size, or the frozen
`advance_hr4_single_screen` implementation.  The S5 base is
`d4bd96ca21c5815b57f1a3c2ee8f8cec54dfb220`; the source and input receipt is
[`baseline_receipt.json`](../../artifacts/hr4e5s_s5/baseline_receipt.json).

## Current qualification status and S4R handoff (non-normative)

This status note does not alter the frozen contract, its S3 science base, the
fault hooks, or the 1+1 configuration. S4R is
`CLOSED / REDUCED-SCOPE / SCIENTIFICALLY_QUALIFIED /
PERFORMANCE_OPTIMIZATION_DEFERRED`. Its missing formal Streaming 1+6 point is
`NOT TESTED / RESOURCE_UNAVAILABLE`, a deferred performance-matrix point rather
than an S5-1 prerequisite. S4R scientific qualification therefore no longer
blocks S5; performance optimization and any optional higher-resource topology
re-evaluation defer to HR-5.

S5-0 is `CLOSED / RESTART_FAULT_QUALIFICATION_CONTRACT_FROZEN`. The current S5
lifecycle branch/head is `96558e6e79a9c0adae63a62963e3a11d1189a084`; this is a
lifecycle baseline/head distinct from the frozen S3 science base above. Live
Slurm readback on 2026-09-12 records the 1 optical + 1 hydro clean reference
job `238355` as `FAILED / 1:0` after 40 s. Consequently S5-1 is
`OPEN / CLEAN_REFERENCE_TERMINAL_REVIEW_REQUIRED`, not PASS; no F01-F06 case is
submitted, replaced, or advanced by this documentation update.

Fault injection is test-only and disabled by default.  It is enabled only by
the three explicit environment variables recorded in
[`fault_contract.json`](../../artifacts/hr4e5s_s5/fault_contract.json).  It
never uses a timer, `sleep`, signal timing, or random process termination.

## Real lifecycle and authoritative state

The implementation's per-screen lifecycle is:

```text
CURRENT_READY -> OPTICAL_IN_PROGRESS -> DEPOSITION_FINALIZED
  -> POST_COMMITTED -> HYDRO_QUEUED -> HYDRO_RUNNING
  -> NEXT_COMMITTED -> BARRIER_VALIDATED
```

Generation-level completion is then `barrier.status == PASS`, followed by the
`authoritative_generation.json` promotion pointer.  `CURRENT` is authoritative
only through a validated manifest entry.  `POST` and `NEXT` become
authoritative only after their atomic NPZ rename, hash validation, and manifest
reference.  A PASS barrier is not a promotion; NEXT becomes the authoritative
generation only through a valid promotion pointer paired with the manifest
promotion record.

Files ending in `.tmp`, unreferenced artifacts, RAM queues, worker-local
intermediates, and incomplete manifest/pointer transactions are never
authoritative.  Restart must either safely discard a staged temporary artifact
or fail closed; it may never promote or hash-accept it.

## Recovery contract

Computations may run at least once when their corresponding authoritative
commit did not occur: optical work before POST and hydro work before NEXT.  By
contrast, the state transitions POST commit, NEXT commit, and promotion are
exactly once.  A recovery first validates durable inputs, reconstructs queue
work from POST entries without NEXT entries, then executes barrier and
promotion only when their own preconditions hold.

The six S5 faults map to the executor's real boundaries.  F01 stops after
deposition finalization before POST staging; F02 after the POST manifest save;
F03 after the durable hydro claim; F04 after the fsynced NEXT temporary write
and before rename; F05 after NEXT's manifest save; and F06 after the durable
PASS barrier before promotion.  Their state expectations and exact comparison
objects are normative in `fault_contract.json`.

Two additional real transaction boundaries are qualified locally: F07 after
the NEXT rename but before its manifest reference, and F08 after the promotion
pointer write but before the manifest save.  They close write-order gaps that
the six mandatory boundaries cannot otherwise observe.

F09 similarly qualifies the POST rename-before-manifest gap.  Restart removes
only a temporary JSON sibling or the precisely expected missing POST/NEXT
artifact for its durable predecessor state; all other artifact inventory
anomalies remain fail-closed.

## S5-1 acceptance

Each case starts in a clean independent case root, produces a deterministic
non-zero injected exit, and records its durable state immediately.  Recovery
uses the ordinary reconstruction path with no manual scientific-state edits.
It must match the one 1-optical + 1-hydro clean reference exactly for POST,
NEXT, final optical field, deposition ledger, completion map, normalized
manifest, ownership, barrier, and promotion generation.  Only timestamp,
Slurm-job, PID, and explicit fault provenance metadata may differ.

Any duplicate authoritative commit, lost screen, staged artifact accepted as
authoritative, early/multiple promotion, mixed ownership, scientific mismatch,
or fault-off regression is a hard failure and leaves S5-1 open.
