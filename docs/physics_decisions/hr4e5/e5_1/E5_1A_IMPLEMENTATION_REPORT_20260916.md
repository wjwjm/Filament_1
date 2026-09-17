# E5-1A implementation and local qualification

Issued 2026-09-16; implementation and validation 2026-09-17.
Base: `131daa541ac37e54d43b29cd3bfc12ba83c3b91d`.
Branch: `codex/hr4e5-e5-1a-streaming-implementation`.

**Disposition: E5_1A_PARTIAL / BLOCKED_BY_FORMAL_DRIVER_AND_RECOVERY_QUALIFICATION.**
The new code, bounded CPU evidence and review archive are delivered for review.
This is not A acceptance or authorization to start B/C. E5-0 remains
`CLOSED / STREAMING_PREPARATION_CONTRACT_ACCEPTED`.

## Implemented boundary

Four new modules provide a pinned Streaming CURRENT read view and POST hook,
real original optical-call adapter, prefix/PRE0 helpers, independent successor
copies, final POST-only validation/replay, durable pair progress, exact evidence,
and one campaign storage ledger with reservations and per-file reclamation.
`hr4e5s_streaming.py` required no narrow extension and remains unchanged.
Scientific propagation, nonlinear/ionization/Raman/hydro operators, frozen
configs/LUTs and E5-0 evidence remain unchanged. The existing local test wrapper
adds only the new lightweight test file.

The actual three-pair test uses separate R and C roots. R uses original
`HR4CThreeFieldStore`/`evolve_hr4_full_z`; C uses original Streaming claims,
queue16/block8, worker, barrier and promotion. Both tracks share scientific
hydro operators but never each other's saved answers. Optical in this integrated
test is explicitly a deterministic test double. Separate tiny CPU tests call
original field construction, lens path and `propagate_one_pulse` with actual
state/sink/hook injection, including interrupted final optical replay. Those
tests do not establish a three-pulse real-physics equivalence result. Focused review fixes additionally reject disabled HR-3B, mismatched schedule/grid and conflicting direct POST replay before new optical writes.

## Qualification and remaining blockers

Final executed checks: **30 new tests passed** (71.50 s), **34 original
Streaming/HR4D tests passed** (18.91 s), sanity **1 passed**, backend and explicit
compileall passed. No tests were marked skipped; deselected tests were outside
the bounded selected suites. The 30 tests do not mean all 18 acceptance contracts
are fully qualified: partial rows and concrete limitations remain in the JSON.
The authoritative command results and A01-A18 disposition are in
[test results](E5_1A_TEST_RESULTS.json). The
[change matrix](E5_1A_CHANGE_AND_QUALIFICATION_MATRIX.md) distinguishes inherited
interfaces from new evidence. Passing individual tests is not full A acceptance.

1. The concrete end-to-end independent R/C driver is currently a test fixture.
   `PairedCampaign` is callback orchestration and accepts callback-reported PASS;
   it does not itself validate a complete formal scientific object inventory or
   bind every recovered step to the referenced durable report. A production
   invocation cannot be qualified merely by supplying PASS callbacks.
2. The real optical adapter is tested at tiny dimensions but formal config,
   source, LUT, full schedule/grid identity binding and admission of PRE0 and
   successor creation are not yet an enforced single outer entry. Public create
   helpers can be called without the campaign reservation. This is a code-level
   A gap, distinct from unknown site quota.
3. Full transition interruption qualification is incomplete: pointer/index
   windows, stale epoch/fork decisions, and interrupted pair handoff plus GC
   resumption are not all covered by a single restartable formal coordinator.
   The successful fresh-process test stops only at the fully completed first
   pair boundary. Lower-level replay/GC tests have narrower guarantees.
4. File registration records verified identities and hashes, but it is not a
   universal proof that an arbitrary caller created a pre-existing file.
   Fixture writers register only their newly created files. General formal
   creation/ownership binding remains part of the missing outer entry.

5. Terminal output-directory ownership, full sink/orphan inventory and persisted
   exact-row-to-array provenance remain incomplete for formal reuse. Current
   child payloads and retained terminal artifacts are re-read, but this does not
   replace full formal source/config/LUT/schedule and report binding.

These concrete gaps prohibit `E5_1A_IMPLEMENTATION_READY_FOR_REVIEW`. No scientific
operator change is proposed to bypass them. B remains closed until A gaps are
fixed and independently reviewed; site qualification is an additional later gate.

## Storage model tied to current fixture layout

[Resource CSV](E5_1A_RESOURCE_BUDGET.csv) gives each phase in exact bytes;
[retention contract](E5_1A_STORAGE_RETENTION_CONTRACT.md) defines the gates.
The single campaign hard cap is **322122547200 bytes (300 GiB)**, including both
tracks, copies, attempts/residue, evidence, final outputs and positive headroom.
Final-output cap is **68719476736 bytes (64 GiB)** and is included in the total.

For K8048, Ny351, Nx301, Nt384, G=20406701952 bytes for three slow fields and
O=649119744 bytes per optical array. R nonfinal has PRE/POST/NEXT snapshots 3G,
Batch slots 2G and six sinks 2G: 7G. C has CURRENT/POST/NEXT 3G and sinks 2G: 5G.
Pair exact therefore retains 12G. Sequential R successor creation peaks at 13G
before reclaiming R; only then is the C successor copied. The conservative
failed-handoff stage adds another G of uncredited residue, four retained optical
outputs, one source-copy O, 2 GiB metadata and 8 GiB safety margin:
**299676844288 bytes**, below the hard cap by 22445702912 bytes.
Final default retention is 2G candidate CURRENT+POST, 6O outputs, one source-copy O
and 2 GiB metadata: **47504725760 bytes**. Extra diagnostics/checkpoints are
explicitly selected and rejected when they exceed the final cap.

This is a scalar model aligned to the implemented fixture's file lifetimes,
not a measured formal scientific campaign. Formal driver alignment is blocked
by the missing entry described above. No K8048 scientific arrays were created.
The model reserves failure residue and stops; it does not allow unbounded retries.
N-dependent final optics/reports grow with N even though live slow-state history
is reclaimed within a bounded window.

Host RAM is separate: successor construction uses lists plus `np.stack`, with
about 2G host-array payload transient (40813403904 bytes) before additional
library overhead. The PRE view caches up to G; the reference fixture loads full
volumes and is not an RSS-qualified formal runner. Disk figures are logical
payload plus allowances; allocated filesystem blocks, memmap address space,
RSS and VRAM are not inferred from them and have not been measured at K8048.

## Recovery, retention and limits

The integrated fixture actually compares 1344 screen arrays (42x32), 27 ledger
arrays and three final optical arrays before reclamation. First-pair parent
arrays are removed; the old scientific process exits, a distinct OS process and
epoch continues the remaining two pairs from durable roots, and its retained
results are compared to an independent uninterrupted run. This supports only
`NEW_PROCESS_PERSISTENT_RESUME` at that completed-pair boundary.

Candidate final CURRENT+POST remains reopenable. Historical parent/sink arrays
were compared before deletion and reclaimed under the fixture contract; the
retained hashes do not allow redoing all historical elementwise comparisons.
No cross-allocation, cross-node, host reboot or power-loss guarantee is claimed.
The few-write K8048 manifest prototype measures local serialization only; it is
not K8048 lifecycle execution, a K-squared stress test or an HPC I/O estimate.

No Slurm, GPU, S5 monitoring/worktree action, formal input materialization, B/C,
merge or force push occurred. B additionally requires manual A acceptance,
S5 terminal/patch compatibility, materialized prefix hashes, site quota/free
space/QoS/retention, actual RAM/VRAM/I/O feasibility and separate authorization.
Runtime orchestration changed=yes; scientific operators changed=no;
Slurm-GPU-S5 actions=no.
