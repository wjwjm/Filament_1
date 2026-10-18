# N3 journal metadata-drift repair

Job 274998 failed in the R optical process at `deposition_finalized`, when
the cached journal stat changed without a HEAD advance. Both retained
journals replayed successfully. A bounded synthetic JuiceFS fixture reproduced
the defect by changing only the timestamp: the old implementation rejected
unchanged content. This establishes the faulty decision rule; it does not
prove what changed the timestamps during the historical job.

## Implementation

The journal authority now retains a process-local incremental SHA256 of the
verified raw prefix. Initialize, initial replay, validated delta replay and
successful commit update that state without changing checkpoint/journal/HEAD
formats. Ordinary delta refresh hashes only the appended bytes.

Metadata drift invokes a bounded streaming verification under the existing
caller lock. The committed bytes must match the cached/candidate digest and
HEAD, inode and size must remain consistent. Comparable path-stat samples
include timestamps; descriptor-stat samples compare identity and size because
Windows exposes different descriptor/path timestamp behavior. An unstable
snapshot gets at most one additional pass. Stable corruption, replacement,
uncommitted/truncated tails and failed reads remain errors, never repairs.

Load and delta replay use the same exceptional verification if their path
metadata changes while reading. Cached digest state is published only after
successful validation. Existing lifecycle poisoning prevents reuse of a
partially replayed failed view.

Exceptions carry cached/observed stat, HEAD and digest context. Successful
slow checks emit `JOURNAL_METADATA_RECHECK` with PID, sequence, committed
offset, reason, count and elapsed totals. Ordinary transactions emit no new
marker. Monitoring may use this marker without changing durable protocols.

## Validation and release boundary

Local journal regressions cover timestamp-only changes, same-length content
corruption, old-prefix corruption, replay-window mutation, bounded unstable
snapshots, fresh replay, digest synchronization and slow-path logging. The
repository backend, sanity, compile and affected N3/timeout checks are required.

The isolated site fixture uses no GPU or scientific solver. It checks the
old/new timestamp behavior and content-corruption rejection, and measures
raw streaming scans of 211,863,864 bytes (largest normal N3 root) and
278,972,728 bytes (that root plus one maximum terminal transaction). These
are synthetic scanner measurements, not full journal-performance qualification.

The old performance gate binds the old hot-path hash. This source change
invalidates that reuse. The old gate must continue to reject the repaired
module; neither replacing its hash nor relabeling the microbenchmark as PASS
is acceptable. Formal resubmission remains blocked until qualifying site
performance evidence and its explicit new-source admission binding exist.
The historical full site suite took about 549 seconds and used an allocation;
it has not been rerun as a long login-node task or as an unauthorized allocation.

## Failed-run storage and monitoring

The user's later cleanup authorization supersedes retention of all failed
arrays. Cleanup is restricted to the allowlisted generated NPY/NPZ files in
job 274998's R/C pulse-0 optical/state directories, after an independently
downloaded and hash-verified forensic archive. Preserve journal, HEAD,
checkpoint, ledger, logs, cache, frozen scientific inputs and sources. The old
root is forensic-only after cleanup and must never be resumed.

A future attempt starts at pulse 0 under a new root and identity, with the
unchanged N3/K8048/H4, 6-GPU/48-CPU, 120-hour and exclusion constraints.
No new job is created by this repair. Automation `n3` is configured but paused
until a verified new job ID exists: queued 30 minutes; first 2 running hours
10 minutes; then 30 minutes; after 2 stable hours and progress beyond the old
failure point, 2 hours. Suspected stalls restore 10-minute observation without
automatic cancellation, recovery or submission. E5_1C and E5_2 remain OPEN.
