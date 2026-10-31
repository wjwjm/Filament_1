# CPU-S4-6E: independent A/P24 and C/P24 propagation

Approved on 2026-11-01. Baseline: 4431ba1192ee0d14b0eddf2efb138fe2a549d49f.
Science tree: 821fec5bdacd832b0347e1f2e84285cb54172e57.

Order: A small-grid, conditional A full-size; C real-input mechanism,
conditional C small-grid and full-size. Each candidate is compared separately
with L1. No B/P48, P_LOCAL, 5C/5D or candidate combination is included.

Read STATE.json and MONITOR.md for current execution, not this static entry.
Per-candidate/grid performance uses the fixed 14-window protocol, with exact
qualification, body ordinals 2..8, balanced order/block and RR checks, POST
and full-application guards. A valid negative result closes that candidate.
Raw arrays and failed evidence stay at the execution site. Review packages
contain source identities, comparison metadata and replayable scalar clocks.

DEFAULT=I0; PRODUCTION_QUALIFIED=false; MANUAL_REVIEW_PENDING.
Multiple CPU submissions are explicitly authorized by the user. The normal
path has at most five conditional jobs and at most 528 allocated core-hours;
all run serially on t0, with no automatic requeue or production promotion.
