# CPU S4-6F original-size parallel candidates

User-authorized independent original-size validation of EQ10_P48,
INTENSITY_P24 (magnitude released after the square barrier), and
PHASE_ABSORB_P24 against REF_L1, in that order.

HPC scientific/performance shape is [384,351,301], complex128. Microgrid
[384,17,19] is local engineering only. Historical small-grid performance
is not an admission gate for this new experiment. Original records remain
immutable and are never pooled with new performance samples.

Each full propagation candidate uses its own fixed fourteen-window serial
allocation, 96 physical CPUs, 128 GiB, 2 hours, no GPU. Phase/absorption
first requires full-size real capture and REF/P1/P24 mechanism evidence
(36 warmups, 90 formal calls), up to one hour on the same resource shape.
Normal path: at most four jobs / 672 reserved allocated core-hours;
maximum one inflight. No automatic requeue or performance rescue samples.

The main metric remains sum(body[1:]) over intervals 2 through 8.
Pool creation, management, join and normal output remain inside the
original application clock. Two-interval stage/boundary qualification;
every eight-interval window retains complete terminal exact comparison.

DEFAULT=I0, PRODUCTION_QUALIFIED=false, MANUAL_REVIEW_PENDING.
Read STATE.json and MONITOR.md for execution status; ignored evidence
and scientific arrays are never embedded into Git by broad staging.
