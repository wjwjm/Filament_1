# Private combined CPU path

This module composes `INTENSITY_P24`, `EQ10_P48` and
`PHASE_ABSORB_P24` over one L1 factory. The qualification order is
`R2, X2, R8, X8`. Formal runs use `(384,351,301)` complex128, a fresh
1.5 GiB linear cache, and the existing frozen input/PRE and comparators.
Local engineering uses `(384,17,19)` with all worker counts set to one.

The original intensity/Eq10 stage is transformed once and both changes
are stripped for the inverse AST proof. C retains its separate G2
binding. Mathematical expressions, FFTs, reductions, precision and
operation order are unchanged. Pools remain separate and execute each
operation in sequence; no stage concurrency or thread search is added.

`entry` keeps the previous application clock and closes/joins all pools
before its normal output anchor. `hpc` reuses the previous supervisor,
reference/input sealing, physical topology checks and exact comparators.
The execution/input schemas are inherited from 6F, with a distinct task,
candidate, base SHA and four-window PLAN. Old modules are unchanged.

A single eight-interval pair is a descriptive performance observation,
not a new stability qualification. Historical default/production flags
remain unchanged; the user's choice of this combined path is recorded
separately from scientific and performance acceptance.

The bounded qualification entry is not a general long-distance or
multi-pulse production runner. Pure factory calls have no eight-interval
mathematical limit, but their source/config/capability and bounded timing
contracts still apply. A normal caller must own and finalize each factory;
it must not silently fall back on failure.
