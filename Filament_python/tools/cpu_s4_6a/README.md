# CPU-S4-6A matched optical measurement

This private namespace binds accepted CPU L1 and the unchanged GPU optical
core to one canonical raw input bundle per grid. It changes acquisition and
observation only. The scientific tree remains `821fec5bdacd832b0347e1f2e84285cb54172e57`.

The fixed matrix has an auxiliary current-environment reference, then
`QUAL_OFF`, `QUAL_LIGHT`, `P1_OFF`, `P1_LIGHT`, `P2_LIGHT`, `P2_OFF`,
`P3_OFF`, `P3_LIGHT`. GPU also has one explicitly serialized phase diagnostic,
whose clocks are separate from the OFF comparison. Each process starts from
fresh source/PRE and a new factory; propagation warm-up is zero.

Use the task's frozen `MEASUREMENT_PLAN.json`, source manifest, budget
admission and `INPUT_BINDINGS.json`. This is not a general campaign launcher.
CPU entry resources are 96 physical cores, temporal FFT48, spatial FFT24,
outer mask24, Raman48 and BLAS1. GPU remains one device/eight host CPUs.

`entry` never submits. `matrix` runs only inside an admitted allocation;
its sbatch files take paths and budget identities from the audited submitter.
Exact comparisons reuse the original current-backend comparators, including
the original NaN/timing exceptions. Cross-backend differences are descriptive
and confer no numerical-equivalence or production qualification.

Local checks go through `tools/run_local_tests.ps1`; the report's fixed target
adapters select these tests and the independent Linux interpreter. GPU manager
simulation and AST compilation are labelled `GPU_REAL_NOT_RUN`.

`DEFAULT=I0`, `PRODUCTION_QUALIFIED=false`, `MANUAL_REVIEW_PENDING`.
