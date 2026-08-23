# Hybrid Propagation 0.60 m validation — job 222014

## Outcome

Mechanical classification: `hybrid_0p60_not_supported`.

This classification is caused solely by an acquired code-execution gate
failure. It is not evidence that the 0.60 m hybrid propagation changes the
filament physics, and it is not a G1/G2/G3 or performance failure: neither the
strict reference nor the hybrid candidate started.

## Provenance

- Baseline SHA: `32703b7080bea0b201ebfd57336a59451183150f`.
- Execution SHA: `fc1a9cb57e2d6ff22c4c7e9a66d114594a6f53b3`.
- Branch: `codex/hybrid-propagation-validation`.
- Job: `222014` on `m4gn1401`, partition `gpu`, 1 GPU, 8 CPU, requested 15 h.
- Run directory: `/data/run01/scvi806/user_Wangjimin/hybrid_propagation_validation_0p60/run_fc1a9cb_20260823T135931Z`.
- Execution lock SHA256: `8f42cf6ea80a283b10efdd986192e9411d7f0fae0272ab475feac86b7461077f`.
- Provenance v2 SHA256: `6a38261482dec02973209d0b01b86e88f5e65baeccbb882af29e650567b27bf6`.

The final repository SHA is the commit containing this report and the
corrective batch-script change; its exact value is reported in the delivery
message after push.

## Validation completed before submission

- Mother config SHA remained
  `942adca964f50b689fa5985c9af46f294da7948646b246c39ca0d50238a1b02a`.
- Reference/hybrid configs differed only in
  `propagation.propagation_mode` and `propagation.z_nl_start` (`0.60 m`).
- H0 bitwise reference, H1 boundary splitting, H2 pure-linear fp64/fp32,
  compileall, sanity, shell syntax, JSON checks and local tests passed.
- Last full local test result before submission: `287 passed, 3 skipped`.
- Final read-only HPC preflight passed account/root, clean repository,
  Git/Slurm tools, fixed conda environment and strict remote provenance.

## Scheduler terminal evidence

`sacct` reported `FAILED`, exit `127:0`, elapsed `00:00:02`, from
`2026-08-23T22:00:15` to `2026-08-23T22:00:17` in the scheduler's displayed
time basis. The allocation was on `m4gn1401` with 1 GPU and 8 CPUs.

The only stderr line was:

```text
/var/spool/slurmd/job222014/slurm_script: line 66: python: command not found
```

No LUT warm-up, reference propagation, hybrid propagation, raw NPZ, paired
metadata, curves, visual review, G1/G2/G3 evaluation, numerical-health result,
GPU peak-memory result or speedup measurement was produced.

## Corrective action and retry boundary

The batch script now invokes the fixed interpreter
`/data/home/scvi806/.conda/envs/Filament_python/bin/python` for provenance
validation before conda activation. This correction was locally validated,
but no second Slurm job was submitted. Any new production pair requires a new
explicit authorization and a newly bound execution SHA, lock and provenance.
