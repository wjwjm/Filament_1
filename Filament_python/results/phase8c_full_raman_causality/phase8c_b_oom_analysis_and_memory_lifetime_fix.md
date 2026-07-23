# Phase 8C-B: Raman-ON OOM analysis and memory-lifetime fix

Date: 2026-07-23

## Observed failure

SHA-locked Raman-ON Test A job `180592` failed at `z ~= 0.537 m` in
`step_linear_bk_nee_factorized`, at the complex128 inverse temporal FFT:

```text
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 1,610,612,736 bytes
(allocated so far: 32,233,253,888 bytes)
```

The requested allocation is one full `512 x 512 x 384 complex128` field. The
RTX 5090 reported `33,668,988,928` bytes of device memory. Raman-OFF job
`180593` completed all 15,000 records; this is not a physical instability.

## Root cause and fix

Full Eq. (27) Raman Strang diagnostics retained 3-D response fields and 2-D
local closure maps after their accounting was complete. The following
mixed-precision BK-NEE halfstep then lacked room for its additional complex128
inverse temporal FFT.

The fix retains the established scalar/2-D accounting, releases those completed
diagnostic payloads before the next linear halfstep, and releases dead
mixed-precision work buffers promptly. It does not alter the Eq. (27) field
update, Heun integrator, Strang order, convolution count, physical parameters,
or `mixed_precision` strategy.

## Local verification

- `python -m compileall Filament_python/KHz_filament`
- `pytest -q Filament_python/tests --maxfail=1` — `194 passed`

No replacement GPU run was submitted by this analysis alone.
