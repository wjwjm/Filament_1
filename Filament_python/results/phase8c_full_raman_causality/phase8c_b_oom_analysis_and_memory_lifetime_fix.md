# Phase 8C-B: Raman-ON OOM analysis and memory-lifetime fix

Date: 2026-07-23

## Observed failure

The SHA-locked Raman-ON Test A job `180592` failed at `z ~= 0.537 m` in
`step_linear_bk_nee_factorized`, at the complex128 inverse temporal FFT:

```text
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 1,610,612,736 bytes
(allocated so far: 32,233,253,888 bytes)
```

The requested allocation is one full `512 x 512 x 384 complex128` field.  The
RTX 5090 reported `33,668,988,928` bytes of device memory.  The Raman-OFF
counterpart, job `180593`, completed all 15,000 records, so this is not a
physical instability or a failure of the frozen ON/OFF setup.

## Root cause

The full Eq. (27) Raman Strang path returns complete 3-D response fields and
2-D local closure maps for both Heun stages of both Raman substeps.  After the
per-step deposition and scalar accounting had already been calculated,
`propagate_one_pulse` kept those diagnostic payloads referenced through
`raman_diag_parts`, `pre_raman_diag`, `post_raman_diag`, and
`raman_step_diag` until the following BK-NEE halfstep.

At that point the mixed-precision inverse temporal FFT needs one additional
full complex128 volume.  This diagnostic retention, together with the
still-live mixed-precision input workspace, exhausted the GPU.  It is a
numerical memory-lifetime issue; it is not physical absorption and does not
change the Raman, Kerr, ionization, or plasma parameters.

## Implemented fix

1. After `Qacc_raman` consumes the local loss map, retain only the established
   scalar Eq. (27) closure/accounting fields and scalar raw-Raman diagnostics.
   Drop the no-longer-needed 3-D responses and 2-D local maps before the next
   linear halfstep.
2. Release the dead complex128 BK-NEE input cast immediately after its forward
   temporal FFT, and release the last per-slice spatial FFT buffers before the
   inverse temporal FFT.

The field update, full Eq. (27) Heun integrator, Strang ordering, two Raman
substeps/four convolutions per z-step, and `mixed_precision` BK-NEE strategy
are unchanged.

## Local verification

```text
python -m compileall Filament_python/KHz_filament
pytest -q Filament_python/tests/test_phase8b_raman_diagnostics.py \
          Filament_python/tests/test_phase8c_full_raman_causality.py \
          Filament_python/tests/test_sanity.py --maxfail=1
# 13 passed

pytest -q Filament_python/tests --maxfail=1
# 194 passed
```

The fix has not yet been exercised on an additional GPU job.  A replacement
full Raman-ON submission must remain separately authorized; this analysis did
not submit a smoke or a production rerun.
