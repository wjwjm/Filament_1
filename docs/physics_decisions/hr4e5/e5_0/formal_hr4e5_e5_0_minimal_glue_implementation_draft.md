# Future task draft: formal HR-4E-5 minimal orchestration glue

**Status:** `DRAFT_ONLY / NOT_AUTHORIZED_TO_IMPLEMENT`

This is the smallest proposed implementation boundary after S5 PASS and a
separate user authorization. It does not authorize code changes, configs,
source materialization, run-root creation, or any Slurm action.

## Objective

Provide one formal entry that uses the existing HR4D/HR4C canonical authority
to run fresh optical pulses, produce canonical POST, execute an optional
non-authoritative streaming hydro adapter, and promote canonical next PRE. It
must preserve every frozen scientific operator and precision setting.

## Expected file scope

| File / area | Classification | Expected change |
| --- | --- | --- |
| new `Filament_python/KHz_filament/hr4e5_formal_entry.py` | `RUNTIME_ENTRY_ONLY` | construct/reopen `HR4DPulseController`; construct the post-lens immutable source; call real `propagate_one_pulse` with `HR4DPulseTransaction`; sequence PRE/POST/interpulse/POST_final. |
| new `Filament_python/KHz_filament/hr4e5_hr4c_streaming_adapter.py` | `ORCHESTRATION_ONLY` | make interval work descriptors from HR4C POST, record non-authoritative NEXT staging and barrier receipts, then hand validated results to HR4C staging. |
| future manifest/receipt schema | `PROVENANCE_ONLY` | bind PRE_0 three-field hashes, source/config/SHA, z identity, pulse/generation/phase, barrier, replay and retention facts. |
| focused tests | `TEST_ONLY` | add tiny CPU tests for N=3, fresh source, exact serial-vs-adapter path, restart boundaries, no duplicate/omitted interval, no pointer, and POST_final. |

The existing modules below are reused unchanged: `propagate_one_pulse`,
HR-2 deposition, HR-3 mapping, `advance_hr4_single_screen`,
`HR4CThreeFieldStore`, `HR4DPulseController`, ionization, Raman, frozen
configs, LUTs, and source arrays.

## Required control flow

1. Validate immutable config/source/PRE_0 provenance before creating a run
   root, lock, receipt, or scheduler action.
2. Create/reopen exactly one HR4D controller. Its manifest-selected HR4C
   state is the only writable canonical state.
3. For each PRE pulse, make a new `E_source.copy()` and pass the HR4D
   transaction as the HR-3 slow-state interface.
4. Finalize only a complete all-interval POST transaction. Record POST receipt.
5. Unless final, adapter workers read canonical POST, create only
   non-authoritative NEXT staging, and emit a barrier receipt.
6. The coordinator verifies the barrier, fills HR4C staging, and commits one
   PRE-next generation. It then advances the one pulse index.
7. On restart, reopen HR4C first; rebuild/reuse adapter staging only against
   the recovered canonical phase and generation. Never use a Streaming pointer
   as formal authority.

## Namespace and retention rules

- Canonical state has one HR4C root and two fixed slots; its generation is in
  the HR4C manifest.
- Adapter staging is namespaced by source POST generation and pulse index.
- Adapter NEXT staging is immutable after validation and non-writable after
  the HR4C next-PRE commit.
- Default canary retention is receipts, hashes and selected diagnostics, not
  prior full-volume state copies. Any checkpoint retention is explicit policy.
- `authoritative_generation.json` is prohibited in a formal HR4C-authoritative
  root.

## Mandatory pre-HPC validation

- required local compile/backend/sanity gates through `run_local_tests.ps1`;
- dedicated tiny N=3 exact serial/reference versus adapter comparison;
- PRE_0 three-field/hash and z-identity audit;
- all declared restart-boundary tests, including staging abort/rebuild;
- formal output/receipt and no-pointer audit;
- existing batch-entry audit and strict remote provenance preflight before any
  run directory, lock, receipt, or `sbatch` side effect.

## Explicit exclusions

No `SCIENTIFIC_OPERATOR_CHANGE` is allowed. A need to alter propagation,
ionization, Raman, HR-2, HR-3, HR-4, precision, source construction physics,
LUTs, or frozen configs is a blocker requiring a new scientific decision.
