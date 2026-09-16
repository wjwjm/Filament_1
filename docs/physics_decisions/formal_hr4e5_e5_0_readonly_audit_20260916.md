# Formal HR-4E-5 E5-0 read-only capability and resource audit

**Status:** `E5_0_READY_FOR_MANUAL_REVIEW`
**Scope:** source/config/runtime inspection and documentation only; no physics,
runtime, LUT, production-input, historical-result, or Slurm-job modification.

## 1. Executive decision

E5-0 has produced an auditable design packet, but it has **not** frozen a
formal execution contract.  The primary entry gate remains
`ENTRY_BLOCKED_BY_S5_FINAL`: at the read-only snapshot on 2026-09-16,
S5-FINAL job `244700` is `PENDING (Priority)`, has not been allocated, and
there is no recovery job.

Two design findings govern the next review:

1. **Multi-pulse entry verdict: `EXISTING_ENTRY_PARTIAL`.**  The repository
   has a tested HR-4D PRE/POST/interpulse controller and it has a runner that
   makes a fresh optical-field copy per pulse, but it has no single production
   entry that binds the real full-z optical/HR-3 POST path, the HR-4D
   controller, and Streaming CURRENT/POST/NEXT across formal pulses.
2. **48-screen verdict: `NOT_A_LEGAL_MULTIPULSE_CASE`.**  S3/S4R/S5 exercise
   a one-pulse, 48-record qualification window while the optical path still
   propagates the complete frozen longitudinal schedule.  They are neither a
   48-step optical propagation nor a legal small multi-pulse E5-1 input.

The observed full-z payload model is large enough to require an explicit
retention policy before any run.  For the illustrative historical geometry
`K=15000`, `Ny=351`, `Nx=301`, float64, the raw data-payload lower bounds are
35.422 GiB for one three-field generation, 70.845 GiB for HR-4C's two slots,
106.267 GiB for Streaming CURRENT+POST+NEXT, and 177.111 GiB if both stores
coexist.  These exclude NPZ/container overhead, temporary files, diagnostics,
optical fields, LUT/cache, logs, and unselected checkpoints; they are not an
authorization to use `K=15000` in formal E5.

## 2. Fixed evidence identity and boundary

| Item | Value / scope |
| --- | --- |
| Repository / review branch | `wjwjm/Filament_1`, `codex/hr4e5s-s5-lifecycle` |
| Review evidence SHA at E5-0 start | `2613778adbb19428ac027145a840f91dd04a7515` |
| Actual queued S5 runtime SHA | `cd456ff8413cbc041d2d60b9b64007a1554028a1` |
| S5 read-only scheduler snapshot | `244700`, `PENDING (Priority)`, `gpu`, QoS `normal`, account `n50r5`, request `12:00:00` |
| Frozen qualification config | `real_post_input_config.json`, raw SHA-256 `eaec83ad326a29af95881912db76a7e5c26943dec9e0d6876ca9c9cc2a100c22` |
| Frozen HR-3B state / array | raw SHA-256 `70677c...81f467`; array SHA-256 `5990da...188d8ad9` |
| S3 source manifest / prepared input manifest | raw SHA-256 `3e2c7557...62adaf32d2` / `280d8d1f...b6b5f5a8a` |

The review SHA names the source tree under audit.  It is deliberately not
substituted for the S5 runtime SHA.  The queued job uses a clean remote
worktree at the runtime SHA; the present documentation is not an input to
that job.  Main evidence sources are the two existing decision documents,
the S5 prepared input and provenance copied under
`Filament_python/results/hr4e5s_s5_1r_238355_failure_audit/`, and the source
modules cited below.

## 3. Real multi-pulse call graph and authority

### 3.1 Implemented HR-4D lifecycle

```text
HR4DPulseController.store authoritative PRE_p (six-slot HR4C memmap)
  -> run_one_pulse_transition: source_template.copy() = fresh E_p
  -> caller-supplied pulse_runner(fresh E_p, HR4DPulseTransaction)
  -> transaction writes all three POST fields to HR4C staging
  -> transaction.finalize(): atomic POST_p authority / generation advance
  -> build_interpulse_step_schedule(f_rep, dt_hydro)
  -> evolve_hr4_full_z(): HR4C staging -> atomic PRE_(p+1) authority
  -> repeat with a new source_template.copy()
  -> POST_final authoritative; run_complete=true; no final interpulse pass
```

`hr4d_pulse_lifecycle.py` owns this contract:

| Link | Code evidence | Input -> output / authority |
| --- | --- | --- |
| Fresh field | `run_one_pulse_transition` | `source_template.copy()` produces one working optical field before the transaction opens. |
| PRE -> POST | `HR4DPulseController.begin_pulse_transition`, `HR4DPulseTransaction.finalize`, `_commit_post_transition` | all three fields are staged then atomically become POST generation. |
| Interpulse duration | `run_interpulse_transition` | `build_interpulse_step_schedule(f_rep, dt_hydro)` supplies exactly the decomposed `1/f_rep` schedule. |
| POST -> next PRE | `evolve_hr4_full_z` and HR4C `commit_staging` | six persistent slots swap authoritative/staging names atomically in the manifest. |
| Restart | `HR4DPulseController.open` plus HR4C manifest | the authoritative generation, phase, pulse index, and transaction state are recovered from the HR4C manifest. |
| Completion | `_metadata_for`, `run_hr4_pulse_train` | only `POST` at `pulse_index == n_pulses - 1` is complete. |

The general runner (`runner.py`) independently implements fresh copies as
`E_source = E` then `E_pulse = E_source.copy()`.  Its integrated multi-pulse
path is tied to `HR3CStateController` (`begin_pulse`, `commit_post_pulse`,
`diffuse_to_next_pre`), not to `HR4DPulseController`.  It breaks immediately
after its final pulse and therefore does not add a final diffusion pass.

### 3.2 Streaming qualification lifecycle

```text
S3 read-only CURRENT source screen
  -> selected optical/HR-3 hook commits disk POST_<ordinal>
  -> queued advance_hr4_single_screen writes disk NEXT_<ordinal>
  -> all-record barrier validates hashes / identities / queue emptiness
  -> authoritative_generation.json points to NEXT
```

`StreamingLifecycle.create` copies each supplied CURRENT screen into one
`current/screen_*.npz`; it does not copy a full state into coordinator RAM.
The `_SelectedStreamingHook` in `hr4e5s_s3.py` is called only for selected
ordinals after HR-3 authoritative output.  `commit_post` and `commit_next`
write one float64 three-field NPZ atomically, and `validate_barrier` verifies
all records before `promote_next_to_current` writes the authoritative pointer.
Restart is handled by `StreamingLifecycle.open` and `reconstruct_queue` from
the manifest plus authoritative artifacts.  Atomic temporary NPZ files may be
discarded only when the strict restart predicates prove that they are
non-authoritative.

Promotion does **not** rename NEXT into the `current/` directory or delete
CURRENT/POST.  After a successful promotion, `current_fields` reads NEXT via
the pointer and the old CURRENT, POST, and NEXT files remain represented in
the manifest.  No source evidence demonstrates a new-pulse namespace reset,
rollover, or cleanup policy.

### 3.3 Capability finding and minimal missing glue

**Verdict: `EXISTING_ENTRY_PARTIAL`.**  `rg` confirms that
`HR4DPulseController` / `run_hr4_pulse_train` are used by their module and
tests, not by the production runner or S3/S4R/S5 runtime.  Conversely,
`hr4e5s_s3.run_optical_path` uses a single complete optical trajectory and
the S3 streaming lifecycle, not HR-4D pulse orchestration.

The smallest identified missing work is orchestration glue, subject to a
separate review:

- inject authoritative full-z PRE state into each fresh optical pulse;
- bind real propagation/HR-3 POST production to exactly one HR-4D/Streaming
  authority model rather than duplicating ownership;
- define generation, namespace, output, and provenance rollover for each
  subsequent pulse; and
- bind restart receipts and pulse-history metadata to that entry.

This audit does not implement any of that glue and does not establish that
both persistent store models may safely coexist.

### 3.4 Final-pulse semantics (illustrative counts only)

| Illustrative `N` | Fresh optical calls | POST commits | Interpulse evolutions | Formal terminal state |
| ---: | ---: | ---: | ---: | --- |
| 3 | 3 | 3 | 2 | `POST_final` |
| 5 | 5 | 5 | 4 | `POST_final` |

The values 3 and 5 explain the implementation only.  They are not proposed
formal `Npulses` values.

## 4. 48-screen qualification scope

`prepare_input_manifest` in `hr4e5s_s3.py` rejects any `run.Npulses != 1`,
requires HR-3B on and HR-3C off, rebuilds the longitudinal schedule, and
requires exact equality with the frozen source z schedule.  It identifies the
first peak of `-min(delta_n)` and selects `peak-24 .. peak+23`: source indices
`7998..8045` around peak `8022`.

`run_optical_path` explicitly runs the **complete frozen optical trajectory**
using `propagate_one_pulse(... dz=prop.dz, z_max=prop.z_max ...)`.  The hook
only persists selected screen records.  The 48 records are therefore
HR-3/Streaming/Hydro qualification points, not optical z steps and not a
shortened propagation domain.

**E5-1 suitability verdict: `NOT_A_LEGAL_MULTIPULSE_CASE`.**  It cannot be
used directly because its prepared input is one-pulse by construction, has
selected rather than full-z slow-state coverage, and has no PRE/POST/NEXT
rollover across pulses.  A legal small E5-1 would still need an authorized
continuous full-z source/schedule, its exact frozen field and slow-state
provenance, a selected `Npulses`/`f_rep` contract, and the reviewed integrated
entry.  S3/S4R/S5 may supply only their qualified operator/lifecycle evidence;
their 48-screen source selection, one-pulse manifest, and 3--4 hour timings
cannot be reused as the formal E5 domain, pulse contract, or runtime estimate.

## 5. Frozen parameter source matrix

The table identifies values available for direct inheritance from the frozen
source.  `INHERITED_FROZEN` means a future approved contract can bind the
named value and hash without asking the user to re-tune it.  It does not grant
execution authority.  `QUALIFICATION_ONLY_NOT_FORMAL` is intentionally not
promoted to a formal campaign value.

| Parameter | Effective available value | Source / key | Status |
| --- | --- | --- | --- |
| wavelength | `8.0e-7 m` | `real_post_input_config.json`, `beam.lam0` | INHERITED_FROZEN |
| pulse duration / definition | `1.2e-13 s` FWHM | config `beam.tau_fwhm` | INHERITED_FROZEN |
| input amplitude definition | `P0_peak=1.7e10 W`, `E0_peak=0`, `energy_J=null` | config `beam` | INHERITED_FROZEN |
| beam / focus | flat-top-cosine, `w0=radius=1.979e-3 m`, edge fraction `0.9`, `f=0.95 m` | config `beam` | INHERITED_FROZEN |
| optical grid | `Nx=301`, `Ny=351`, `Nt=384`; `Lx=3.01e-3 m`, `Ly=3.51e-3 m`, `Twin=9.6e-13 s` | config `grid` | INHERITED_FROZEN |
| optical z schedule | `dz=1e-4 m`, `dz_focus=5e-5 m`, focus stepping on, `z_max=1.3 m` | config `propagation` | INHERITED_FROZEN |
| numerical precision | requested `fp64`; slow state float64 | post-reference and S3 manifests | INHERITED_FROZEN |
| gas / linear constants | `n0=1.00027`, `n2_air=7.8e-24` | config `beam` | INHERITED_FROZEN |
| ionization | full-time RK4, `tdiff`, N2/O2 `ppt_talebpour_i_lut`, rate table enabled | config `ionization` | INHERITED_FROZEN |
| Raman | enabled; `isaacs_rot_sinexp`, full Isaacs Eq.27, IIR / Heun | config `raman` and `propagation` | INHERITED_FROZEN |
| source field construction | runner input field plus thin lens; fresh `E_source.copy()` exists in runner | `runner.py`; S3 calls `build_transverse_input_field` then lens | INHERITED_FROZEN |
| HR-2 / HR-3 selection | HR-3B true; HR-3C false; selected S3 uses authoritative HR-3A/B POST | config `heat`; `hr4e5s_s3.py` | INHERITED_FROZEN |
| hydro geometry | `dx=dy=1e-5 m`, collocated nodal source | S3 input manifest / post reference | INHERITED_FROZEN |
| hydro material constants | `chi=2.17e-5 m2/s`, `nu=1.5e-5 m2/s`, `n0=1.00027`, gravity `(0,-9.81)` | S3 input manifest | INHERITED_FROZEN |
| hydro integration | `dt_hydro=1e-6 s`, 1,000 steps, CFL `1.0` | S3 input manifest | INHERITED_FROZEN |
| queue / block values | block `8`, queue depth `16` | S3 input manifest | QUALIFICATION_ONLY_NOT_FORMAL |
| repetition rate | `f_rep=1000 Hz` | config `heat.f_rep` | INHERITED_FROZEN |
| exact interpulse construction | `build_interpulse_step_schedule(f_rep, dt_hydro)` | `hr4d_pulse_lifecycle.py` | INHERITED_FROZEN |
| source state / PRE identity | E1B HR-3B source hashes named in §2 | prepared input manifest | INHERITED_FROZEN |
| `Npulses=1` | qualification-only single pulse | config `run.Npulses`; S3 guard | QUALIFICATION_ONLY_NOT_FORMAL |
| full-z `K=15000` state shape | historical source state shape `[15000,351,301]` | post-reference manifest | QUALIFICATION_ONLY_NOT_FORMAL |
| formal endpoint / comparison | unselected | no formal manifest | USER_AUTHORIZATION_REQUIRED |
| formal `Npulses` | unselected | no formal manifest | USER_AUTHORIZATION_REQUIRED |
| formal full-z case extent / selected source | unselected | no formal manifest | USER_AUTHORIZATION_REQUIRED |
| checkpoint and retention policy | unimplemented for multi-pulse Streaming rollover | code / no policy receipt | MISSING_FORMAL_VALUE |
| engineering/scientific PASS thresholds | unselected | no formal manifest | USER_AUTHORIZATION_REQUIRED |

## 6. Artifact inventory and storage accounting

Let `G = 3*K*Ny*Nx*dtype_bytes` be one float64 three-field generation.  The
evaluated reference uses only the historical illustration
`K=15000, Ny=351, Nx=301, dtype_bytes=8`; it is not a formal screen-count
selection.  It gives `G=38,034,360,000 B = 35.422 GiB`.

| Object | Owner / location | Layout and lifetime | Authority / simultaneous state | Evidence and confidence |
| --- | --- | --- | --- | --- |
| HR-4C authoritative slot | `HR4CThreeFieldStore`, disk-backed memmaps | 3 × `[K,Ny,Nx]` float64; one generation | retained; one of two fixed slots | `hr4c_state.py`; VERIFIED_FROM_CODE |
| HR-4C staging slot | same | 3 × `[K,Ny,Nx]` float64 | exists through transaction and remains reusable after swap | code; VERIFIED_FROM_CODE |
| Streaming CURRENT | `current/screen_*.npz` | K three-field 2D float64 records | remains after promotion | streaming code; VERIFIED_FROM_CODE |
| Streaming POST | `post/screen_*.npz` | K three-field 2D float64 records | required for barrier/restart; no demonstrated post-promotion deletion | code; VERIFIED_FROM_CODE |
| Streaming NEXT | `next/screen_*.npz` | K three-field 2D float64 records | becomes authoritative by pointer after barrier | code; VERIFIED_FROM_CODE |
| pointer / manifest | JSON at lifecycle root | metadata / hashes / queue / states | durable, negligible relative to arrays | code; VERIFIED_FROM_CODE |
| atomic NPZ temporary | same directory as destination | one 3-field 2D NPZ while writing | may coexist transiently with committed artifacts | `_atomic_npz`; VERIFIED_FROM_CODE |
| final optical field | optical run root | saved by `np.save`; exact formal dtype/retention unselected | per optical run | `run_optical_path`; formal size UNKNOWN |
| scientific ledger / deposition archives | optical run root | selected diagnostic arrays in S3 | output plan for formal E5 unselected | `hr4e5s_s3.py`; UNKNOWN for formal |
| LUT/cache, run logs, telemetry | run root / cache workspace | runtime-dependent | retention unselected | audit pack; UNKNOWN for formal |
| recovery / checkpoints / pulse history | future formal root | required conceptually, not an integrated entry | policy unselected | MISSING_FORMAL_VALUE |

### 6.1 Formulae and evaluated raw-payload lower bounds

| Store / event | Formula | Evaluated illustrative payload |
| --- | --- | ---: |
| one three-field generation | `G` | 35.422 GiB |
| HR-4C authoritative + staging | `2G` | 70.845 GiB |
| Streaming CURRENT | `G` | 35.422 GiB |
| Streaming POST | `G` | 35.422 GiB |
| Streaming NEXT | `G` | 35.422 GiB |
| Streaming persistent total | `3G` | 106.267 GiB |
| HR-4C + Streaming if concurrently retained | `5G` | 177.111 GiB |
| one atomic temporary raw field payload | `3*Ny*Nx*8` | 2.418 MiB |
| HR-4C operator working-set estimate, block 8 | `(6*8+12)*Ny*Nx*8` | 48.363 MiB host/operator accounting |

`70.845 GiB` applies only to HR-4C's two persistent three-field slots.  It
does not include Streaming, NPZ overhead, final fields, diagnostics, LUTs, or
historic checkpoints.  The Streaming three-namespace number is an uncompressed
array-payload lower bound; `np.savez` container/header/metadata, manifest,
directory and filesystem-allocation overhead are additional.  In existing
48-screen evidence, CURRENT files occupy 121,844,368 B, whereas their raw
payload is 121,709,952 B, confirming nonzero container/metadata overhead.

The code supports a constant two-slot HR-4C store over its own successive
generations, but it does not establish a formal multi-pulse Streaming rollover
or retention cleanup.  If independent full lifecycle roots were retained for
`N` pulses, Streaming raw retained payload is `3*N*G`; if namespaces are to be
reused, the lower/upper retained result depends on an unimplemented and
unapproved policy.  Hence the only defensible pre-policy peak range is
`3G..5G` (106.267..177.111 GiB for the illustration) plus all extras above,
not a complete run-root allocation.

## 7. HPC resources, environment, and evidence gaps

### 7.1 Completed qualification evidence

| Evidence | Terminal elapsed |
| --- | ---: |
| S3 batch `233394` | 03:06:26 |
| S3 stream `234320` / `234321` | 03:04:55 / 03:48:42 |
| S4R 1+2 `237236` | 03:44:29 |
| S4R 1+4 `237235` | 02:59:28 |
| S5 clean `241098` | 03:03:53 |

S4R 1+6 `237234` was cancelled before execution and is
`NOT TESTED / RESOURCE_UNAVAILABLE`.  The preferred tested topology remains
one optical plus four hydro GPUs; one plus two is the validated lower-resource
fallback.  The 48-screen timing is qualification evidence only and must not
be extrapolated linearly to full-z formal E5.

The partition has 40 GPUs, `DefCpuPerGPU=8`,
`DefMemPerCPU=MaxMemPerCPU=15750 MiB`, and partition `MaxTime=UNLIMITED`.
S5's 12-hour value is that job's request rather than a measured site ceiling.
The `normal` QoS numerical walltime/TRES limit was not exposed in the evidence.
S5-FINAL is intentionally a two-allocation fault/recovery contract and cannot
be called a single-allocation end-to-end workflow.

### 7.2 Environment and unobserved metrics

The fixed interpreter read on the login node reports Python 3.13.7, CuPy
13.6.0, NumPy 2.3.3, and SciPy 1.16.3.  Existing scheduler records have no
observed `MaxRSS`, `TotalCPU`, `MaxDiskRead`, `MaxDiskWrite`, or GPU peak-memory
value.  Requested memory is therefore not reported as observed memory.

GPU model, VRAM, driver, CUDA runtime, visible-device mapping, container digest,
user/project quota, and retention/purge policy are `NOT_VERIFIED`.  `nvidia-smi`
is unavailable on the login host, `quota` is unavailable, and the observed
JuiceFS space (300 GiB total / about 180 GiB available) is filesystem-wide
space, not an allocation or project quota.

Before a separately authorized formal job, a reviewed non-invasive job
prologue receipt should record interpreter path/package versions, GPU model,
VRAM, driver/CUDA runtime, device mapping, `sacct` identifiers, and measured
CPU/GPU/I/O telemetry locations.  Administrators must separately confirm
user/project quota, retention/purge rules, and effective normal-QoS limits.

## 8. Constraints and required user decisions

No execution should be planned until S5 passes and the following minimal
decisions are recorded:

1. Bind the formal science question, control/reference, formal endpoint, and
   `Npulses`; do not infer any of these from illustrations 3, 5, or 15,000.
2. Confirm use of the exact frozen input/state/source hashes above (or name a
   separately reviewed frozen baseline) and bind `f_rep`/full-z schedule to it.
3. Select 1+4 or 1+2 topology, job/allocation strategy, walltime and memory
   request after site-limit/quota confirmation; do not use 1+6.
4. Select the authoritative persistence design, checkpoint/POST/NEXT retention
   and cleanup policy, and whether HR-4C and Streaming ever coexist.
5. Select engineering/numerical-health acceptance thresholds and diagnostic
   retention; physical pulse-to-pulse trends remain observations, not a
   predeclared engineering PASS condition.
6. Review the minimal entry glue as a separate change request.  No implementation
   or Slurm submission is authorized by this audit.

## 9. Stop/go recommendation

**Go only to manual review and contract design.**  The evidence is internally
consistent enough for E5-0 review.  It does not remove the S5 entry gate,
does not establish a production multi-pulse executor, and does not establish
storage or quota sufficiency for a full-z campaign.  Formal execution remains
`false` until the user records the decisions above after verified S5 closure.
