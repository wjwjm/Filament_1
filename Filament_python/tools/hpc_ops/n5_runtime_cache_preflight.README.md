## Joint E5-N5 runtime environment and Raman cache preflight

The sourced n5_runtime_env.sh binds HOME, XDG_CACHE_HOME, CUPY_CACHE_DIR,
TMPDIR, TMP, and TEMP to private directories beneath
<RUN_ROOT>_entry/runtime_env. Source it after RUN_ROOT is set and before
Conda activation or any project Python/CuPy import. It creates only missing
directories and fails closed for symlinks, noncanonical paths, wrong owner,
non-private mode, or storage on a different filesystem.

The stdlib-first n5_runtime_cache_preflight.py validate command checks the
inherited paths, cwd, Python home and temp resolution, same-filesystem
placement, permissions, write/fsync/atomic-rename/readback, then writes a
no-overwrite JSON receipt. It does not import CuPy.

For a one-GPU allocation, run probe --phase cold with the fixed production
interpreter and an empty project CUPY_CACHE_DIR, then run probe --phase
warm in a fresh process with the same interpreter, environment, config and
single visible GPU token, passing the cold receipt. The probe loads the
formal config, requires Nt=384, isaacs_rot_sinexp, IIR and
exact_piecewise_linear, and evaluates the existing isaacs_raman_stage on
a synthetic complex128 [384,2,2] field. It synchronizes CUDA and checks
finite outputs plus the actual CuPy compiler cache path. It never loads a
scientific source field or calls propagate_one_pulse. Cold receipts require
new cache artifacts; warm receipts require a fresh PID and an unchanged
raw-byte/size/link inventory.

Example shell entry after the single sourced environment contract:

~~~bash
source "$REPO_DIR/Filament_python/tools/hpc_ops/n5_runtime_env.sh"
conda activate /data/home/scvi806/.conda/envs/Filament_python
cd -- "$RUN_ROOT"
/data/home/scvi806/.conda/envs/Filament_python/bin/python -s -B \
  "$REPO_DIR/Filament_python/tools/hpc_ops/n5_runtime_cache_preflight.py" \
  validate --run-root "$RUN_ROOT" --output "$N5_ENTRY_ROOT/RUNTIME_ENV_STARTUP.json"
~~~

The caller supplies UPPE_USE_GPU=1 and exactly one CUDA_VISIBLE_DEVICES
token only to each diagnostic GPU child. All cache and temp path variables
remain inherited unchanged by coordinator, health, probe, and formal driver.
The cold/warm cache inventory and phase precondition run before confio.load_all.
The formal import chain confio -> config_normalize -> constants/hr4 -> device
may import CuPy; it is allowed only after this cache boundary and the inherited
project environment gate. The probe does not assume configuration imports are CPU-only.
