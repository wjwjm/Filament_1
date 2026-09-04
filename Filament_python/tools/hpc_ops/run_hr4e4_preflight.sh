#!/usr/bin/env bash
set -euo pipefail
readonly REPO="$1" OUT="$2" EXPECTED_SHA="$3"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"; readonly PROVENANCE="${OUT%.json}_provenance_v2.json"; readonly AUDIT="${OUT%.json}_advection_reuse.json"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"; test ! -e "$OUT" && test ! -e "$PROVENANCE" && test ! -e "$AUDIT"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" create --repo "$REPO" --output "$PROVENANCE" --tracked Filament_python/KHz_filament/hr4.py Filament_python/KHz_filament/device.py Filament_python/KHz_filament/hr4e_reduced_limits.py Filament_python/tools/audit_hr4e4_advection_reuse.py Filament_python/tools/run_hr4e4_case.py Filament_python/tools/summarize_hr4e4.py Filament_python/tools/hr4e4_case.sbatch Filament_python/tools/hpc_ops/run_hr4e4_preflight.sh Filament_python/tools/hpc_ops/submit_hr4e4.sh >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$REPO" --manifest "$PROVENANCE" --require-hash-scope >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/audit_hr4e4_advection_reuse.py" --out "$AUDIT" >/dev/null
"$PYTHON" - "$OUT" "$AUDIT" "$PROVENANCE" <<'PY'
import json,sys
audit=json.load(open(sys.argv[2],encoding='utf-8'))
assert audit['status']=='ADVECTION_EVIDENCE_REUSE_PASS'
json.dump({'schema':'khz_filament.hr4e4.preflight.v1','status':'PASS','advection_reuse_audit':sys.argv[2],'provenance':sys.argv[3],'e4_case_count':6,'no_hr4e5_hr4f_hr5_started':True},open(sys.argv[1],'w',encoding='utf-8'),indent=2,sort_keys=True)
PY
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","preflight":"%s","audit":"%s","provenance":"%s"}\n' "$OUT" "$AUDIT" "$PROVENANCE"
