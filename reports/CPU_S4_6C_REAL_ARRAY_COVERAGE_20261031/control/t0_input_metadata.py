import json, os
from pathlib import Path
root = Path('/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46a_20261029_r2')
assert not os.environ.get('SLURM_JOB_ID')
plan = json.loads((root/'report_full/MEASUREMENT_PLAN.json').read_text())
canonical = Path(plan['canonical_root'])/'384x351x301'
data = root/'report/external_data/input_bundle'
print(json.dumps(dict(canonical=str(canonical), manifest=json.loads((canonical/'INPUT_MANIFEST.json').read_text()),
                     data_root=str(data), data_config_exists=(data/'SMALLGRID_CONFIG.json').is_file(),
                     historical_receipt_exists=(root/'report/external_data/G2_BOUNDARY_RECEIPT.json').is_file(),
                     python=plan['python'], config=json.loads((canonical/'config.json').read_text())),allow_nan=False))
