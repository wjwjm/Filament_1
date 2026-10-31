"""S46C fixed t0 login check: metadata only, no array payloads or writes."""
import getpass
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

assert getpass.getuser() == 't0s000727' and not os.environ.get('SLURM_JOB_ID'), 'ACCOUNT_LOGIN_NODE'
home = Path('/publicfs01/fs1-t/home/t0s000727')
assert Path.home() == home, 'HOME_IDENTITY'
project = home / 'user_wjm'
old = project / 'filament_cpu_s46a_20261029_r2'
code = old / 'code'
assert code.resolve(strict=True).is_relative_to(project.resolve(strict=True)), 'BASE_PATH'
def command(argv):
    p = subprocess.run(argv, cwd=code, capture_output=True, text=True, timeout=25)
    return dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
head = command(['/usr/bin/git', 'rev-parse', 'HEAD'])
assert head['exit_code'] == 0 and head['stdout'].strip() == 'ed8754041625549a80959db7a8abd804d5d219ae', 'FROZEN_BASE'
clean = command(['/usr/bin/git', 'status', '--porcelain'])
assert clean['exit_code'] == 0 and not clean['stdout'].strip(), 'FROZEN_BASE_CLEAN'
stage = old / 'report_full/runs/CPU/384x351x301/QUAL_OFF/work/formal/STAGES'
files = []
for name in ('TREE.json', 'ALIASES.json', 'FINGERPRINTS.json', 'ARRAYS.zip'):
    path = stage / name
    assert path.is_file() and os.access(path, os.R_OK), 'REQUIRED_STAGE_FILE:' + name
    files.append(dict(path=str(path), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
metadata = {name: json.loads((stage / name).read_text()) for name in ('TREE.json', 'FINGERPRINTS.json')}
commands = [command(['squeue', '-u', 't0s000727', '-h', '-o', '%i|%j|%T|%M|%l|%C|%m|%R']),
            command(['scontrol', 'show', 'partition', 'v6_384'])]
assert all(c['exit_code'] == 0 for c in commands), 'SCHEDULER_ACCESS'
print(json.dumps(dict(account=getpass.getuser(), home=str(home), executable=sys.executable,
                     python=sys.version, versions={p: importlib.metadata.version(p) for p in
                        ('numpy', 'scipy', 'numba', 'llvmlite', 'psutil', 'threadpoolctl')},
                     base_commit=head['stdout'].strip(), base_clean=True, stage_files=files,
                     stage_metadata=metadata, commands=commands,
                     project_free_bytes=shutil.disk_usage(project).free,
                     new_job_ids=0, remote_writes=0, scientific_payloads_opened=0), allow_nan=False))
