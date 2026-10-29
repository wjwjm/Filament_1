"""Bounded SSH byte relay; scientific arrays are never saved on this machine."""
import datetime,json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent.parent
src=['ssh','-o','BatchMode=yes','t0-hpc','/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python','-s','-B','/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46a_20261029_r2/preposition_full_source_A01.py','c8d2358605ee5a2d3f7b51bbbe416a550a7d63572f34cf258fe4ab66021fe456']
dst=['ssh','-o','BatchMode=yes','scvi-hpc','/data/home/scvi806/.conda/envs/Filament_python/bin/python','-s','-B','/data/run01/scvi806/user_Wangjimin/projects/filament_cpu_s46a_20261029_r6/preposition_full_receive_A01.py','59eaa5fcae52a331b5d29a6f54fadde65d8ed616a37553affbe4aa4a0d4a7a4d']
start=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/'FULL_CROSS_RELAY_SOURCE_A01.stderr.log').open('xb') as se,(r/'FULL_CROSS_RELAY_DEST_A01.stderr.log').open('xb') as de,(r/'FULL_CROSS_RELAY_DEST_A01.stdout.log').open('xb') as do:
    receiver=subprocess.Popen(dst,stdin=subprocess.PIPE,stdout=do,stderr=de)
    source=subprocess.Popen(src,stdout=receiver.stdin,stderr=se);receiver.stdin.close()
    a=source.wait(timeout=1800);b=receiver.wait(timeout=600)
with (r/'FULL_CROSS_RELAY_RECEIPT_A01.json').open('x',encoding='utf8') as f:json.dump(dict(source_argv=src,destination_argv=dst,start_utc=utc,elapsed_s=time.monotonic()-start,source_exit=a,destination_exit=b,raw_arrays_saved_local=False),f,indent=2)
assert a==b==0
print((r/'FULL_CROSS_RELAY_DEST_A01.stdout.log').read_text())
