"""Actual Windows/Linux ZIP extraction, lossless restore and readonly replay."""
import argparse,json,os,runpy,subprocess,sys,time,zipfile
from pathlib import Path

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def native(value):return value.replace('\\','/').replace('C:/','/mnt/c/').replace('D:/','/mnt/d/')
def require(ok,reason):
    if not ok:raise RuntimeError(reason)

def main():
    p=argparse.ArgumentParser();p.add_argument('platform',choices=['Windows','Linux']);args=p.parse_args()
    c=Path(__file__).resolve().parent
    build=load(c/'ZIP_COMPACT_BUILD_RECEIPT_A01.json')
    if args.platform=='Windows':
        archive=Path(build['path']);runtime=sys.executable
        base=c/'FINAL_COMPACT_WINDOWS_A01'
        helper=c.parent/'CPU_S4_6A_REVIEW_20261029_002/source_roles/CPU/executed_source/Filament_python/tools/hpc_ops/provenance_v2.py'
    else:
        ctx=load(c/'LINUX_DELIVERY_CONTEXT_A02.json')
        archive=Path(native(build['path']));runtime=ctx['runtime']
        base=Path(ctx['native_root'])/'FINAL_COMPACT_LINUX_A01'
        helper=Path(ctx['package'])/'source_roles/CPU/executed_source/Filament_python/tools/hpc_ops/provenance_v2.py'
    pv=runpy.run_path(str(helper))
    require(pv['raw_sha256_file'](archive)==build['raw_sha256'],'ACTUAL_COMPACT_ZIP_SHA256')
    base.mkdir(exist_ok=False);extracted=base/'extracted';restored=base/'restored_review'
    started=time.perf_counter()
    with zipfile.ZipFile(archive) as z:z.extractall(extracted)
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1');env.pop('PYTHONPATH',None)
    commands=[]
    def run(label,argv):
        result=subprocess.run(argv,cwd=extracted,env=env,capture_output=True,text=True,encoding='utf-8')
        for stream,value in [('stdout',result.stdout),('stderr',result.stderr)]:
            with (c/(args.platform.upper()+'_COMPACT_'+label+'_A01.'+stream+'.log')).open('x',encoding='utf-8') as f:f.write(value)
        commands.append(dict(stage=label,argv=argv,exit_code=result.returncode))
        if result.stdout:print(result.stdout.strip(),flush=True)
        require(result.returncode==0,'COMPACT_'+label+'_EXIT:'+str(result.returncode))
    run('RESTORE',[runtime,'-s','-B',str(extracted/'restore_review.py'),'--output',str(restored)])
    output=c/(args.platform.upper()+'_FINAL_COMPACT_REPLAY_A01.json')
    run('REPLAY',[runtime,'-s','-B',str(restored/'validate_final_delivery_A01.py'),'--root',str(restored),'--output',str(output)])
    restore=load(restored.with_name(restored.name+'_RESTORE_RECEIPT.json'))
    require(restore['logical_files']==build['logical_files'],'RESTORED_LOGICAL_FILE_COUNT')
    receipt=dict(status='PASS_ACTUAL_FINAL_COMPACT_ZIP_RESTORE_AND_REPLAY',platform=args.platform,
        archive_raw_sha256=build['raw_sha256'],archive_size_bytes=build['size_bytes'],
        restored_original_logical_files=restore['logical_files'],restore=restore,
        replay_output=str(output),commands=commands,wall_s=time.perf_counter()-started,
        scientific_arrays=0,conclusion='TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED')
    with (c/(args.platform.upper()+'_FINAL_COMPACT_EXECUTION_A01.json')).open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
