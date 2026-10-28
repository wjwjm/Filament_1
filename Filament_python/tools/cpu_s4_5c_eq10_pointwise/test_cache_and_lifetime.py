"""Fixed serial/parallel cache crossover, lifetime, and exceptional mask tests."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from .kernels import KernelExecutor

def test_s45c_retained_outputs_and_exception_mask(tmp_path):
    import numba
    executor=KernelExecutor(workers=2,diagnostic_local=True)
    x=np.ones((384,3,35),dtype=np.complex128)*(1.25-.5j)
    coeff=(1j*np.arange(384,dtype=np.float64)*float(2.5e-16))[:,None,None]
    old=executor.k1(coeff,x);before=old.tobytes();old2=executor.k2(old.real,1.125);before2=old2.tobytes()
    executor.k1(coeff,x*2.0);executor.k2(old.real,2.25)
    assert old.tobytes()==before and old2.tobytes()==before2
    assert not np.shares_memory(old,x) and not np.shares_memory(old2,old)
    saved=executor._runtime['dispatchers']['k1'];mask=numba.get_num_threads()
    def fail(*unused):raise RuntimeError('INTENTIONAL_FIXED_KERNEL_FAILURE')
    fail.py_func=saved[1].py_func;fail.signatures=[]
    executor._runtime['dispatchers']['k1']=(saved[0],fail)
    try:
        with pytest.raises(RuntimeError,match='INTENTIONAL'):executor.k1(coeff,x)
        assert numba.get_num_threads()==mask
        assert executor.snapshot()['dispatches'][-1]['status']=='ERROR'
    finally:executor._runtime['dispatchers']['k1']=saved

def test_s45c_serial_parallel_serial_new_process_cache(tmp_path):
    code='''import json,sys,numpy as np
from tools.cpu_s4_5c_eq10_pointwise.kernels import KernelExecutor
w=int(sys.argv[1]);e=KernelExecutor(workers=w,diagnostic_local=w==2)
x=np.ones((384,3,35),dtype=np.complex128)*(1.25-.5j)
c=(1j*np.arange(384,dtype=np.float64)*float(2.5e-16))[:,None,None]
y=e.k1(c,x);z=e.k2(y.real,1.125)
if y.tobytes()!=(c*x).tobytes() or z.tobytes()!=(y.real/1.125).tobytes():raise RuntimeError('CACHE_CROSS_NONEXACT')
d=e._runtime['dispatchers'];s=e.snapshot()
s['cache_observed']={name:{'hits':sum(fn._cache_hits.values()),'misses':sum(fn._cache_misses.values())} for name,pair in d.items() for fn in [pair[0 if w==1 else 1]]}
print(json.dumps(s))'''
    env=os.environ.copy();env['NUMBA_NUM_THREADS']='2';env['NUMBA_CACHE_DIR']=str(tmp_path/'isolated_shared_jit')
    rows=[]
    for i,workers in enumerate((1,2,1)):
        proc=subprocess.run([sys.executable,'-s','-B','-c',code,str(workers)],env=env,capture_output=True,text=True,timeout=90)
        (tmp_path/f'cache_{i}.stdout.json').write_text(proc.stdout,encoding='utf-8');(tmp_path/f'cache_{i}.stderr.log').write_text(proc.stderr,encoding='utf-8')
        assert proc.returncode==0,proc.stderr
        row=json.loads(proc.stdout);rows.append(row)
        assert all(item['active_thread_ids']==list(range(workers)) for item in row['dispatches'])
        assert all(item['thread_mask_after']==item['thread_mask_before'] for item in row['dispatches'])
    assert all(item['misses']>0 for item in rows[0]['cache_observed'].values())
    assert all(item['misses']>0 for item in rows[1]['cache_observed'].values())
    assert all(item['hits']>0 for item in rows[2]['cache_observed'].values())
    (tmp_path/'CACHE_CROSSOVER_RECEIPT.json').write_text(json.dumps(dict(status='PASS_ACTUAL_SERIAL_PARALLEL_SERIAL',records=rows),indent=2),encoding='utf-8')
