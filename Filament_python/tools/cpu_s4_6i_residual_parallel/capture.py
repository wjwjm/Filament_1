"""Reference-only capture of twelve real power-volume inputs, without retained arrays."""
from pathlib import Path
import numpy as np
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from .selected_protocol import CAPTURE_INTERVALS, CONTEXTS
from .power_volume import SOURCE_SITE, validate_power_volume_input

def fingerprint(array):
    value=validate_power_volume_input(array)
    return dict(shape=list(value.shape),dtype=str(value.dtype),strides=list(value.strides),
        c_contiguous=bool(value.flags.c_contiguous),writeable=bool(value.flags.writeable),
        owns_data=bool(value.flags.owndata),raw_sha256=pv.raw_sha256(memoryview(value).cast('B')))

def array_record(path):
    return dict(path=path.name,size_bytes=path.stat().st_size,
        classification='external',hash_scope='raw_bytes',raw_sha256=pv.raw_sha256_file(path))

class CaptureStore:
    def __init__(self,root,*,shape):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=False)
        self.shape=tuple(shape)
        self.calls=[]
        self.records=[]
        self.pending=None
        self.closed=False

    def before_call(self,*,context,interval_index,source,source_site):
        if self.closed or self.pending is not None:
            raise RuntimeError('S46I_CAPTURE_LIFECYCLE')
        index=len(self.calls)
        if index>=32 or (interval_index,context)!=(index//4,CONTEXTS[index%4]):
            raise RuntimeError('S46I_CAPTURE_CALL_ORDER')
        validate_power_volume_input(source)
        if tuple(source.shape)!=self.shape or source_site!=SOURCE_SITE:
            raise RuntimeError('S46I_CAPTURE_DOMAIN')
        self.calls.append(dict(context=context,interval_index=interval_index,source_site=source_site))
        if interval_index not in CAPTURE_INTERVALS:
            return None
        key=f'i{interval_index+1}_{context}'
        fp=fingerprint(source)
        path=self.root/(key+'_input.npy')
        with path.open('xb') as handle:
            np.save(handle,source,allow_pickle=False)
        token=dict(key=key,context=context,interval_index=interval_index,
                   source_site=source_site,input_fingerprint=fp,input_file=array_record(path))
        self.pending=token
        return token

    def after_call(self,token,*,source,output):
        if token is None:
            return
        if token is not self.pending or fingerprint(source)!=token['input_fingerprint']:
            raise RuntimeError('S46I_CAPTURE_INPUT_CHANGED')
        if np.shares_memory(source,output) or output is source:
            raise RuntimeError('S46I_CAPTURE_OUTPUT_ALIAS')
        token['output_fingerprint']=fingerprint(output)
        token['input_output_share_memory']=False
        path=self.root/(token['key']+'_output.npy')
        with path.open('xb') as handle:
            np.save(handle,output,allow_pickle=False)
        token['output_file']=array_record(path)
        self.records.append(dict(token))
        self.pending=None

    def finish(self,*,binding):
        if self.closed or self.pending is not None or len(self.calls)!=32 or len(self.records)!=12:
            raise RuntimeError('S46I_CAPTURE_COMPLETENESS')
        self.closed=True
        receipt=dict(schema='filament.cpu_s4_6i.power_capture.v1',shape=list(self.shape),
            selected_interval_indices=list(CAPTURE_INTERVALS),contexts=list(CONTEXTS),
            total_calls=32,captured_inputs=12,calls=self.calls,records=self.records,
            source_binding=binding,records_scalar_only=True,retained_array_references=0,
            hash_scope='raw_bytes',array_storage='NPY_ALLOW_PICKLE_FALSE')
        sg.write(self.root/'CAPTURE.json',receipt)
        return receipt

def load_capture(root,*,shape):
    root=Path(root).resolve(strict=True)
    receipt=sg.read(root/'CAPTURE.json')
    expected=[(i,context) for i in CAPTURE_INTERVALS for context in CONTEXTS]
    if receipt.get('shape')!=list(shape) or receipt.get('total_calls')!=32 or receipt.get('captured_inputs')!=12:
        raise RuntimeError('S46I_CAPTURE_PACKAGE_DOMAIN')
    if [(r['interval_index'],r['context']) for r in receipt['records']]!=expected:
        raise RuntimeError('S46I_CAPTURE_PACKAGE_ORDER')
    for row in receipt['records']:
        for name in ('input','output'):
            record=row[name+'_file']
            path=root/record['path']
            if path.parent!=root or record!=array_record(path):
                raise RuntimeError('S46I_CAPTURE_ARTIFACT_IDENTITY')
    return receipt
