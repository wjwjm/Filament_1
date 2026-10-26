"""G1 tree service around the unchanged Q1 bounded-memory, full-element exact gate."""
from __future__ import annotations
import io,json,zipfile
from pathlib import Path
import numpy as np
from g1_common import require,write

CLOCK_LEAVES=frozenset({'walltime_s','operator_walltime_s'})

def layout(a):
    depth=0;base=a
    while isinstance(base.base,np.ndarray):depth+=1;base=base.base
    return dict(shape=list(a.shape),dtype=str(a.dtype),strides=list(a.strides),C=bool(a.flags.c_contiguous),F=bool(a.flags.f_contiguous),OWNDATA=bool(a.flags.owndata),WRITEABLE=bool(a.flags.writeable),base_chain_depth=depth,base_ndarray=isinstance(a.base,np.ndarray))

def arrays(value,path='bundle'):
    if isinstance(value,np.ndarray):yield path,value
    elif isinstance(value,dict):
        for key,child in value.items():yield from arrays(child,path+'.'+str(key))
    elif isinstance(value,(tuple,list)):
        for index,child in enumerate(value):yield from arrays(child,path+'.'+str(index))

def aliases(value):
    named=list(arrays(value));result=[]
    for i,(name,left) in enumerate(named):
        for other,right in named[i:]:result.append(dict(left=name,right=other,same_object=left is right,shares_memory=bool(np.shares_memory(left,right))))
    return result

def fingerprints(value):
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    return {name:dict(canonical_sha256=j1._canonical_hash(a),layout=layout(a),nbytes=int(a.nbytes)) for name,a in arrays(value)}

def normalized_return(value,post,sinks,profile_path):
    """Retain original profile JSON and decode only its frozen timing leaves."""
    result=list(value);diag=dict(result[2]);sidecars={}
    for key in ('bk_nee_profile_halfstep_1_json','bk_nee_profile_halfstep_2_json'):
        if key in diag:
            array=diag[key];sidecars[key]=dict(original_values=array.tolist(),original_layout=layout(array))
            diag[key]=[json.loads(str(v)) for v in array]
    if sidecars:write(profile_path,dict(raw_profile_sidecars=sidecars,clock_leaves=sorted(CLOCK_LEAVES),additional_timing_exclusions=[]))
    result[2]=diag
    return dict(returned=tuple(result),POST=post,sinks=sinks)

def save_tree(root,value,*,compressed=False):
    root=Path(root);root.mkdir(exist_ok=False);counter=0
    archive=zipfile.ZipFile(root/'ARRAYS.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) if compressed else None
    def encode(v,name='bundle'):
        nonlocal counter
        if isinstance(v,dict):return dict(kind='dict',items={str(k):encode(w,name+'.'+str(k)) for k,w in v.items() if str(k) not in CLOCK_LEAVES})
        if isinstance(v,(tuple,list)):return dict(kind=type(v).__name__,items=[encode(w,name+'.'+str(i)) for i,w in enumerate(v)])
        if isinstance(v,np.ndarray):
            filename=f'{counter:04d}.npy';counter+=1
            if archive is None:
                with (root/filename).open('xb') as stream:np.save(stream,v,allow_pickle=False)
            else:
                with archive.open(filename,'w',force_zip64=True) as stream:np.lib.format.write_array(stream,v,allow_pickle=False)
            return dict(kind='array',file=filename,layout=layout(v))
        if isinstance(v,np.generic):return dict(kind='scalar',dtype=str(v.dtype),value=v.item())
        if isinstance(v,(float,int,bool,complex)):
            a=np.asarray(v);return dict(kind='scalar',dtype=str(a.dtype),value=a.item())
        return dict(kind='other',value=v)
    try:tree=encode(value)
    finally:
        if archive is not None:archive.close()
    write(root/'TREE.json',tree);write(root/'ALIASES.json',aliases(value));write(root/'FINGERPRINTS.json',fingerprints(value))
    return str(root)

def compare(expected,actual,output,*,disk=False):
    """Full byte/value/NaN/sign-zero comparison; no tolerance or sampling."""
    from tools.cpu_o1 import q1_t0
    from tools.cpu_o1.numeric_gate import classify_comparison,TIMING_FIELDS,NAN_PLACEHOLDER_FIELDS
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    zip_reader=None
    if disk:
        reference=Path(expected);tree=json.loads((reference/'TREE.json').read_text());expected_aliases=json.loads((reference/'ALIASES.json').read_text())
        if (reference/'ARRAYS.zip').is_file():zip_reader=zipfile.ZipFile(reference/'ARRAYS.zip','r')
    else:
        reference=None;expected_aliases=aliases(expected)
        def node(v):
            if isinstance(v,dict):return dict(kind='dict',items={str(k):node(w) for k,w in v.items() if str(k) not in CLOCK_LEAVES})
            if isinstance(v,(tuple,list)):return dict(kind=type(v).__name__,items=[node(w) for w in v])
            if isinstance(v,np.ndarray):return dict(kind='array',value=v,layout=layout(v))
            if isinstance(v,(float,int,bool,complex,np.generic)):
                a=np.asarray(v);return dict(kind='scalar',dtype=str(a.dtype),value=v)
            return dict(kind='other',value=v)
        tree=node(expected)
    count=0;layout_count=0;clock_count=0
    with output.open('x',encoding='utf-8') as stream:
        def emit(row):stream.write(json.dumps(row,allow_nan=False,default=str)+'\n');stream.flush()
        def visit(n,got,name):
            nonlocal count,layout_count,clock_count
            leaf=name.rsplit('.',1)[-1]
            if leaf in CLOCK_LEAVES:
                clock_count+=1;emit(dict(field_name=name,classification='TIMING_EXCLUDED',reason='frozen_clock_field_only'));return
            kind=n['kind']
            if kind=='dict':
                require(isinstance(got,dict) and set(n['items'])==set(got)-CLOCK_LEAVES,'TREE_FIELDS',name)
                for key,child in n['items'].items():visit(child,got[key],name+'.'+key)
                return
            if kind in ('tuple','list'):
                require(type(got).__name__==kind and len(got)==len(n['items']),'TREE_SEQUENCE',name)
                for i,child in enumerate(n['items']):visit(child,got[i],name+'.'+str(i))
                return
            if kind=='other':require(got==n['value'],'TREE_METADATA',name);emit(dict(field_name=name,classification='METADATA_EXACT'));return
            if kind=='array':
                if not disk:left=n['value']
                elif zip_reader is None:left=np.load(reference/n['file'],mmap_mode='r',allow_pickle=False)
                else:
                    with zip_reader.open(n['file']) as member:left=np.lib.format.read_array(member,allow_pickle=False)
            else:left=np.asarray(n['value'],dtype=n['dtype'])
            right=np.asarray(got)
            if kind=='array':
                require(isinstance(got,np.ndarray),'ARRAY_TOPOLOGY',name)
                layout_count+=1;actual_layout=layout(got)
                if n['layout']!=actual_layout:
                    write(output.with_suffix('.FIRST_FAILURE.json'),dict(field=name,reason='NATIVE_LAYOUT',expected=n['layout'],actual=actual_layout))
                    with output.with_suffix('.FAILED_ARRAY.npy').open('xb') as failure:np.save(failure,got,allow_pickle=False)
                    raise RuntimeError('NATIVE_LAYOUT:'+name)
            if left.dtype.kind in 'US':
                require(left.shape==right.shape and left.dtype==right.dtype and np.array_equal(left,right),'STRING_ARRAY_EXACT',name)
                emit(dict(field_name=name,classification='METADATA_EXACT',reference_layout=n.get('layout'),candidate_layout=layout(got) if isinstance(got,np.ndarray) else None));return
            scope='timing' if leaf in TIMING_FIELDS else ('diagnostics' if leaf in NAN_PLACEHOLDER_FIELDS else 'science')
            row=q1_t0._array_comparison_row(leaf,left,got,scope=scope,zero_mask_path=output.parent/(output.stem+'_zero_masks')/(f'{count:05d}.bin'))
            gate=classify_comparison(row);row.update(tree_field=name,classification=gate,reference_layout=n.get('layout'),candidate_layout=layout(got) if isinstance(got,np.ndarray) else None)
            count+=1;emit(row)
            if not gate.get('proceed'):
                write(output.with_suffix('.FIRST_FAILURE.json'),row)
                with output.with_suffix('.FAILED_ARRAY.npy').open('xb') as failure:np.save(failure,got,allow_pickle=False)
                raise RuntimeError('FIRST_SCIENTIFIC_NONEXACT:'+name)
        visit(tree,actual,'bundle')
    if zip_reader is not None:zip_reader.close()
    actual_aliases=aliases(actual);alias_ok=actual_aliases==expected_aliases
    write(output.with_suffix('.ALIASES.json'),dict(status='PASS' if alias_ok else 'FAIL',reference=expected_aliases,candidate=actual_aliases))
    require(alias_ok,'ALL_PAIRWISE_ALIAS_EXACT',str(output))
    return dict(status='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS',numeric_rows=count,array_layouts=layout_count,alias_relations=len(actual_aliases),comparison_path=str(output),additional_timing_exclusions=[])
