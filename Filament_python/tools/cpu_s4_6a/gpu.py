"""Original CuPy optical path with scalar host clocks and deferred CUDA events."""
from __future__ import annotations
import ast
from contextlib import contextmanager, nullcontext
import copy
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
from types import FunctionType
from .inputs import context, require, write
from .entry import prepare_function, verify_source, post_timeline


def read_snapshot(root):
    """Restore an existing G1 snapshot for its unchanged element comparator."""
    import io
    import zipfile
    import numpy as np
    root = Path(root)
    with zipfile.ZipFile(root/'ARRAYS.zip') as archive:
        def decode(node):
            kind = node['kind']
            if kind == 'array':
                return np.load(io.BytesIO(archive.read(node['file'])),allow_pickle=False)
            if kind == 'dict':
                return {key:decode(value) for key,value in node['items'].items()}
            if kind in ('tuple','list'):
                return (tuple if kind=='tuple' else list)(decode(value) for value in node['items'])
            if kind == 'scalar':
                return np.asarray(node['value'],dtype=node['dtype'])[()]
            return node['value']
        return decode(json.loads((root/'TREE.json').read_text(encoding='utf-8')))


class Meter:
    """Retain scalar clocks/events only; never retain numerical arguments."""
    def __init__(self, cp, mode, intervals):
        self.cp, self.mode, self.intervals = cp, mode, intervals
        self.rows, self.stack, self.events = [], [], []
        self.interval = 0
        self.stream = int(cp.cuda.get_current_stream().ptr)

    @contextmanager
    def scope(self, label, interval_index=None, occurrence_class='PER_INTERVAL', source_site=''):
        if interval_index is not None:
            self.interval = int(interval_index)
        index = self.interval
        if self.mode in ('OFF', 'REFERENCE') and label != 'INTERVAL_BODY':
            yield
            return
        require(len(self.rows) < 8192, 'GPU_SCALAR_RECORD_CAP')
        require(int(self.cp.cuda.get_current_stream().ptr) == self.stream, 'GPU_STREAM_IDENTITY')
        if self.mode == 'SERIALIZED':
            self.cp.cuda.runtime.deviceSynchronize()
        token = len(self.rows)
        row = dict(token=token, parent=self.stack[-1] if self.stack else None, label=label,
                   interval=index, source_site=source_site, stream_ptr=self.stream,
                   start_wall_ns=time.perf_counter_ns(), start_cpu_ns=time.process_time_ns())
        start = end = None
        if self.mode in ('LIGHT', 'SERIALIZED'):
            start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
            start.record()
        self.rows.append(row); self.stack.append(token)
        try:
            yield
        except BaseException:
            row['failed'] = True
            raise
        finally:
            if end is not None:
                end.record()
            anchor = label == 'INTERVAL_BODY' and index in (0, self.intervals - 1)
            if anchor or self.mode == 'SERIALIZED':
                self.cp.cuda.runtime.deviceSynchronize()
            row.update(end_wall_ns=time.perf_counter_ns(), end_cpu_ns=time.process_time_ns(),
                       completion_anchor=anchor, failed=row.get('failed', False))
            require(self.stack.pop() == token, 'GPU_SCOPE_STACK')
            if start is not None:
                self.events.append((token, start, end))

    def finish(self):
        self.cp.cuda.runtime.deviceSynchronize()
        require(not self.stack, 'GPU_UNCLOSED_SCOPE')
        for token, start, end in self.events:
            self.rows[token]['device_event_span_ms'] = float(self.cp.cuda.get_elapsed_time(start, end))
        for row in self.rows:
            row['wall_ns'] = row['end_wall_ns'] - row['start_wall_ns']
            row['process_cpu_ns'] = row['end_cpu_ns'] - row['start_cpu_ns']
        return dict(records=self.rows, device_clock='CUDA events on one verified current stream; includes device idle gaps',
                    host_clock='perf_counter_ns/process_time_ns; async submission except explicit anchors',
                    kernel_activity='UNKNOWN_WITHOUT_TRACE', mode=self.mode, events_read_after_one_final_synchronization=True)


def host_tree(value, cp, transfers, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, cp.ndarray):
        if id(value) not in memo:
            memo[id(value)] = value.get(order='A')
            transfers.append(dict(direction='D2H', bytes=int(value.nbytes), reason='NORMAL_OUTPUT_OR_QUALIFICATION_SNAPSHOT'))
        return memo[id(value)]
    if isinstance(value, dict):
        return {key: host_tree(item, cp, transfers, memo) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(host_tree(item, cp, transfers, memo) for item in value)
    return value


def native_layout(value, cp):
    rows, groups = [], {}
    def visit(item, path):
        if isinstance(item,cp.ndarray):
            base = int(item.data.mem.ptr)
            groups.setdefault(base,len(groups))
            rows.append(dict(path=path,shape=list(item.shape),dtype=str(item.dtype),strides=list(item.strides),
                             storage_group=groups[base],offset_bytes=int(item.data.ptr)-base,
                             c_contiguous=bool(item.flags.c_contiguous),f_contiguous=bool(item.flags.f_contiguous)))
        elif isinstance(item,dict):
            for name,child in sorted(item.items()):visit(child,path+'.'+str(name))
        elif isinstance(item,(tuple,list)):
            for index,child in enumerate(item):visit(child,path+'.'+str(index))
    visit(value,'bundle')
    return rows


def _load_s45b_instrument_service():
    """Load only S45B's scalar AST service, bypassing its CPU package init."""
    import importlib.util
    from types import ModuleType
    from tools.hpc_ops import provenance_v2 as pv

    package_root = Path(__file__).resolve().parents[1] / 'cpu_s4_5b_steady_cost'
    repo = Path(__file__).resolve().parents[3]
    package_name = '_filament_cpu_s46a_s45b_scalar'
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = [str(package_root)]
        package.__package__ = package_name
        package.__file__ = str(package_root / '__init__.py')
        sys.modules[package_name] = package

    def binding(path):
        path = Path(path).resolve(strict=True)
        relative = path.relative_to(repo).as_posix()
        raw = path.read_bytes()
        oid = pv._git(repo, 'rev-parse', 'HEAD:' + relative, check=False)
        if not oid:
            raise RuntimeError('GPU_S45B_SCALAR_SOURCE_GIT_BINDING_MISSING:' + relative)
        frozen = pv._read_git_blob(repo, oid)
        if pv.canonical_lf_bytes(frozen) != pv.canonical_lf_bytes(raw):
            raise RuntimeError('GPU_S45B_SCALAR_SOURCE_CANONICAL_BINDING:' + relative)
        return dict(path=relative, hash_scope=pv.TRACKED_HASH_SCOPE,
                    git_blob_oid=oid, raw_sha256=pv.raw_sha256(raw),
                    canonical_lf_sha256=pv.canonical_lf_sha256(raw), size_bytes=len(raw))

    paths = {name: package_root / (name + '.py') for name in ('timing', 'instrument')}
    bindings = {name: binding(path) for name, path in paths.items()}
    modules = {}
    for name in ('timing', 'instrument'):
        full_name = package_name + '.' + name
        existing = sys.modules.get(full_name)
        if existing is not None:
            modules[name] = existing
            continue
        spec = importlib.util.spec_from_file_location(full_name, paths[name])
        if spec is None or spec.loader is None:
            raise RuntimeError('GPU_S45B_SCALAR_PRIVATE_SPEC:' + name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(full_name, None)
            raise
        modules[name] = module
    return modules['instrument'], dict(package=package_name, files=bindings)


def _support_module(name):
    from tools.cpu_s4_checkpoint_support import support_module
    return support_module(name)


def _s46a_fingerprints(ctx):
    """Use the original array identity helper without the CPU benchmark."""
    from KHz_filament.hr4e5_evidence import sha256_array
    return dict(source=sha256_array(ctx['source']),
                PRE={key: sha256_array(value) for key, value in sorted(ctx['pre_fields'].items())})


def _normal_output_writer(intervals):
    """Clone the unchanged G2 output writer with only its interval constant."""
    original = _support_module('g2_hpc_entry')._write_normal_outputs
    namespace = dict(original.__globals__)
    namespace['INTERVALS'] = int(intervals)
    return FunctionType(original.__code__, namespace, original.__name__,
                        original.__defaults__, original.__closure__)


def _preload_host_services():
    """Load the frozen host-only comparison services before GPU science starts."""
    import importlib
    import os

    entry = _support_module('g2_hpc_entry')
    quiet_runtime = entry._quiet_runtime()
    previous_gpu = os.environ.get('UPPE_USE_GPU')
    try:
        # These imports are intentionally host-side service code only.  J1's
        # legacy module sets UPPE_USE_GPU=0 at import time; restore the caller's
        # verified GPU selection before returning to the real run.
        j1 = importlib.import_module('tools.cpu_p1r2.cpu_p1r2_j1')
        q1_t0 = importlib.import_module('tools.cpu_o1.q1_t0')
        exact = _support_module('g1_exact')
        extended = _support_module('g2_hpc_exact')
        writer = _normal_output_writer(8)
        require(callable(getattr(j1, '_canonical_hash', None)), 'GPU_HOST_J1_SERVICE')
        require(callable(getattr(q1_t0, '_numeric_scalar_diagnostics', None)),
                'GPU_HOST_Q1_T0_SERVICE')
        require(callable(getattr(exact, 'fingerprints', None)), 'GPU_HOST_G1_EXACT_SERVICE')
        require(callable(writer), 'GPU_HOST_NORMAL_WRITER_SERVICE')
        return {
            'observer_check': quiet_runtime,
            'modules': {
                'g2_hpc_entry': str(entry.__name__),
                'g2_hpc_exact': str(extended.__name__),
                'g1_exact': str(exact.__name__),
                'cpu_p1r2_j1': str(j1.__name__),
                'cpu_o1_q1_t0': str(q1_t0.__name__),
            },
            'normal_output_writer_loaded': True,
            'up_use_gpu_restored': True,
            'scientific_math_executed': False,
        }
    finally:
        if previous_gpu is None:
            os.environ.pop('UPPE_USE_GPU', None)
        else:
            os.environ['UPPE_USE_GPU'] = previous_gpu


def build(meter, capture_root, qualify):
    from KHz_filament.propagate import propagate_one_pulse as original
    instrument_service, scalar_binding = _load_s45b_instrument_service()
    instrument_propagation_ast = instrument_service.instrument_propagation_ast
    strip_generated_nodes = instrument_service.strip_generated_nodes
    from tools.cpu_s4_checkpoint_support import load_g2_adapter
    from tools.hpc_ops import provenance_v2 as pv
    exact = _support_module('g1_exact')
    source = textwrap.dedent(inspect.getsource(original))
    module = ast.parse(source); baseline = ast.dump(module, include_attributes=False)
    ast.increment_lineno(module,original.__code__.co_firstlineno-1)
    boundary = _support_module('g2_boundary'); adapter = load_g2_adapter()
    bindings = boundary.load_boundary_mapping()['diagnostic_bindings']
    if qualify:
        adapter._insert_boundary(module, bindings)
    before = ast.dump(module, include_attributes=False)
    receipt = instrument_propagation_ast(module.body[0]) if meter.mode != 'REFERENCE' else {'reference_uninstrumented':True}
    receipt['s45b_scalar_source_binding'] = scalar_binding
    require(ast.dump(strip_generated_nodes(copy.deepcopy(module)), include_attributes=False) == before, 'GPU_TIMERS_STRIP_TO_ORIGINAL')
    captured = []
    transfers = []

    def capture(lease, **values):
        if values['interval_index'] != 0:
            return
        require(not captured, 'GPU_ONE_FIRST_BOUNDARY')
        # The same 112 logical fields are selected explicitly at the original
        # boundary. No locals(), frame observer, trace or writable view escapes.
        post = values['thermal_slow_state'].post_fields(0, values['state_after'])
        tree = dict(science={name: values[name] for name in ('E','Qacc','Qacc_raman')}, POST=post,
                    diagnostics={name: item for (name, unused), item in zip(bindings, values['diagnostic_values'])},
                    scalars={name: values[name] for name in boundary.DYNAMIC_SCALARS})
        field_count=sum(len(item) for item in tree.values())
        require(field_count==112 and all(type(value).__module__.startswith('numpy') for value in post.values()),'GPU_ACTUAL_112_HOST_POST_FIELDS')
        write(Path(capture_root).parent/'BOUNDARY_GPU_NATIVE_LAYOUT.json',native_layout(tree,meter.cp))
        host = host_tree(tree, meter.cp, transfers)
        import numpy as np
        host={**host['science'],**{'POST.'+name:value for name,value in host['POST'].items()},
              **{'diagnostics.'+name:np.asarray(value) for name,value in host['diagnostics'].items()},
              **{'scalars.'+name:value for name,value in host['scalars'].items()}}
        require(len(host)==112,'PERSIST_ALL_112_FIELDS_INCLUDING_ORIGINAL_TIMING_EXCEPTIONS')
        path = exact.save_tree(capture_root, host, compressed=True)
        captured.append(dict(path=path, fields=field_count, live_array_refs_retained=0))

    namespace = dict(original.__globals__)
    namespace.update(__cpu_s45b_scope__=meter.scope, __cpu_s45b_marker__=lambda *a, **kw: None,
                     _g2_boundary_capture=capture, _g2_boundary_lease=None)
    calls = {}; intermediate_types=[]
    labels = {'step_linear_bk_nee_factorized':'LINEAR', 'step_linear_full':'LINEAR', 'apply_nonlinear':'NONLINEAR',
              'isaacs_raman_stage':'RAMAN_STAGE', 'apply_isaacs_raman_operator_step':'RAMAN_HALF',
              'evolve_rho_time':'ION_RECURSION', 'make_Wfunc':'ION_SETUP', 'inten_ion':'NONLINEAR_ARRAYS',
              'field_amplitude_from_intensity':'NONLINEAR_ARRAYS', 'ib_alpha':'NONLINEAR_ARRAYS',
              'operator_correct_scalar':'NONLINEAR_ARRAYS', 'kerr_phase_from_deltan':'NONLINEAR_ARRAYS', 'plasma_phase':'NONLINEAR_ARRAYS'}
    for name, label in labels.items():
        function = namespace.get(name)
        if not callable(function):
            continue
        if name=='apply_isaacs_raman_operator_step' and meter.mode in ('LIGHT','SERIALIZED'):
            inner=dict(function.__globals__);inner['isaacs_raman_stage']=namespace['isaacs_raman_stage']
            function=FunctionType(function.__code__,inner,function.__name__,function.__defaults__,function.__closure__)
            function.__kwdefaults__=namespace[name].__kwdefaults__
        def forward(*args, _fn=function, _label=label, _name=name, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            with meter.scope(_label):
                output=_fn(*args, **kwargs)
                if hasattr(output,'dtype'):
                    intermediate_types.append(dict(function=_name,dtype=str(output.dtype),shape=list(output.shape)))
                elif _name=='isaacs_raman_stage' and isinstance(output,dict):
                    intermediate_types.extend(dict(function=_name,field=key,dtype=str(value.dtype),shape=list(value.shape))
                                              for key,value in output.items() if hasattr(value,'dtype'))
                require(len(intermediate_types)<1024,'BOUNDED_GPU_INTERMEDIATE_TYPE_RECORDS')
                return output
        if meter.mode in ('LIGHT','SERIALIZED'):
            namespace[name] = forward
    ast.fix_missing_locations(module)
    exec(compile(module, original.__code__.co_filename, 'exec'), namespace)
    receipt.update(source_canonical_lf_sha256=pv.canonical_lf_sha256(source.encode()), original_ast=baseline,
                   timer_stripping_restores_source=True, capture_fields=112 if qualify else 0,
                   scientific_expression_changes=0, source_function='KHz_filament.propagate.propagate_one_pulse')
    receipt['intermediate_type_records']=intermediate_types
    return namespace['propagate_one_pulse'], receipt, captured, transfers, calls


def run(args, started):
    import numpy as np
    from KHz_filament.device import cp, USE_GPU
    require(USE_GPU and cp is not None, 'GPU_REQUIRED_NO_CPU_FALLBACK')
    host_services = _preload_host_services()
    require(not args.local, 'GPU_REAL_ALLOCATION_REQUIRED')
    identity = verify_source(args.report, False)
    args.work.mkdir(parents=True, exist_ok=False)
    ctx = context(args.bundle)
    from .records import load_json_strict
    from tools.hpc_ops import provenance_v2 as pv
    import os
    input_binding=load_json_strict(args.report/'INPUT_BINDINGS.json')
    require(pv.raw_sha256_file(args.report/'INPUT_BINDINGS.json')==os.environ['S46A_INPUT_BINDINGS_SHA256'],'FROZEN_INPUT_BINDINGS')
    require(input_binding['manifest_raw_sha256'][','.join(map(str,ctx['source'].shape))]==ctx['input_identity']['manifest_raw_sha256']
            and args.intervals==8,'GPU_FORMAL_CANONICAL_INPUT_IDENTITY')
    cp.cuda.runtime.deviceSynchronize()
    meter = Meter(cp, args.mode, args.intervals)
    prop, binding, captured, transfers, calls = build(meter, args.work/'BOUNDARY', args.qualify)
    cases = _support_module('g2_cases')
    exact = _support_module('g1_exact')
    ingress = _s46a_fingerprints(ctx)
    prepare, prepare_binding = prepare_function(cases.prepare, ctx)
    field_host, kwargs, view = prepare(ctx, args.work/'PROPAGATION', intervals=args.intervals)
    field = cp.asarray(field_host)
    transfers.append(dict(direction='H2D', bytes=int(field_host.nbytes), reason='FRESH_CANONICAL_E'))
    transfers.extend(dict(direction='H2D',bytes=int(value.nbytes),reason='CANONICAL_AXIS:'+name) for name,value in ctx['host_axes'].items())
    qualification_readback_s = 0.
    if args.qualify:
        t = time.perf_counter()
        require(np.array_equal(cp.asnumpy(field).view(np.uint8), field_host.view(np.uint8)), 'EXACT_GPU_INPUT_TRANSFER_VALUES')
        for name, value in ctx['host_axes'].items():
            require(np.array_equal(cp.asnumpy(getattr(ctx['axes'],name)).view(np.uint8), value.view(np.uint8)), 'EXACT_GPU_AXIS_TRANSFER:' + name)
        qualification_readback_s = time.perf_counter() - t
    cp.cuda.runtime.deviceSynchronize()
    window_start_ns, cpu_start_ns = time.perf_counter_ns(), time.process_time_ns()
    try:
        result = prop(field, **kwargs)
        cp.cuda.runtime.deviceSynchronize()
        window_end_ns, cpu_end_ns = time.perf_counter_ns(), time.process_time_ns()
        layout = native_layout(result,cp)
        require(bool(result[2]['diagnostic_validation_passed']), 'GPU_ORIGINAL_DIAGNOSTIC_VALIDATION')
        host_result = host_tree(result, cp, transfers)
        bundle = cases.finish(host_result, view, args.work/'PROPAGATION', intervals=args.intervals)
    finally:
        view.close()
    require(_s46a_fingerprints(ctx)==ingress,'GPU_CANONICAL_SOURCE_PRE_IMMUTABLE')
    ordinary = _normal_output_writer(args.intervals)(bundle,args.work/'PROPAGATION',args.work,role='GPU_NATIVE',raw_diagnostics=host_result[2])
    cp.cuda.runtime.deviceSynchronize()
    application_end_ns,application_end_cpu_ns=time.perf_counter_ns(),time.process_time_ns()
    app_s = (application_end_ns-args.application_started_ns)/1e9
    # All subsequent tree serialization/comparisons are research work.
    timing = meter.finish()
    # Persist existing completed endpoints after the application clock has ended.
    # This adds no device synchronization, event, or timestamp acquisition.
    window_clocks = dict(start_ns=window_start_ns,end_ns=window_end_ns,
                         start_cpu_ns=cpu_start_ns,end_cpu_ns=cpu_end_ns)
    window_token = 'completed_propagation_window'
    for row in timing['records']:
        if row['parent'] is None:
            row['parent'] = window_token
    timing['records'] = [dict(token=window_token,parent=None,label='PROPAGATION',
        start_wall_ns=window_start_ns,end_wall_ns=window_end_ns,
        start_cpu_ns=cpu_start_ns,end_cpu_ns=cpu_end_ns,
        wall_ns=window_end_ns-window_start_ns,process_cpu_ns=cpu_end_ns-cpu_start_ns,
        completed=True,failed=False,completion_anchor=True,
        clock_domain='completed host wall; existing deviceSynchronize endpoints',
        device_event_span_ms=None),*timing['records']]
    write(args.work/'RAW_GPU_TIMING.json', timing)
    output = exact.save_tree(args.work/'OUTPUT', bundle, compressed=True)
    comparisons = {}
    extended = _support_module('g2_hpc_exact')
    if args.reference_output:
        comparisons['terminal'] = extended.compare_full_output(args.reference_output,bundle,args.work/'TERMINAL_EXACT.jsonl')
        require(comparisons['terminal']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS', 'GPU_SAME_BACKEND_TERMINAL_EXACT')
        reference_layout = Path(args.reference_output).parent/'GPU_NATIVE_LAYOUT.json'
        require(layout == json.loads(reference_layout.read_text(encoding='utf-8')), 'GPU_SAME_BACKEND_NATIVE_LAYOUT_ALIAS')
    if args.reference_boundary and args.qualify:
        comparisons['boundary'] = exact.compare(args.reference_boundary, read_snapshot(captured[0]['path']), args.work/'BOUNDARY_EXACT.jsonl', disk=True)
        require(comparisons['boundary']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS', 'GPU_SAME_BACKEND_BOUNDARY_EXACT')
        require(json.loads((Path(args.reference_boundary).parent/'BOUNDARY_GPU_NATIVE_LAYOUT.json').read_text(encoding='utf-8'))
                == json.loads((args.work/'BOUNDARY_GPU_NATIVE_LAYOUT.json').read_text(encoding='utf-8')), 'GPU_BOUNDARY_NATIVE_LAYOUT_ALIAS')
    intervals = [row for row in meter.rows if row['label']=='INTERVAL_BODY']
    require(args.mode=='REFERENCE' or (len(intervals)==args.intervals and [row['interval'] for row in intervals]==list(range(args.intervals))), 'GPU_ACTUAL_CONTINUOUS_INTERVALS')
    value = dict(backend='GPU',case=args.case,mode=args.mode,shape=list(ctx['source'].shape),intervals=args.intervals,
        application_service_s=app_s,window_completed_s=(window_end_ns-window_start_ns)/1e9,
        application_process_cpu_s=(application_end_cpu_ns-args.application_started_cpu_ns)/1e9,
        application_clocks=dict(start_ns=args.application_started_ns,start_cpu_ns=args.application_started_cpu_ns,
                                application_end_ns=application_end_ns,application_end_cpu_ns=application_end_cpu_ns),
        input_identity=ctx['input_identity'],
        window_process_cpu_s=(cpu_end_ns-cpu_start_ns)/1e9,source_binding=identity,
        completed_window_clocks=window_clocks,
        first_interval_s=intervals[0]['wall_ns']/1e9 if intervals else None,
        later_body_s=(intervals[-1]['end_wall_ns']-intervals[1]['start_wall_ns'])/1e9 if intervals else None,
        output=output,boundary_path=captured[0]['path'] if captured else None,comparisons=comparisons,
        normal_outputs=ordinary,host_service_preload=host_services,calls=calls,transfers=transfers,qualification_only_readback_s=qualification_readback_s,
        transfer_accounting_scope='Known explicit input and output calls only; internal PRE/thermalization/scalar transfers UNKNOWN without trace',
        memory_pool=dict(used_bytes=cp.get_default_memory_pool().used_bytes(),reserved_bytes=cp.get_default_memory_pool().total_bytes()),
        required_completions='deviceSynchronize before window, interval1 end, final interval end, function return, normal output completion',
        default='I0',production_qualified=False,review_gate='MANUAL_REVIEW_PENDING')
    value['POST_timeline']=post_timeline(args.work/'PROPAGATION',args.application_started_ns)
    value.update(schema='filament.cpu_s46a.case.v1',status='CURRENT_GPU_NATIVE_REFERENCE' if args.mode=='REFERENCE' else 'SAME_BACKEND_TERMINAL_EXACT_RECORDED')
    extended.save_file_index(args.work/'OUTPUT_RAW_INDEX.json',{'OUTPUT':output,'PROPAGATION':args.work/'PROPAGATION',
        'SCIENTIFIC_QACC':ordinary['scientific_Qacc'],'SCIENTIFIC_SCALARS':ordinary['scientific_scalar_diagnostics']})
    value['output_raw_index']=str(args.work/'OUTPUT_RAW_INDEX.json')
    write(args.work/'GPU_SOURCE_BINDING.json',binding);write(args.work/'PREPARE_ACQUISITION_BINDING.json',prepare_binding)
    write(args.work/'GPU_NATIVE_LAYOUT.json',layout)
    write(args.work/'CASE_RECEIPT.json',value)
    return value
