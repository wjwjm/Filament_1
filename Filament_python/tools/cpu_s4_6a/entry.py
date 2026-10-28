"""One fresh S46A window. Existing L1/G3 and original output/exact code are reused."""
from __future__ import annotations
import argparse
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from types import FunctionType, SimpleNamespace

from .inputs import context, require, write
from .records import load_json_strict
from tools.hpc_ops import provenance_v2 as pv

SCIENCE = '821fec5bdacd832b0347e1f2e84285cb54172e57'


def clone(function, namespace):
    value = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    value.__kwdefaults__ = function.__kwdefaults__
    return value


def prepare_function(original, ctx):
    """Only axes acquisition changes: consume the already sealed axis values."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    ast.increment_lineno(tree,original.__code__.co_firstlineno-1)
    baseline = ast.dump(tree, include_attributes=False)
    changed = 0
    originals=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'axes':
            require(isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'make_axes', 'PREPARE_AXES_ANCHOR')
            originals.append(copy.deepcopy(node.value))
            node.value = ast.parse("ctx['axes']", mode='eval').body
            changed += 1
    require(changed == 1, 'ONE_PREPARE_AXIS_ACQUISITION')
    ast.fix_missing_locations(tree)
    restored=copy.deepcopy(tree)
    for node in ast.walk(restored):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='axes':node.value=originals[0]
    require(ast.dump(restored,include_attributes=False)==baseline,'PREPARE_STRIP_PROOF')
    namespace = dict(original.__globals__)
    exec(compile(tree, original.__code__.co_filename, 'exec'), namespace)
    return namespace['prepare'], dict(kind='ACQUISITION_ONLY', original_ast=baseline,
                                     change="axes=make_axes(...) -> axes=ctx['axes']", mathematical_expressions_changed=0)


def verify_source(report, local):
    repo = Path(__file__).resolve().parents[3]
    require(sys.flags.optimize == 0, 'OPTIMIZE_FORBIDDEN')
    require(pv._git(repo, 'rev-parse', 'HEAD:Filament_python/KHz_filament') == SCIENCE, 'UNCHANGED_SCIENCE_TREE')
    if local:
        require(not os.environ.get('SLURM_JOB_ID'), 'LOCAL_NO_SLURM')
        return dict(head=pv._git(repo, 'rev-parse', 'HEAD'), local=True)
    plan = load_json_strict(report / 'MEASUREMENT_PLAN.json')
    require(pv.raw_sha256_file(report / 'MEASUREMENT_PLAN.json') == os.environ.get('S46A_PLAN_SHA256'), 'PRE_DATA_PLAN_RAW_BINDING')
    require(pv._git(repo, 'rev-parse', 'HEAD') == plan['execution_commit'], 'ACTUAL_EXECUTION_COMMIT')
    pv._ensure_clean(repo)
    source = load_json_strict(report / 'EXECUTION_SOURCE_MANIFEST.json')
    for row in source['records']:
        require(row['hash_scope'] == pv.TRACKED_HASH_SCOPE, 'TRACKED_SCOPE')
        require(pv._git(repo, 'rev-parse', 'HEAD:' + row['path']) == row['git_blob_oid']
                and pv.canonical_lf_sha256((repo / row['path']).read_bytes()) == row['canonical_lf_sha256'], 'EXECUTION_SOURCE:' + row['path'])
    node = load_json_strict(report / 'NODE_GATE.json')
    require(node['job_id'] == os.environ.get('SLURM_JOB_ID') and node['status'] == 'PASS', 'ACTUAL_NODE_GATE')
    return dict(head=plan['execution_commit'], plan_sha256=pv.raw_sha256_file(report / 'MEASUREMENT_PLAN.json'))


def cpu_run(args, started):
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    from tools.cpu_s4_5b_steady_cost import candidate as measured
    from tools.cpu_s4_5a_l1_residual import measurement
    from tools.cpu_s4_4a_linear_cache import candidate as accepted
    from tools.cpu_s4_checkpoint_support import support_module
    from KHz_filament.device import USE_GPU
    require(not USE_GPU, 'CPU_REQUIRES_NUMPY')
    support_module('g2_hpc_entry')._quiet_runtime()
    profile = args.mode == 'LIGHT'
    pair = measured.make_recorder(profile)
    args._recorders=pair
    timer = pair.b
    app_token = timer.begin('APPLICATION', common=True, context=timer.context(0, occurrence_class='WINDOW_ONCE'))
    identity = verify_source(args.report, args.local)
    args.work.mkdir(parents=True, exist_ok=False)
    with timer.scope('INPUT_PREPARATION'):
        ctx = context(args.bundle)
    if not args.local:
        require(args.intervals==8,'FORMAL_EIGHT_ONLY')
        binding=load_json_strict(args.report/'INPUT_BINDINGS.json')
        require(pv.raw_sha256_file(args.report/'INPUT_BINDINGS.json')==os.environ['S46A_INPUT_BINDINGS_SHA256'],'FROZEN_INPUT_BINDINGS')
        require(binding['manifest_raw_sha256'][','.join(map(str,ctx['source'].shape))]==ctx['input_identity']['manifest_raw_sha256'],'FORMAL_CANONICAL_INPUT_IDENTITY')
    shape = tuple(ctx['source'].shape)
    if args.local:
        require(shape in ((32, 17, 19), (384, 17, 19)), 'LOCAL_MICRO_ONLY')
    mode = 'QUALIFY_BOUNDARY' if args.qualify else 'PURE_OFF_SMALLGRID'
    if args.mode == 'REFERENCE':
        built = accepted.build_role(mode, capture_root=args.work / 'BOUNDARY', local_engineering=args.local, pool_override=1 if args.local else None)
    else:
        built = measured.build_role(mode, capture_root=args.work / 'BOUNDARY', local_engineering=args.local,
                                    pool_override=1 if args.local else None, profile=profile, recorder_pair=pair)
    if shape == (384, 351, 301):
        from tools.cpu_s4_4b_fullsize_linear_cache.candidate import _replace_empty_cache
        _replace_empty_cache(built[1], local_engineering=False, configured_shape=shape)
    cases, exact = sg.frozen_tools(args.report)
    prepare, prepare_binding = prepare_function(cases.prepare, ctx)
    if args.local and args.mode=='REFERENCE':
        # Micro qualification compares against the accepted acquisition path,
        # including its original axis construction, before replacing acquisition.
        import numpy as np
        from KHz_filament.grids import make_axes
        native_ctx=sg.context(args.report,shape=shape)
        native_ctx['prop'].diag_bk_nee_profile=False
        require(sg.fingerprints(native_ctx)==sg.fingerprints(ctx),'ORIGINAL_GENERATOR_CANONICAL_MICRO_VALUES')
        original_axes=make_axes(ctx['grid'].Nx,ctx['grid'].Ny,ctx['grid'].Nt,ctx['grid'].Lx,ctx['grid'].Ly,ctx['grid'].Twin)
        for name,value in ctx['host_axes'].items():
            require(np.array_equal(getattr(original_axes,name).view(np.uint8),value.view(np.uint8)),'ORIGINAL_CPU_AXIS_BYTES:'+name)
        ctx={**native_ctx,**{key:ctx[key] for key in ('host_axes','axes','input_manifest','input_identity','bundle')}}
        prepare=cases.prepare
        prepare_binding.update(reference='ACCEPTED_ORIGINAL_PREPARE_NO_AXIS_REPLACEMENT')
    namespace = dict(vars(sg))
    namespace.update(context=lambda report, shape=None: ctx, SHAPE=shape,
                     frozen_tools=lambda report: (SimpleNamespace(prepare=prepare, finish=cases.finish), exact))
    clocks={}
    def writer_factory(report,intervals):
        original=sg.normal_writer(report,intervals)
        def writer(*values,**named):
            result=original(*values,**named)
            clocks.update(application_end_ns=time.perf_counter_ns(),application_end_cpu_ns=time.process_time_ns())
            return result
        return writer
    namespace['normal_writer']=writer_factory
    manifest = dict(shape=list(shape), fingerprints=sg.fingerprints(ctx))
    write(args.work / 'LEGACY_HARNESS_INPUT_VIEW.json', manifest)
    namespace['_run_window'] = clone(sg._run_window, namespace)
    private_sg = SimpleNamespace(**namespace)
    request = SimpleNamespace(report=args.report, work=args.work, role='G3_SMALL', reported_role='CPU_L1',
        mode=mode, intervals=args.intervals, local_engineering=args.local, input_manifest=args.work / 'LEGACY_HARNESS_INPUT_VIEW.json',
        reference_output=args.reference_output, reference_boundary=None, reference_stages=args.reference_stages)
    request.execution_domain='FULLSIZE_CONFIRMATION' if shape==(384,351,301) else 'SMALLGRID_BENCHMARK'
    if args.mode == 'REFERENCE':
        result = clone(sg.run_window, namespace)(request, built, label='formal', intervals=args.intervals, capture=args.qualify, application_started=started)
        timer.end(app_token)
    else:
        measured_namespace = dict(vars(measurement)); measured_namespace['sg'] = private_sg
        result = clone(measurement.run_window, measured_namespace)(request, built, timer=timer, application_token=app_token,
            label='formal', intervals=args.intervals, capture=args.qualify, application_started=started)
        write(args.work / 'INSTRUMENTATION_BINDING_AFTER.json', measured.verify_bindings(built[1]))
    boundary = args.work / 'formal/BOUNDARY_RECEIPT.json'
    if args.qualify:
        require(built[5].transparency()['retired'] and result['boundary']['retained_live_array_refs'] == 0, 'BOUNDARY_RETIREMENT')
        extended = support_module('g2_hpc_exact')
        metadata = extended.inspect_boundary_metadata(boundary, require_members=True, header_only=True)
        require(metadata['status'] == 'PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY', 'BOUNDARY_JSON_READBACK')
        if args.reference_boundary:
            result['comparisons']['boundary'] = extended.compare_current_boundary_pair(args.reference_boundary, boundary, args.work / 'BOUNDARY_EXACT.jsonl')
            require(result['comparisons']['boundary']['status'] == 'PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT', 'FIRST_BOUNDARY_EXACT')
    snapshot = pair.snapshot()
    write(args.work / 'RAW_CPU_TIMING.json', snapshot)
    write(args.work / 'PREPARE_ACQUISITION_BINDING.json', prepare_binding)
    result.update(backend='CPU', case=args.case, mode=args.mode, source_binding=identity,input_identity=ctx['input_identity'],
                  legacy_application_service_s=result['optical_application_window_s'],
                  application_service_s=(clocks['application_end_ns']-args.application_started_ns)/1e9,
                  application_process_cpu_s=(clocks['application_end_cpu_ns']-args.application_started_cpu_ns)/1e9,
                  application_clocks=dict(start_ns=args.application_started_ns,start_cpu_ns=args.application_started_cpu_ns,**clocks),
                  clock_scope='APPLICATION_INCLUDES_REQUIRED_INPUT_AND_OUTPUT; research/exact after original application boundary separate',
                  boundary_path=str(boundary) if args.qualify else None, fullsize_cache_bytes=1610612736 if shape == (384,351,301) else 268435456)
    result['POST_timeline']=post_timeline(result['normal_outputs']['propagation_root'],args.application_started_ns)
    result.update(schema='filament.cpu_s46a.case.v1',legacy_harness_status=result['status'],
                  status='CURRENT_ACCEPTED_L1_REFERENCE' if args.mode=='REFERENCE' else 'SAME_BACKEND_TERMINAL_EXACT_RECORDED')
    write(args.work / 'CASE_RECEIPT.json', result)
    return result


def post_timeline(root,application_started_ns):
    manifest=load_json_strict(Path(root)/'state/streaming_manifest.json')
    events=[row for row in manifest['rate_events'] if row['event']=='POST_COMMITTED']
    values=[row['time_s'] for row in events]
    require(values and all(b>a for a,b in zip(values,values[1:])),'NATIVE_POST_EVENT_ORDER')
    return dict(events=events,first_POST_from_application_s=values[0]-application_started_ns/1e9,
                adjacent_POST_gaps_s=[b-a for a,b in zip(values,values[1:])],clock='Original perf_counter seconds',
                definition='Durably committed host POST source before second linear half; final optical field completes later',
                GPU_completion_basis='Original StreamingPulseReadView.post_fields returns detached NumPy payload; StreamingLifecycle commits and records event after payload materialization')


def main():
    started_ns,started_cpu_ns=time.perf_counter_ns(),time.process_time_ns()
    started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'work', 'bundle'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--backend', choices=('CPU', 'GPU'), required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--mode', choices=('REFERENCE', 'OFF', 'LIGHT', 'SERIALIZED'), required=True)
    p.add_argument('--intervals', type=int, choices=(2,8), default=8)
    p.add_argument('--local', action='store_true'); p.add_argument('--qualify', action='store_true')
    for name in ('reference-output', 'reference-boundary', 'reference-stages'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args()
    args.application_started_ns,args.application_started_cpu_ns=started_ns,started_cpu_ns
    require(args.mode=='REFERENCE' or args.reference_output is not None,'SAME_BACKEND_REFERENCE_REQUIRED')
    require(not args.qualify or args.mode=='REFERENCE' or args.reference_boundary is not None,'QUALIFICATION_BOUNDARY_REFERENCE_REQUIRED')
    require(not (args.backend=='CPU' and args.mode=='SERIALIZED'),'CPU_SERIALIZED_MODE_NOT_DEFINED')
    try:
        if args.backend == 'CPU':value = cpu_run(args, started)
        else:
            from .gpu import run
            value = run(args, started)
    except BaseException as error:
        if hasattr(args,'_recorders'):
            for recorder in (args._recorders.s45a,args._recorders.b):
                try:
                    while recorder._stack:recorder.end(recorder._stack[-1].token,failed=True)
                except BaseException as cleanup_error:recorder.retire_incomplete(type(cleanup_error).__name__)
        if args.work.is_dir():
            write(args.work/'FAILED_CASE.json',dict(error=repr(error),scientific_fallback=False,case=args.case))
            if hasattr(args,'_recorders'):write(args.work/'INCOMPLETE_TIMING.json',args._recorders.snapshot())
        raise
    print(json.dumps(dict(case=args.case, backend=args.backend, application_service_s=value['application_service_s'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
