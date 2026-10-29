"""GPU manager simulation only: no CUDA or GPU mathematical qualification."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

def test_gpu_current_memory_guardian_without_kernel_peak():
    from tools.cpu_o1.cpu_s4.memory_limits import memory_fractions
    unconstrained_root=dict(status='READ',effective_limit_bytes=1000,enforced_cgroup_limits=[])
    assert memory_fractions(rss_bytes=801,pss_bytes=None,cgroup=unconstrained_root,
                            allocation_bytes=1000)['max_enforced_fraction']>.8
    parent_limited=dict(status='READ',effective_limit_bytes=1000,
                        enforced_cgroup_limits=[dict(path='/job',current_bytes=801,limit_bytes=1000)])
    assert memory_fractions(rss_bytes=100,pss_bytes=None,cgroup=parent_limited,
                            allocation_bytes=1000)['max_enforced_fraction']>.8
from .gpu import Meter
from .gpu_resource_exact import ResourceTelemetryError, compare_terminal_with_resource_policy, inspect_candidate_resources
from .gpu_boundary_exact import compare_boundary_with_saved_native_metadata


class Cuda:
    def __init__(self):
        self.syncs=0; self.tick=0; self.runtime=self
    def deviceSynchronize(self):self.syncs+=1
    def get_current_stream(self):return SimpleNamespace(ptr=7)
    def Event(self):
        parent=self
        class Event:
            def record(self):
                parent.tick+=1; self.tick=parent.tick
        return Event()
    def get_elapsed_time(self,start,end):return float(end.tick-start.tick)


def test_s46a_gpu_completion_anchors_and_deferred_event_read():
    for mode in ('OFF','LIGHT'):
        cuda=Cuda(); meter=Meter(SimpleNamespace(cuda=cuda),mode,8)
        for index in range(8):
            with meter.scope('INTERVAL_BODY',index):
                with meter.scope('LINEAR'):
                    pass
        assert cuda.syncs==2
        result=meter.finish()
        assert cuda.syncs==3
        bodies=[row for row in result['records'] if row['label']=='INTERVAL_BODY']
        assert len(bodies)==8 and [row['interval'] for row in bodies]==list(range(8))
        assert all(row['completion_anchor']==(row['interval'] in (0,7)) for row in bodies)
        assert ('device_event_span_ms' in bodies[0])==(mode=='LIGHT')
        assert result['kernel_activity']=='UNKNOWN_WITHOUT_TRACE'


def test_s46a_gpu_original_ast_compiles_without_gpu_math(tmp_path):
    from .gpu import build
    for mode in ('REFERENCE','OFF','LIGHT'):
        meter=Meter(SimpleNamespace(cuda=Cuda()),mode,8)
        function,binding,captures,transfers,calls=build(meter,tmp_path/mode,False)
        assert callable(function) and binding['timer_stripping_restores_source']
        assert binding['scientific_expression_changes']==0 and not captures and not transfers


def test_s46a_gpu_host_loader_blocks_cpu_optional_dependencies_in_new_process(tmp_path):
    """The GPU host seam loads frozen services without CPU optional dependencies."""
    repo = Path(__file__).resolve().parents[3]
    script = r'''
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

class BlockOptional:
    blocked = {"numba", "llvmlite", "psutil"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise ImportError("blocked optional CPU dependency: " + fullname)
        return None

sys.meta_path.insert(0, BlockOptional())
assert os.environ.get("UPPE_USE_GPU") == "1"
from tools.cpu_s4_6a import gpu

class Cuda:
    def __init__(self):
        self.runtime = self
        self.tick = 0
    def get_current_stream(self):
        return SimpleNamespace(ptr=7)
    def Event(self):
        parent = self
        class Event:
            def record(self):
                parent.tick += 1
                self.tick = parent.tick
        return Event()
    def get_elapsed_time(self, start, end):
        return float(end.tick - start.tick)
    def deviceSynchronize(self):
        return None

meter = gpu.Meter(SimpleNamespace(cuda=Cuda()), "REFERENCE", 8)
function, binding, captures, transfers, calls = gpu.build(meter, Path(tempfile.mkdtemp()) / "BOUNDARY", False)
assert callable(function)
assert binding["timer_stripping_restores_source"] is True
assert binding["scientific_expression_changes"] == 0
assert not captures and not transfers
services = gpu._preload_host_services()
assert services["normal_output_writer_loaded"] is True
assert services["scientific_math_executed"] is False
assert services["observer_check"]["trace"] is None
assert services["observer_check"]["profile"] is None
assert os.environ.get("UPPE_USE_GPU") == "1"
cases = gpu._support_module("g2_cases")
exact = gpu._support_module("g1_exact")
entry = gpu._support_module("g2_hpc_entry")
assert callable(cases.prepare)
assert callable(gpu._normal_output_writer(8))
import numpy as np
tiny = np.zeros((2,), dtype=np.float64)
fingerprints = exact.fingerprints({"tiny": tiny})
assert fingerprints["bundle.tiny"]["nbytes"] == int(tiny.nbytes)
source_pre = entry._source_and_pre_fingerprints({
    "source": np.zeros((2,), dtype=np.complex128),
    "pre_fields": {"tiny": tiny},
})
assert source_pre["source"]["shape"] == [2]
q1_t0 = __import__("tools.cpu_o1.q1_t0", fromlist=["_numeric_scalar_diagnostics"])
scalar_payload, scalar_values = q1_t0._numeric_scalar_diagnostics({"tiny": np.float64(1.0)})
assert "tiny" in scalar_payload and "tiny" in scalar_values
assert os.environ.get("UPPE_USE_GPU") == "1"
assert "tools.cpu_s4_3g3_smallgrid" not in sys.modules
assert "tools.cpu_s4_5b_steady_cost" not in sys.modules
assert "_filament_cpu_s46a_s45b_scalar.instrument" in sys.modules
assert "tools.cpu_p1r2.cpu_p1r2_j1" in sys.modules
assert "tools.cpu_o1.q1_t0" in sys.modules
assert not any(name.split(".", 1)[0] in {"numba", "llvmlite", "psutil"} for name in sys.modules)
'''
    env = os.environ.copy()
    env['UPPE_USE_GPU'] = '1'
    env['PYTHONPATH'] = str(repo / 'Filament_python')
    completed = subprocess.run(
        [sys.executable, '-s', '-B', '-c', script],
        cwd=str(repo), env=env, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + '\n' + completed.stderr


def _resource_fixture(*, resource_offset=0, science_delta=0, layout_change=False, alias_break=False):
    from .gpu import _support_module

    shared = np.arange(8, dtype=np.float64).reshape(2, 4) + float(science_delta)
    scientific = np.asfortranarray(shared) if layout_change else shared
    first = scientific.copy() if alias_break else scientific
    allocated = np.arange(16, 24, dtype=np.int64) + int(resource_offset)
    reserved = allocated + np.int64(8)
    return {
        'returned': (
            first,
            np.arange(4, dtype=np.int64),
            {
                'science': scientific,
                'gpu_allocated_step_bytes': allocated,
                'gpu_reserved_step_bytes': reserved,
            },
        ),
        'POST': {'post': np.linspace(0.0, 1.0, 4, dtype=np.float64)},
        'sinks': {'label': 'fixture'},
    }, _support_module('g2_hpc_exact')


def _save_fixture(tmp_path, bundle, name):
    from .gpu import _support_module

    exact = _support_module('g1_exact')
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=False)
    exact.save_tree(root / 'OUTPUT', bundle, compressed=True)
    return root / 'OUTPUT'


def _run_resource_compare(tmp_path, candidate_bundle, *, name='candidate'):
    reference_bundle, extended = _resource_fixture()
    reference = tmp_path / 'reference' / 'OUTPUT'
    if not reference.is_dir():
        reference = _save_fixture(tmp_path, reference_bundle, 'reference')
    candidate_root = _save_fixture(tmp_path, candidate_bundle, name)
    result = compare_terminal_with_resource_policy(
        extended, reference, candidate_bundle,
        candidate_root.parent / 'TERMINAL_EXACT.jsonl',
        device_total_bytes=4096,
    )
    return result, reference, candidate_root


def test_s46a_resource_values_change_science_compare_pass_and_snapshot_unchanged(tmp_path):
    reference_bundle, _ = _resource_fixture()
    reference = _save_fixture(tmp_path, reference_bundle, 'reference')
    frozen_names = ('TREE.json', 'ALIASES.json', 'FINGERPRINTS.json', 'ARRAYS.zip')
    before = {name: (reference / name).read_bytes() for name in frozen_names}

    candidate_bundle, _ = _resource_fixture(resource_offset=3)
    result, _, candidate_output = _run_resource_compare(tmp_path, candidate_bundle)

    assert result['status'] == 'PASS_SCIENCE_VIEW_WITH_APPROVED_RESOURCE_TELEMETRY'
    assert result['scientific_status'] == 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
    assert result['full_output_exact_old_scope'] is False
    sidecar = json.loads(Path(result['resource_telemetry_sidecar']).read_text(encoding='utf-8'))
    assert sidecar['allowed_keys'] == [
        'gpu_allocated_step_bytes', 'gpu_reserved_step_bytes'
    ]
    assert sidecar['vectors']['gpu_allocated_step_bytes']['per_step_delta_candidate_minus_reference'] == [3] * 8
    assert sidecar['vectors']['gpu_reserved_step_bytes']['per_step_delta_candidate_minus_reference'] == [3] * 8
    assert set(sidecar['excluded_paths']) == {
        'bundle.returned.2.gpu_allocated_step_bytes',
        'bundle.returned.2.gpu_reserved_step_bytes',
    }
    assert all((reference / name).read_bytes() == before[name] for name in frozen_names)
    assert set(candidate_bundle['returned'][2]) == {
        'science', 'gpu_allocated_step_bytes', 'gpu_reserved_step_bytes'
    }
    view_tree = json.loads((Path(result['reference_view']) / 'TREE.json').read_text(encoding='utf-8'))
    assert 'gpu_allocated_step_bytes' not in view_tree['items']['returned']['items'][2]['items']
    assert 'gpu_reserved_step_bytes' not in view_tree['items']['returned']['items'][2]['items']
    assert (candidate_output / 'FINGERPRINTS.json').is_file()


@pytest.mark.parametrize(
    'kwargs, pattern',
    [
        ({'science_delta': 1}, 'FIRST_SCIENTIFIC_NONEXACT'),
        ({'layout_change': True}, 'NATIVE_LAYOUT'),
        ({'alias_break': True}, 'ALL_PAIRWISE_ALIAS_EXACT'),
    ],
)
def test_s46a_resource_policy_keeps_science_layout_and_alias_gates(tmp_path, kwargs, pattern):
    candidate_bundle, _ = _resource_fixture(**kwargs)
    with pytest.raises(RuntimeError, match=re.escape(pattern)):
        _run_resource_compare(tmp_path, candidate_bundle, name='candidate_bad')


@pytest.mark.parametrize('kind', ('float', 'shape', 'negative', 'allocated_gt_reserved', 'over_device'))
def test_s46a_resource_telemetry_invalid_values_fail_closed(kind):
    bundle, _ = _resource_fixture()
    diagnostics = bundle['returned'][2]
    if kind == 'float':
        diagnostics['gpu_allocated_step_bytes'] = np.ones(8, dtype=np.float64)
    elif kind == 'shape':
        diagnostics['gpu_allocated_step_bytes'] = np.ones((2, 4), dtype=np.int64)
    elif kind == 'negative':
        diagnostics['gpu_allocated_step_bytes'] = np.array([-1] + [1] * 7, dtype=np.int64)
    elif kind == 'allocated_gt_reserved':
        diagnostics['gpu_allocated_step_bytes'] = np.full(8, 20, dtype=np.int64)
        diagnostics['gpu_reserved_step_bytes'] = np.full(8, 19, dtype=np.int64)
    elif kind == 'over_device':
        diagnostics['gpu_reserved_step_bytes'] = np.full(8, 4097, dtype=np.int64)
    with pytest.raises(ResourceTelemetryError):
        inspect_candidate_resources(bundle, 4096)


def _boundary_fixture(*, science_delta=0, alias_break=False, fortran=True):
    values = np.arange(8, dtype=np.float64).reshape(2, 4) + float(science_delta)
    scientific = np.asfortranarray(values) if fortran else np.ascontiguousarray(values)
    first = scientific.copy(order='F' if fortran else 'C') if alias_break else scientific
    return {
        'returned': (
            first,
            np.arange(4, dtype=np.int64),
            {'science': scientific},
        ),
        'POST': {'post': np.linspace(0.0, 1.0, 4, dtype=np.float64)},
        'sinks': {'label': 'boundary-fixture'},
    }


def _boundary_112_fixture(*, science_delta=0, alias_break=False):
    """Small 112-field boundary with an owned F-order ``E`` field."""
    values = np.arange(8, dtype=np.float64).reshape(2, 4) + float(science_delta)
    native_e = np.asfortranarray(values)
    candidate_e = native_e.copy(order='F') if alias_break else native_e
    bundle = {'E': candidate_e, 'E_alias': native_e}
    for index in range(2, 112):
        bundle[f'field_{index:03d}'] = np.array([index], dtype=np.float64)
    assert len(bundle) == 112
    return bundle


def _read_boundary_snapshot(path):
    from .gpu import read_snapshot

    return read_snapshot(path)


def test_s46a_boundary_saved_native_metadata_adapter_preserves_value_gate_and_snapshot_bytes(tmp_path):
    from .gpu import _support_module

    exact = _support_module('g1_exact')
    reference_bundle = _boundary_112_fixture()
    candidate_bundle = _boundary_112_fixture()
    reference = _save_fixture(tmp_path, reference_bundle, 'boundary_reference')
    candidate = _save_fixture(tmp_path, candidate_bundle, 'boundary_candidate')
    names = ('TREE.json', 'ALIASES.json', 'ARRAYS.zip')
    before_reference = {name: (reference / name).read_bytes() for name in names}
    before_candidate = {name: (candidate / name).read_bytes() for name in names}
    decoded = _read_boundary_snapshot(candidate)

    with pytest.raises(RuntimeError, match='NATIVE_LAYOUT'):
        exact.compare(reference, decoded, tmp_path / 'old' / 'BOUNDARY_EXACT.jsonl', disk=True)

    result = compare_boundary_with_saved_native_metadata(
        reference, candidate, decoded, tmp_path / 'new' / 'BOUNDARY_EXACT.jsonl',
        source_binding={'head': 'a59765c', 'hash_scope': 'tracked_text'},
    )
    assert result['status'] == 'PASS_RECORDED_NATIVE_BOUNDARY_WITH_FROZEN_VALUE_GATE'
    assert result['value_gate_status'] == 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
    sidecar = json.loads(Path(result['readback_layout_provenance']).read_text(encoding='utf-8'))
    assert sidecar['native_metadata_not_filled_from_reference'] is True
    assert sidecar['source_binding_pv2']['head'] == 'a59765c'
    assert len(sidecar['layout_observations']) == 112
    assert sidecar['layout_observations'][0]['path'] == 'bundle.E'
    assert sidecar['layout_observations'][0]['saved_native_layout']['F'] is True
    assert sidecar['layout_observations'][0]['saved_native_layout']['OWNDATA'] is True
    assert sidecar['layout_observations'][0]['decoded_readback_layout']['OWNDATA'] is False
    assert all('saved_native_layout' in item and 'decoded_readback_layout' in item for item in sidecar['layout_observations'])
    assert all((reference / name).read_bytes() == before_reference[name] for name in names)
    assert all((candidate / name).read_bytes() == before_candidate[name] for name in names)


@pytest.mark.parametrize(
    'kwargs, pattern',
    [
        ({'science_delta': 1}, 'FIRST_SCIENTIFIC_NONEXACT'),
        ({'alias_break': True}, 'ALL_PAIRWISE_ALIAS_EXACT'),
    ],
)
def test_s46a_boundary_adapter_keeps_value_and_alias_failures(kwargs, pattern, tmp_path):
    reference = _save_fixture(tmp_path, _boundary_112_fixture(), 'boundary_reference')
    candidate = _save_fixture(tmp_path, _boundary_112_fixture(**kwargs), 'boundary_candidate')
    with pytest.raises(RuntimeError, match=re.escape(pattern)):
        compare_boundary_with_saved_native_metadata(
            reference, candidate, _read_boundary_snapshot(candidate),
            tmp_path / 'failure' / 'BOUNDARY_EXACT.jsonl',
        )


def test_s46a_boundary_adapter_rejects_candidate_saved_layout_change(tmp_path):
    reference = _save_fixture(tmp_path, _boundary_112_fixture(), 'boundary_reference')
    candidate = _save_fixture(tmp_path, _boundary_112_fixture(), 'boundary_candidate')
    tree_path = candidate / 'TREE.json'
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    tree['items']['E']['layout']['C'] = True
    tree_path.write_text(json.dumps(tree, indent=2) + '\n', encoding='utf-8', newline='\n')
    with pytest.raises(RuntimeError, match='NATIVE_LAYOUT'):
        compare_boundary_with_saved_native_metadata(
            reference, candidate, _read_boundary_snapshot(candidate),
            tmp_path / 'layout_failure' / 'BOUNDARY_EXACT.jsonl',
        )
