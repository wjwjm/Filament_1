"""GPU manager simulation only: no CUDA or GPU mathematical qualification."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from .gpu import Meter


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
