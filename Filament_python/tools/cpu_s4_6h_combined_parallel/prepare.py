"""Reuse the PV2 full-size packager with a distinct combined task and source set."""
from pathlib import Path
from types import FunctionType

from tools.cpu_s4_6f_fullsize_parallel import prepare as parent
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from .protocol import BASE, CANDIDATE, SCIENCE_TREE, SHAPE, TASK, candidate_spec, plan


def _candidate_paths(repo):
    names = parent._candidate_paths(repo)
    # Include actual component pools and shared worker implementation, not just
    # their high-level wrappers. Existing inherited closure remains preserved.
    for package in ("cpu_s4_6c_array_coverage", "cpu_s4_6d_eq10_propagation",
                    "cpu_s4_6e_array_propagation", "cpu_s4_6h_combined_parallel"):
        names.update(path.relative_to(repo).as_posix()
                     for path in (repo / "Filament_python/tools" / package).glob("*.py"))
    return names


def prepare(report, **kwargs):
    namespace = dict(parent.prepare.__globals__)
    namespace.update(__file__=__file__, BASE=BASE, TASK=TASK, SCIENCE_TREE=SCIENCE_TREE,
                     SHAPE=SHAPE, candidate_spec=candidate_spec, plan=plan,
                     _candidate_paths=_candidate_paths)
    # The serialized 6F execution/input schemas and strict PV2 validator are
    # reused unchanged; task, candidate, base and PLAN identify this new study.
    function = FunctionType(parent.prepare.__code__, namespace)
    function.__kwdefaults__ = parent.prepare.__kwdefaults__
    binding = function(report, candidate=CANDIDATE, **kwargs)
    # The reused packager's legacy top-level deadline is 7080 seconds.
    # This new four-window study has an explicit 3480-second deadline.
    root = Path(report).resolve(strict=True)
    execution = parent.read(root / "HPC_EXECUTION_PLAN.json")
    execution["work_deadline_s"] = plan()["resources"]["work_deadline_s"]
    sg.write(root / "HPC_EXECUTION_PLAN.json", execution)
    binding["hpc_plan_raw_sha256"] = pv.raw_sha256_file(root / "HPC_EXECUTION_PLAN.json")
    sg.write(root / "EXECUTION.json", binding)
    return binding
