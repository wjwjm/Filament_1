"""Reuse PV2 preparation and bind the independently selected R0 reference."""
import json
import shutil
from pathlib import Path
from types import FunctionType
from tools.hpc_ops import provenance_v2 as pv
from . import prepare as cost
from .alpha_protocol import BASE, TASK, SCIENCE_TREE, SHAPE, CANDIDATE, candidate_spec, plan

def prepare(report, *, reference_selection, **kwargs):
    namespace = dict(cost.prepare.__globals__, __file__=__file__, BASE=BASE, TASK=TASK,
        SCIENCE_TREE=SCIENCE_TREE, SHAPE=SHAPE, CANDIDATE=CANDIDATE,
        candidate_spec=candidate_spec, plan=plan)
    binding = FunctionType(cost.prepare.__code__, namespace)(report, **kwargs)
    root = Path(report).resolve(strict=True)
    shutil.copyfile(reference_selection, root / "REFERENCE_SELECTION.json")
    binding["reference_selection_raw_sha256"] = pv.raw_sha256_file(root / "REFERENCE_SELECTION.json")
    (root / "EXECUTION.json").write_text(json.dumps(binding, indent=2, allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    return binding
