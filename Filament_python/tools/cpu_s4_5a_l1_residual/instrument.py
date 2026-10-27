"""Private source-bound timing seam for the CPU-S4-5A L1 profile.

This module only wraps the already constructed private 4A Raman state.  The
wrappers carry scalar timing metadata and never receive or retain arrays beyond
the original call.  Guard functions, the public propagator, and the numerical
operators remain owned by the accepted 4A/G3 implementation.
"""

from __future__ import annotations

import ast
import copy
import functools
import dis
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

from tools.hpc_ops import provenance_v2 as pv

from .timing import TimingContext, TimingRecorder, context


INSTRUMENT_SCHEMA = "filament.cpu_s4_5a.l1_instrument.v1"
_RAMAN_REQUIRED = (
    "raman_convolve_intensity",
    "make_raman_kernel",
    "stable_field_fluence_and_loss",
    "_s43c_product",
    "_s43c_normalized_derivative",
    "_s43c_rhs",
    "_s43c_predictor",
    "_s43c_corrector",
)


class InstrumentationBindingError(RuntimeError):
    """The private state cannot be instrumented without widening scope."""


@functools.lru_cache(maxsize=1)
def _structural_hasher():
    from tools.cpu_s4_checkpoint_support import load_g2_adapter
    return load_g2_adapter()._code_structure_sha256


def _code_identity(function: Any) -> str:
    if isinstance(function, functools.partial):
        function = function.func
    code = getattr(function, "__code__", None)
    if code is None:
        return ""
    # Reuse the frozen adapter's recursive code representation.  repr(code)
    # contains an address and cannot establish reconstructed-code identity.
    return _structural_hasher()(code)


def _clone_function(function: Callable[..., Any], namespace: dict[str, Any], name: str) -> Callable[..., Any]:
    code = getattr(function, "__code__", None)
    if code is None:
        raise InstrumentationBindingError(f"INSTRUMENT_FUNCTION_CODE_MISSING:{name}")
    clone = FunctionType(code, namespace, name, getattr(function, "__defaults__", None), getattr(function, "__closure__", None))
    clone.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
    clone.__annotations__ = getattr(function, "__annotations__", {})
    clone.__module__ = getattr(function, "__module__", "")
    clone.__qualname__ = getattr(function, "__qualname__", name)
    return clone


def _canonical_raman_binding() -> dict[str, Any]:
    """Prove that the frozen Raman source still exposes required anchors."""

    repo = Path(__file__).resolve().parents[3]
    path = (repo / "Filament_python/KHz_filament/raman.py").resolve(strict=True)
    raw = path.read_bytes()
    tree = ast.parse(pv.canonical_lf_bytes(raw).decode("utf-8"), filename=str(path))
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {"raman_convolve_intensity", "make_raman_kernel", "stable_field_fluence_and_loss", "isaacs_raman_stage"}
    if not required.issubset(names):
        raise InstrumentationBindingError("CANONICAL_RAMAN_ANCHORS_DRIFTED")
    return {
        "path": str(path),
        "classification": "tracked_text",
        "hash_scope": pv.TRACKED_HASH_SCOPE,
        "git_blob_oid": pv._git(repo, "rev-parse", "HEAD:Filament_python/KHz_filament/raman.py"),
        "checkout_raw_sha256": pv.raw_sha256(raw),
        "canonical_lf_sha256": pv.canonical_lf_sha256(raw),
        "required_function_anchors": sorted(required),
    }


def _canonical_tree() -> tuple[Path, ast.Module]:
    """Load the repository Raman AST using the canonical text hash path."""

    record = _canonical_raman_binding()
    path = Path(str(record["path"])).resolve(strict=True)
    tree = ast.parse(
        pv.canonical_lf_bytes(path.read_bytes()).decode("utf-8"),
        filename=str(path),
    )
    return path, tree


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise InstrumentationBindingError(
            f"CANONICAL_FUNCTION_ANCHOR_AMBIGUOUS:{name}:{len(matches)}"
        )
    return matches[0]


def _compile_function_node(
    node: ast.FunctionDef,
    *,
    namespace: dict[str, Any],
    source_path: Path,
) -> Callable[..., Any]:
    """Compile one function with the private namespace as its globals."""

    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    local_namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace, local_namespace)
    function = local_namespace.get(node.name)
    if not callable(function):
        raise InstrumentationBindingError(f"AST_FUNCTION_NOT_CREATED:{node.name}")
    return function


def _canonical_compiled_function(
    name: str,
    *,
    namespace: Mapping[str, Any],
    source_path: Path,
    tree: ast.Module,
) -> Callable[..., Any]:
    """Compile the complete canonical module for direct-function identity."""

    module = copy.deepcopy(tree)
    ast.fix_missing_locations(module)
    module_namespace = dict(namespace)
    local_namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), module_namespace, local_namespace)
    function = local_namespace.get(name) or module_namespace.get(name)
    if not callable(function):
        raise InstrumentationBindingError(f"CANONICAL_FUNCTION_NOT_CREATED:{name}")
    return function


def _copy_function_metadata(
    function: Callable[..., Any],
    original: Callable[..., Any],
) -> None:
    """Keep the frozen function surface while changing only private code."""

    function.__defaults__ = getattr(original, "__defaults__", None)
    function.__kwdefaults__ = getattr(original, "__kwdefaults__", None)
    function.__annotations__ = dict(getattr(original, "__annotations__", {}) or {})
    function.__doc__ = getattr(original, "__doc__", None)
    function.__module__ = getattr(original, "__module__", "")
    function.__qualname__ = getattr(original, "__qualname__", function.__name__)


def _wrap_statement(statement: ast.stmt, label: str) -> ast.With:
    """Wrap one existing statement with one fixed scalar timing scope."""

    context_call = ast.Call(
        func=ast.Name(id="__cpu_s45a_scope__", ctx=ast.Load()),
        args=[ast.Constant(value=str(label))],
        keywords=[],
    )
    result = ast.With(
        items=[ast.withitem(context_expr=context_call, optional_vars=None)],
        body=[statement],
        type_comment=None,
    )
    result._cpu_s45a_timing_wrapper = True
    ast.copy_location(result, statement)
    return result


def _wrap_stage_sites(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Wrap only the frozen stage intensity and Eq.10 energy statements."""

    intensity_nodes = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "intensity"
    ]
    energy_nodes = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "return_energy"
    ]
    if len(intensity_nodes) != 1 or len(energy_nodes) != 1:
        raise InstrumentationBindingError(
            "RAMAN_STAGE_TIMING_ANCHORS_CHANGED"
        )
    targets = {
        id(intensity_nodes[0]): "RAMAN_INTENSITY",
        id(energy_nodes[0]): "RAMAN_ENERGY",
    }
    locations: list[dict[str, Any]] = []

    def rewrite(node: ast.AST) -> None:
        for field, value in list(ast.iter_fields(node)):
            if not isinstance(value, list):
                if isinstance(value, ast.AST):
                    rewrite(value)
                continue
            rewritten: list[Any] = []
            for item in value:
                if isinstance(item, ast.stmt) and id(item) in targets:
                    label = targets[id(item)]
                    rewritten.append(_wrap_statement(item, label))
                    locations.append(
                        {
                            "label": label,
                            "line": int(getattr(item, "lineno", 0)),
                            "column": int(getattr(item, "col_offset", 0)),
                        }
                    )
                else:
                    if isinstance(item, ast.AST):
                        rewrite(item)
                    rewritten.append(item)
            setattr(node, field, rewritten)

    rewrite(function_node)
    return {"sites": locations}


def _wrap_iir_site(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Time complete existing loops; separate the two actual PWL coefficient spans."""
    locations = []
    branches=[node for node in ast.walk(function_node) if isinstance(node,ast.If)
              and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name)
              and node.test.left.id=='method' and len(node.test.comparators)==1
              and isinstance(node.test.comparators[0],ast.Constant) and node.test.comparators[0].value=='iir']
    if len(branches)!=1:raise InstrumentationBindingError("UNIQUE_FROZEN_IIR_BRANCH_REQUIRED")
    branch=branches[0]
    class Loops(ast.NodeTransformer):
        def visit_For(self, node):
            if not isinstance(node.target, ast.Name) or node.target.id != "n":
                raise InstrumentationBindingError("UNEXPECTED_IIR_LOOP_TARGET")
            locations.append(dict(label="RAMAN_IIR",line=node.lineno,column=node.col_offset))
            return _wrap_statement(node, "RAMAN_IIR")
    Loops().visit(branch)
    coefficient_groups = 0
    def assignment_names(node):
        if not isinstance(node, ast.Assign):return ()
        return tuple(child.id for target in node.targets for child in ast.walk(target) if isinstance(child, ast.Name))
    def coefficients(node):
        nonlocal coefficient_groups
        for field, value in list(ast.iter_fields(node)):
            if isinstance(value,list) and value and all(isinstance(item,ast.stmt) for item in value):
                starts=[i for i,item in enumerate(value) if assignment_names(item) in (("omega","Gamma"),("c1",))]
                if starts:
                    if len(starts)!=1:raise InstrumentationBindingError("IIR_COEFFICIENT_GROUP_AMBIGUOUS")
                    start=starts[0];first=assignment_names(value[start]);last=("k",) if first==("omega","Gamma") else ("c0",)
                    ends=[i for i in range(start,len(value)) if assignment_names(value[i])==last]
                    if len(ends)!=1:raise InstrumentationBindingError("IIR_COEFFICIENT_END_CHANGED")
                    end=ends[0];span=value[start:end+1]
                    wrapped=_wrap_statement(span[0],"RAMAN_COEFFICIENTS");wrapped.body=span
                    locations.append(dict(label="RAMAN_COEFFICIENTS",line=span[0].lineno,last_line=span[-1].end_lineno,column=span[0].col_offset))
                    setattr(node,field,value[:start]+[wrapped]+value[end+1:]);coefficient_groups+=1
                    for item in value[:start]+value[end+1:]:coefficients(item)
                else:
                    for item in value:coefficients(item)
            elif isinstance(value,ast.AST):coefficients(value)
    coefficients(branch)
    if coefficient_groups!=2 or sum(row['label']=='RAMAN_IIR' for row in locations)!=5:
        raise InstrumentationBindingError("FROZEN_IIR_COEFFICIENT_LOOP_COUNT_CHANGED")
    return dict(sites=locations,coefficient_groups=coefficient_groups,loop_events="one entire existing time loop per response; never per time sample")


def _strip_timing_nodes(function_node: ast.FunctionDef) -> ast.FunctionDef:
    """Remove only generated timing ``with`` nodes for AST equality checks."""

    result = copy.deepcopy(function_node)

    class Stripper(ast.NodeTransformer):
        def generic_visit(self, node: ast.AST) -> ast.AST:
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    restored: list[Any] = []
                    for item in value:
                        if getattr(item, "_cpu_s45a_timing_wrapper", False):
                            restored.extend(self.visit(child) for child in item.body)
                        else:
                            restored.append(self.visit(item) if isinstance(item, ast.AST) else item)
                    setattr(node, field, restored)
                elif isinstance(value, ast.AST):
                    setattr(node, field, self.visit(value))
            return node

    return Stripper().visit(result)


def _expected_transformed_node(
    *,
    name: str,
    transform_name: str,
    candidate_namespace: Mapping[str, Any],
) -> tuple[Path, ast.FunctionDef, Callable[..., Any]]:
    """Rebuild one R8 AST transformation without accepting a guessed layout."""

    transform = candidate_namespace.get(transform_name)
    if not callable(transform):
        raise InstrumentationBindingError(f"R8_TRANSFORM_MISSING:{transform_name}")
    source_path, tree = _canonical_tree()
    node = copy.deepcopy(_function_node(tree, name))
    transform(node)
    expected = _compile_function_node(node, namespace=dict(candidate_namespace), source_path=source_path)
    return source_path, node, expected


def _verify_uninstrumented_r8_pair(
    state: Any,
    original_stage: Callable[..., Any],
    original_apply: Callable[..., Any],
) -> dict[str, Any]:
    """Prove the actual private stage/apply code is the frozen R8 transform."""

    candidate_namespace = getattr(state, "_candidate_namespace", None)
    if not isinstance(candidate_namespace, dict):
        raise InstrumentationBindingError("R8_CANDIDATE_NAMESPACE_MISSING")
    stage_path, stage_node, expected_stage = _expected_transformed_node(
        name="isaacs_raman_stage",
        transform_name="_transform_stage",
        candidate_namespace=candidate_namespace,
    )
    apply_path, apply_node, expected_apply = _expected_transformed_node(
        name="apply_isaacs_raman_operator_step",
        transform_name="_transform_apply",
        candidate_namespace=candidate_namespace,
    )
    if _code_identity(expected_stage) != _code_identity(original_stage):
        raise InstrumentationBindingError("R8_STAGE_STRUCTURAL_FINGERPRINT_MISMATCH")
    if _code_identity(expected_apply) != _code_identity(original_apply):
        raise InstrumentationBindingError("R8_APPLY_STRUCTURAL_FINGERPRINT_MISMATCH")
    if getattr(expected_stage, "__defaults__", None) != getattr(original_stage, "__defaults__", None):
        raise InstrumentationBindingError("R8_STAGE_DEFAULTS_CHANGED")
    if getattr(expected_apply, "__defaults__", None) != getattr(original_apply, "__defaults__", None):
        raise InstrumentationBindingError("R8_APPLY_DEFAULTS_CHANGED")
    return {
        "stage_source_path": str(stage_path),
        "apply_source_path": str(apply_path),
        "stage_uninstrumented_code_identity": _code_identity(original_stage),
        "apply_uninstrumented_code_identity": _code_identity(original_apply),
        "stage_ast_identity": pv.raw_sha256(
            ast.dump(stage_node, include_attributes=False).encode("utf-8")
        ),
        "apply_ast_identity": pv.raw_sha256(
            ast.dump(apply_node, include_attributes=False).encode("utf-8")
        ),
        "r8_transform_names": ["_transform_stage", "_transform_apply"],
    }


def _metadata(state: Any, *, stage: str = "none") -> TimingContext:
    local_phase = str(getattr(getattr(state, "_local", None), "g_phase", "none") or "none")
    phase = local_phase if local_phase in ("none", "pre", "post") else "none"
    scope = str(getattr(state, "_g_active_cap_scope", "none") or "none")
    if scope in ("pre", "post"):
        phase = scope
    if stage not in ("none", "stage1", "stage2"):
        stage = "none"
    if stage == "none" and local_phase in ("stage1", "stage2"):
        stage = local_phase
    try:
        interval = int(getattr(state, "_g_segment_index", 0))
    except (TypeError, ValueError, OverflowError):
        raise InstrumentationBindingError("TIMING_INTERVAL_NOT_SCALAR") from None
    if interval < 0:
        raise InstrumentationBindingError("TIMING_INTERVAL_NEGATIVE")
    return context(interval, phase, stage)


def _timed_wrapper(
    original: Callable[..., Any],
    recorder: TimingRecorder,
    state: Any,
    label: str,
    *,
    stage: str = "none",
    nested: tuple[str, ...] = (),
) -> Callable[..., Any]:
    """Wrap one fixed helper with scalar scopes and preserve its call surface."""

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        metadata = _metadata(state, stage=stage)
        with recorder.scope(label, context=metadata):
            if not nested:
                return original(*args, **kwargs)
            # The nested labels are fixed source-level categories, never
            # caller supplied callbacks.  Each nested scope surrounds the same
            # original call and therefore remains a transparent observation.
            manager = recorder.scope(nested[0], context=metadata)
            with manager:
                return original(*args, **kwargs)

    setattr(wrapped, "__cpu_s45a_instrumented__", True)
    setattr(wrapped, "__cpu_s45a_original__", original)
    setattr(wrapped, "__cpu_s45a_original_code_identity__", _code_identity(original))
    setattr(wrapped, "__cpu_s45a_label__", label)
    setattr(wrapped, "__cpu_s45a_nested_labels__", tuple(nested))
    return wrapped


def install_propagation_wrappers(propagator: Callable[..., Any], state: Any) -> None:
    """Wrap only non-guard helpers in the existing private G clone globals."""
    if not state._cpu_s45a_profile:return
    namespace=propagator.__globals__
    labels={
        "inten_ion":"NONLINEAR_ARRAYS", "field_amplitude_from_intensity":"NONLINEAR_ARRAYS",
        "make_Wfunc":"ION_SETUP", "evolve_rho_time":"ION_RECURSION", "ib_alpha":"NONLINEAR_ARRAYS",
        "operator_correct_scalar":"NONLINEAR_ARRAYS", "kerr_phase_from_deltan":"NONLINEAR_ARRAYS", "plasma_phase":"NONLINEAR_ARRAYS",
    }
    originals={}
    for name,label in labels.items():
        original=namespace.get(name)
        if not callable(original):raise InstrumentationBindingError("PROPAGATION_HELPER_MISSING:"+name)
        originals[name]=original
        namespace[name]=_timed_wrapper(original,state._cpu_s45a_recorder,state,label)
    state._cpu_s45a_propagation_originals=originals
    state._cpu_s45a_bindings["propagation_helpers"]={name:dict(label=label,original_code_identity=_code_identity(originals[name]),wrapper_code_identity=_code_identity(namespace[name])) for name,label in labels.items()}


def bind_active_propagation(wrapped: Callable[..., Any], state: Any) -> None:
    """Bind the actual G2-produced clone after its unchanged guard is installed."""
    if not state._cpu_s45a_profile:return
    active=getattr(wrapped,"__g2_inner__",None)
    if not callable(active):raise InstrumentationBindingError("ACTUAL_G2_INNER_REQUIRED")
    state._cpu_s45a_active_propagation_namespace=active.__globals__
    state._cpu_s45a_active_propagation_function=active
    state._cpu_s45a_active_propagation_code=active.__code__
    seal_fixed_callables(state)


def seal_fixed_callables(state: Any) -> None:
    """Register the fixed inserted function/code/closure slots, without call values."""
    namespace=state._cpu_s45a_private_namespace
    functions={name:namespace[name] for name in _RAMAN_REQUIRED}
    functions['__cpu_s45a_scope__']=namespace['__cpu_s45a_scope__']
    active=state._cpu_s45a_active_propagation_namespace
    functions.update({'propagation.'+name:active[name] for name in state._cpu_s45a_propagation_originals})
    state._cpu_s45a_callable_slots={name:(function,function.__code__,function.__globals__,function.__closure__,
        tuple(cell.cell_contents for cell in (function.__closure__ or ()))) for name,function in functions.items()}
    state._cpu_s45a_global_code_slots={name:_code_identity(value) for name,value in state._cpu_s45a_global_slots.items() if isinstance(value,FunctionType)}


def _install_internal_wrappers(state: Any, recorder: TimingRecorder) -> dict[str, Any]:
    original_stage = getattr(state, "_internal_stage_inner", None)
    original_apply = getattr(state, "_internal_apply_inner", None)
    if not callable(original_stage) or not callable(original_apply):
        raise InstrumentationBindingError("INTERNAL_STAGE_OR_APPLY_MISSING")
    source_globals = getattr(original_stage, "__globals__", None)
    if not isinstance(source_globals, dict) or getattr(original_apply, "__globals__", None) is not source_globals:
        raise InstrumentationBindingError("INTERNAL_STAGE_APPLY_NAMESPACE_MISMATCH")
    missing = [name for name in _RAMAN_REQUIRED if not callable(source_globals.get(name))]
    if missing:
        raise InstrumentationBindingError("INTERNAL_RAMAN_ANCHOR_MISSING:" + ",".join(missing))

    canonical_binding = _canonical_raman_binding()
    r8_binding = _verify_uninstrumented_r8_pair(state, original_stage, original_apply)

    namespace = dict(source_globals)
    # The private AST clones call this fixed closure.  It only retains the
    # scalar owner/recorder references; TimingRecorder itself stores no call
    # arguments, arrays, exception traces, or callbacks.
    def scope(label: str) -> Any:
        return recorder.scope(label, context=_metadata(state))

    namespace["__cpu_s45a_scope__"] = scope
    originals: dict[str, Callable[..., Any]] = {}
    canonical_function_identities: dict[str, str] = {}
    canonical_path, canonical_ast = _canonical_tree()
    canonical_direct = {
        name: _canonical_compiled_function(
            name,
            namespace=source_globals,
            source_path=canonical_path,
            tree=canonical_ast,
        )
        for name in ("raman_convolve_intensity", "make_raman_kernel", "stable_field_fluence_and_loss")
    }
    for name in _RAMAN_REQUIRED:
        original = source_globals[name]
        originals[name] = original
        if name == "raman_convolve_intensity":
            # The complete canonical IIR branch is instrumented as one AST
            # span.  This keeps coefficient/setup and the recursive loop in a
            # single bounded row without point-level logging.
            source_path, convolution_node, expected_convolution = _expected_transformed_node(
                name="raman_convolve_intensity",
                transform_name="__cpu_s45a_noop_transform__",
                candidate_namespace={"__cpu_s45a_noop_transform__": lambda _node: None},
            )
            canonical_function = canonical_direct[name]
            if _code_identity(canonical_function) != _code_identity(original):
                raise InstrumentationBindingError("RAMAN_CONVOLUTION_SOURCE_CHANGED")
            canonical_function_identities[name] = _code_identity(canonical_function)
            convolution_binding = _wrap_iir_site(convolution_node)
            instrumented = _compile_function_node(
                convolution_node, namespace=namespace, source_path=source_path
            )
            _copy_function_metadata(instrumented, original)
            setattr(instrumented, "__cpu_s45a_instrumented__", True)
            setattr(instrumented, "__cpu_s45a_original__", original)
            setattr(instrumented, "__cpu_s45a_original_code_identity__", _code_identity(original))
            setattr(instrumented, "__cpu_s45a_label__", "RAMAN_IIR")
            setattr(instrumented, "__cpu_s45a_nested_labels__", ())
            namespace[name] = instrumented
        elif name == "make_raman_kernel":
            canonical_function = canonical_direct[name]
            if _code_identity(canonical_function) != _code_identity(original):
                raise InstrumentationBindingError(f"RAMAN_HELPER_SOURCE_CHANGED:{name}")
            canonical_function_identities[name] = _code_identity(canonical_function)
            namespace[name] = _timed_wrapper(original, recorder, state, "RAMAN_COEFFICIENTS")
        elif name == "stable_field_fluence_and_loss":
            canonical_function = canonical_direct[name]
            if _code_identity(canonical_function) != _code_identity(original):
                raise InstrumentationBindingError(f"RAMAN_HELPER_SOURCE_CHANGED:{name}")
            canonical_function_identities[name] = _code_identity(canonical_function)
            namespace[name] = _timed_wrapper(original, recorder, state, "RAMAN_ENERGY")
        elif name == "_s43c_normalized_derivative":
            namespace[name] = _timed_wrapper(original, recorder, state, "FFT")
        else:
            namespace[name] = _timed_wrapper(original, recorder, state, "MATH_KERNELS")

    source_path, stage_node, _ = _expected_transformed_node(
        name="isaacs_raman_stage",
        transform_name="_transform_stage",
        candidate_namespace=getattr(state, "_candidate_namespace", {}),
    )
    stage_sites = _wrap_stage_sites(stage_node)
    instrumented_stage = _compile_function_node(
        stage_node, namespace=namespace, source_path=source_path
    )
    _copy_function_metadata(instrumented_stage, original_stage)
    setattr(instrumented_stage, "__cpu_s45a_instrumented__", True)
    setattr(instrumented_stage, "__cpu_s45a_original__", original_stage)
    setattr(instrumented_stage, "__cpu_s45a_original_code_identity__", _code_identity(original_stage))
    setattr(instrumented_stage, "__cpu_s45a_label__", "RAMAN_STAGE_BODY")
    setattr(instrumented_stage, "__cpu_s45a_stage_sites__", copy.deepcopy(stage_sites))
    state._internal_stage_inner = instrumented_stage
    state._internal_apply_inner = _clone_function(original_apply, namespace, "_s45a_apply_inner")
    setattr(state._internal_apply_inner, "__cpu_s45a_original__", original_apply)
    setattr(state._internal_apply_inner, "__cpu_s45a_original_code_identity__", _code_identity(original_apply))
    bindings: dict[str, Any] = {
        "schema": INSTRUMENT_SCHEMA,
        "canonical_raman_source": canonical_binding,
        "canonical_function_code_identities": canonical_function_identities,
        "r8_structural_binding": r8_binding,
        "stage_timing_sites": stage_sites,
        "iir_timing_sites": convolution_binding,
        "stage_stripped_ast_identity": pv.raw_sha256(
            ast.dump(_strip_timing_nodes(stage_node), include_attributes=False).encode("utf-8")
        ),
        "iir_stripped_ast_identity": pv.raw_sha256(
            ast.dump(_strip_timing_nodes(convolution_node), include_attributes=False).encode("utf-8")
        ),
        "internal_stage": {
            "original_code_identity": _code_identity(original_stage),
            "instrumented_code_identity": _code_identity(state._internal_stage_inner),
            "original_name": getattr(original_stage, "__name__", ""),
        },
        "internal_apply": {
            "original_code_identity": _code_identity(original_apply),
            "instrumented_code_identity": _code_identity(state._internal_apply_inner),
            "original_name": getattr(original_apply, "__name__", ""),
        },
        "helpers": {
            name: {
                "label": getattr(namespace[name], "__cpu_s45a_label__", ""),
                "original_code_identity": _code_identity(originals[name]),
                "wrapper_code_identity": _code_identity(namespace[name]),
                "original_module": str(getattr(originals[name], "__module__", "")),
                "original_name": str(getattr(originals[name], "__name__", name)),
                "globals_are_private_namespace": bool(
                    getattr(namespace[name], "__globals__", None) is namespace
                ),
            }
            for name in _RAMAN_REQUIRED
        },
    }
    if ast.dump(_strip_timing_nodes(stage_node), include_attributes=False) != ast.dump(
        _expected_transformed_node(
            name="isaacs_raman_stage",
            transform_name="_transform_stage",
            candidate_namespace=getattr(state, "_candidate_namespace", {}),
        )[1],
        include_attributes=False,
    ):
        raise InstrumentationBindingError("RAMAN_STAGE_STRIPPED_AST_MISMATCH")
    # The convolution function was cloned directly from canonical Raman; strip
    # its one generated IIR wrapper and prove the source tree is unchanged.
    _, original_convolution_node, _ = _expected_transformed_node(
        name="raman_convolve_intensity",
        transform_name="__cpu_s45a_noop_transform__",
        candidate_namespace={"__cpu_s45a_noop_transform__": lambda _node: None},
    )
    if ast.dump(_strip_timing_nodes(convolution_node), include_attributes=False) != ast.dump(
        original_convolution_node, include_attributes=False
    ):
        raise InstrumentationBindingError("RAMAN_IIR_STRIPPED_AST_MISMATCH")
    state._cpu_s45a_private_namespace = namespace
    state._cpu_s45a_original_helpers = originals
    state._cpu_s45a_bindings = bindings
    state._cpu_s45a_original_stage = original_stage
    state._cpu_s45a_original_apply = original_apply
    # Register actual loaded global objects, including NumPy/FFT aliases and
    # the fresh stage2 dispatcher.  These are existing namespace dependencies,
    # never per-call field/response/result values.
    names={instruction.argval for function in (instrumented_stage,state._internal_apply_inner)
           for instruction in dis.get_instructions(function) if instruction.opname=='LOAD_GLOBAL'}
    names.update(('xp','_time','c0','eps0','isaacs_raman_stage','__cpu_s45a_scope__'))
    state._cpu_s45a_global_slots={name:namespace[name] for name in names if name in namespace}
    bindings['registered_global_dependencies']=sorted(state._cpu_s45a_global_slots)
    return bindings


def make_instrumented_state_class(
    e_module: Any,
    *,
    candidate_schema: str,
    candidate_unsupported: type[BaseException],
    handoff_rejected: type[BaseException],
    profile: bool,
    recorder: TimingRecorder | None,
) -> type:
    """Build the accepted 4A state subclass with optional scalar timing."""

    from tools.cpu_s4_4a_linear_cache.linear_adapter import make_controlled_state_class

    base = make_controlled_state_class(
        e_module,
        candidate_schema=candidate_schema,
        candidate_unsupported=candidate_unsupported,
        handoff_rejected=handoff_rejected,
    )
    active_recorder = recorder if recorder is not None else TimingRecorder(enabled=bool(profile))

    class _InstrumentedState(base):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._cpu_s45a_profile = bool(profile)
            self._cpu_s45a_recorder = active_recorder if profile else TimingRecorder(enabled=False)
            self._cpu_s45a_bindings: dict[str, Any] = {}
            self._cpu_s45a_private_namespace: dict[str, Any] = {}
            self._cpu_s45a_original_helpers: dict[str, Callable[..., Any]] = {}
            self._cpu_s45a_guard_linear_function = getattr(self, "_g_linear_function", None)
            self._cpu_s45a_guard_nonlinear_function = getattr(self, "_g_nonlinear_function", None)
            self._cpu_s45a_guard_linear_code = getattr(self, "_g_linear_code", None)
            self._cpu_s45a_guard_nonlinear_code = getattr(self, "_g_nonlinear_code", None)
            if self._cpu_s45a_profile:
                self._cpu_s45a_bindings = _install_internal_wrappers(self, self._cpu_s45a_recorder)

        def _g_bind_function_identity(self, linear_function, nonlinear_function, binding):
            super()._g_bind_function_identity(linear_function, nonlinear_function, binding)
            self._cpu_s45a_guard_linear_function = self._g_linear_function
            self._cpu_s45a_guard_nonlinear_function = self._g_nonlinear_function
            self._cpu_s45a_guard_linear_code = self._g_linear_code
            self._cpu_s45a_guard_nonlinear_code = self._g_nonlinear_code

        def _g_controlled_substep(self, *args: Any, **kwargs: Any) -> Any:
            if not self._cpu_s45a_profile:
                return super()._g_controlled_substep(*args, **kwargs)
            metadata=_metadata(self)
            phase=str(getattr(getattr(self,"_g_pending_cap",None),"_scope","none"))
            metadata=context(metadata.interval,phase if phase in ('pre','post') else 'none',metadata.stage)
            with self._cpu_s45a_recorder.scope("RAMAN_HALF", context=metadata):
                return super()._g_controlled_substep(*args, **kwargs)

        def _g_linear_call(self, site: int, original: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
            if not self._cpu_s45a_profile:
                return super()._g_linear_call(site, original, args, kwargs)
            metadata=context(max(0,int(self._g_segment_index)-(1 if site==1 else 0)),"pre" if site==0 else "post")
            with self._cpu_s45a_recorder.scope("LINEAR", context=metadata):
                return super()._g_linear_call(site, original, args, kwargs)

        def _g_nonlinear_call(self, original: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
            if not self._cpu_s45a_profile:
                return super()._g_nonlinear_call(original, args, kwargs)
            with self._cpu_s45a_recorder.scope("NONLINEAR", context=_metadata(self)):
                return super()._g_nonlinear_call(original, args, kwargs)

        def _internal_stage(self, call_context: Any, field: Any, kwargs: Mapping[str, Any], *, stage2: bool) -> Any:
            if not self._cpu_s45a_profile:
                return super()._internal_stage(call_context, field, kwargs, stage2=stage2)
            label_stage = "stage2" if stage2 else "stage1"
            with self._cpu_s45a_recorder.scope("RAMAN_STAGE", context=_metadata(self, stage=label_stage)):
                return super()._internal_stage(call_context, field, kwargs, stage2=stage2)

        def _internal_apply(self, call_context: Any, field: Any, dz: Any, kwargs: Mapping[str, Any]) -> Any:
            if not self._cpu_s45a_profile:
                return super()._internal_apply(call_context, field, dz, kwargs)
            with self._cpu_s45a_recorder.scope("HEUN_ASSEMBLY", context=_metadata(self)):
                return super()._internal_apply(call_context, field, dz, kwargs)

        def metrics(self) -> dict[str, Any]:
            result = super().metrics()
            result["cpu_s4_5a_profile"] = bool(self._cpu_s45a_profile)
            result["cpu_s4_5a_instrumentation_bindings"] = copy.deepcopy(self._cpu_s45a_bindings)
            return result

    _InstrumentedState.__name__ = "_ControlledHandoffState"
    _InstrumentedState.__qualname__ = "_ControlledHandoffState"
    _InstrumentedState.__module__ = __name__
    return _InstrumentedState


def verify_bindings(state: Any) -> dict[str, Any]:
    """Verify guard identities, helper wrappers, and observer refusal state."""

    if not bool(getattr(state, "_cpu_s45a_profile", False)):
        return {"status": "PASS_PROFILE_DISABLED_NO_INSERTED_TIMERS"}
    linear = getattr(state, "_g_linear_function", None)
    nonlinear = getattr(state, "_g_nonlinear_function", None)
    if linear is None or nonlinear is None:
        raise InstrumentationBindingError("GUARD_PRODUCER_BINDING_MISSING")
    if linear is not getattr(state, "_cpu_s45a_guard_linear_function", None):
        raise InstrumentationBindingError("GUARD_LINEAR_FUNCTION_CHANGED")
    if nonlinear is not getattr(state, "_cpu_s45a_guard_nonlinear_function", None):
        raise InstrumentationBindingError("GUARD_NONLINEAR_FUNCTION_CHANGED")
    if getattr(state, "_g_linear_code", None) is not getattr(state, "_cpu_s45a_guard_linear_code", None):
        raise InstrumentationBindingError("GUARD_LINEAR_CODE_BINDING_CHANGED")
    if getattr(state, "_g_nonlinear_code", None) is not getattr(state, "_cpu_s45a_guard_nonlinear_code", None):
        raise InstrumentationBindingError("GUARD_NONLINEAR_CODE_BINDING_CHANGED")
    if getattr(linear, "__code__", None) is not getattr(state, "_g_linear_code", None):
        raise InstrumentationBindingError("GUARD_LINEAR_CODE_CHANGED")
    if getattr(nonlinear, "__code__", None) is not getattr(state, "_g_nonlinear_code", None):
        raise InstrumentationBindingError("GUARD_NONLINEAR_CODE_CHANGED")
    original_helpers = getattr(state, "_cpu_s45a_original_helpers", {})
    namespace = getattr(state, "_cpu_s45a_private_namespace", {})
    bindings = getattr(state, "_cpu_s45a_bindings", {})
    original_stage = getattr(state, "_cpu_s45a_original_stage", None)
    original_apply = getattr(state, "_cpu_s45a_original_apply", None)
    stage = getattr(state, "_internal_stage_inner", None)
    apply = getattr(state, "_internal_apply_inner", None)
    if not callable(original_stage) or not callable(stage) or getattr(stage, "__cpu_s45a_original__", None) is not original_stage:
        raise InstrumentationBindingError("INSTRUMENT_STAGE_BINDING_MISSING")
    if not callable(original_apply) or not callable(apply) or getattr(apply, "__cpu_s45a_original__", None) is not original_apply:
        raise InstrumentationBindingError("INSTRUMENT_APPLY_BINDING_MISSING")
    if _code_identity(original_stage) != bindings.get("internal_stage", {}).get("original_code_identity"):
        raise InstrumentationBindingError("INSTRUMENT_STAGE_SOURCE_CHANGED")
    if _code_identity(original_apply) != bindings.get("internal_apply", {}).get("original_code_identity"):
        raise InstrumentationBindingError("INSTRUMENT_APPLY_SOURCE_CHANGED")
    if _code_identity(stage) != bindings.get("internal_stage", {}).get("instrumented_code_identity"):
        raise InstrumentationBindingError("INSTRUMENT_STAGE_CODE_CHANGED")
    if _code_identity(apply) != bindings.get("internal_apply", {}).get("instrumented_code_identity"):
        raise InstrumentationBindingError("INSTRUMENT_APPLY_CODE_CHANGED")
    helper_records = bindings.get("helpers", {})
    for name in _RAMAN_REQUIRED:
        original = original_helpers.get(name)
        wrapper = namespace.get(name)
        if not callable(original) or not callable(wrapper):
            raise InstrumentationBindingError(f"INSTRUMENT_HELPER_BINDING_MISSING:{name}")
        if getattr(wrapper, "__cpu_s45a_original__", None) is not original:
            raise InstrumentationBindingError(f"INSTRUMENT_HELPER_ORIGINAL_CHANGED:{name}")
        if _code_identity(original) != helper_records.get(name, {}).get("original_code_identity"):
            raise InstrumentationBindingError(f"INSTRUMENT_HELPER_SOURCE_CHANGED:{name}")
        if _code_identity(wrapper) != helper_records.get(name, {}).get("wrapper_code_identity"):
            raise InstrumentationBindingError(f"INSTRUMENT_HELPER_WRAPPER_CODE_CHANGED:{name}")
        canonical_identity = bindings.get("canonical_function_code_identities", {}).get(name)
        if canonical_identity and _code_identity(original) != canonical_identity:
            raise InstrumentationBindingError(f"INSTRUMENT_HELPER_CANONICAL_SOURCE_CHANGED:{name}")
        if name == "raman_convolve_intensity":
            if getattr(wrapper, "__globals__", None) is not namespace:
                raise InstrumentationBindingError(f"INSTRUMENT_HELPER_NAMESPACE_CHANGED:{name}")
            if getattr(wrapper, "__cpu_s45a_label__", None) != "RAMAN_IIR":
                raise InstrumentationBindingError(f"INSTRUMENT_HELPER_LABEL_CHANGED:{name}")
        elif getattr(wrapper, "__globals__", None) is namespace:
            raise InstrumentationBindingError(f"INSTRUMENT_HELPER_NAMESPACE_UNEXPECTED:{name}")
    if getattr(stage, "__globals__", None) is not namespace or getattr(apply, "__globals__", None) is not namespace:
        raise InstrumentationBindingError("INSTRUMENT_INTERNAL_NAMESPACE_CHANGED")
    if getattr(state, "_raman_namespace", None) is namespace:
        raise InstrumentationBindingError("INSTRUMENT_RAMAN_NAMESPACE_REUSED")
    if getattr(state, "_candidate_namespace", None) is namespace:
        raise InstrumentationBindingError("INSTRUMENT_CANDIDATE_NAMESPACE_REUSED")
    for name,expected in state._cpu_s45a_global_slots.items():
        if namespace.get(name) is not expected:
            raise InstrumentationBindingError("INSTRUMENT_GLOBAL_ALIAS_CHANGED:"+name)
        code_hash=state.__dict__.get('_cpu_s45a_global_code_slots',{}).get(name)
        if code_hash is not None and _code_identity(expected)!=code_hash:
            raise InstrumentationBindingError("INSTRUMENT_GLOBAL_CODE_CHANGED:"+name)
    for name,(function,code,globals_dict,closure,contents) in state.__dict__.get('_cpu_s45a_callable_slots',{}).items():
        if function.__code__ is not code or function.__globals__ is not globals_dict or function.__closure__ is not closure:
            raise InstrumentationBindingError("INSTRUMENT_FIXED_CALLABLE_CHANGED:"+name)
        if any(cell.cell_contents is not expected for cell,expected in zip(closure or (),contents)):
            raise InstrumentationBindingError("INSTRUMENT_FIXED_CLOSURE_CHANGED:"+name)
    active_namespace=getattr(state,"_cpu_s45a_active_propagation_namespace",None)
    if active_namespace is not None:
        active=state._cpu_s45a_active_propagation_function
        if active.__code__ is not state._cpu_s45a_active_propagation_code or active.__globals__ is not active_namespace:
            raise InstrumentationBindingError("ACTIVE_PROPAGATION_CODE_OR_NAMESPACE_CHANGED")
        for name,row in bindings.get("propagation_helpers",{}).items():
            actual=active_namespace.get(name);original=state._cpu_s45a_propagation_originals[name]
            if getattr(actual,"__cpu_s45a_original__",None) is not original or _code_identity(actual)!=row["wrapper_code_identity"] or _code_identity(original)!=row["original_code_identity"]:
                raise InstrumentationBindingError("ACTIVE_PROPAGATION_HELPER_CHANGED:"+name)
    state._g_check_observer_state()
    return {
        "status": "PASS_S45A_PROFILE_BINDINGS",
        "guard_linear_code_identity": _code_identity(linear),
        "guard_nonlinear_code_identity": _code_identity(nonlinear),
        "helper_count": len(_RAMAN_REQUIRED),
        "private_namespace": True,
        "observer_state_checked": True,
    }


__all__ = [
    "INSTRUMENT_SCHEMA",
    "InstrumentationBindingError",
    "make_instrumented_state_class",
    "install_propagation_wrappers",
    "bind_active_propagation",
    "verify_bindings",
]
