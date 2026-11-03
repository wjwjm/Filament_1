"""Bounded, source-bound timing scopes for S4-6J factory decomposition.

The existing S4-5B recorder remains the only clock and metadata store.  This
module adds a small ContextVar gate so factory code can opt into labelled
scopes during one explicit top-level transaction.  Disabled instances do not
call the recorder at all.  Scope labels stay within the recorder's frozen
contract; the S4-6J category is carried by a fixed source site in the scalar
context.
"""

from __future__ import annotations

import contextvars
import copy
import importlib
import os
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_5b_steady_cost.timing import TimingRecorder


FACTORY_TIMING_SCHEMA = "filament.cpu.s4_6j.factory_timing.v1"
FACTORY_TIMING_SCOPE_LABEL = "L1_FACTORY"

# These are deliberately a finite registry.  A category is a reporting unit,
# not an invitation to add an arbitrary source site at runtime.
FACTORY_TIMING_CATEGORIES = (
    "source_read_hash",
    "git_blob_read",
    "ast_parse_lookup",
    "ast_transform_compile",
    "binding_verify",
    "metadata",
    "pool_lifecycle",
    "unattributed_factory",
)
_CATEGORY_SET = frozenset(FACTORY_TIMING_CATEGORIES)
FACTORY_TIMING_SITES = {
    "source_read_hash": "S46J.factory.source_read_hash",
    "git_blob_read": "S46J.factory.git_blob_read",
    "ast_parse_lookup": "S46J.factory.ast_parse_lookup",
    "ast_transform_compile": "S46J.factory.ast_transform_compile",
    "binding_verify": "S46J.factory.binding_verify",
    "metadata": "S46J.factory.metadata",
    "pool_lifecycle": "S46J.factory.pool_lifecycle",
    "unattributed_factory": "S46J.factory.unattributed_factory",
    "source_read_hash.power_hook.owned_source_identity.read_bytes":
        "S46J.factory.source_read_hash.power_hook.owned_source_identity.read_bytes",
    "git_blob_read.power_hook.owned_source_identity.git_objects":
        "S46J.factory.git_blob_read.power_hook.owned_source_identity.git_objects",
    "source_read_hash.power_hook.source_function.read_text":
        "S46J.factory.source_read_hash.power_hook.source_function.read_text",
    "ast_parse_lookup.power_hook.source_function.parse_lookup":
        "S46J.factory.ast_parse_lookup.power_hook.source_function.parse_lookup",
    "ast_transform_compile.power_hook.clone_function":
        "S46J.factory.ast_transform_compile.power_hook.clone_function",
    "ast_transform_compile.power_hook.instrument":
        "S46J.factory.ast_transform_compile.power_hook.instrument",
    "ast_transform_compile.power_hook.strip":
        "S46J.factory.ast_transform_compile.power_hook.strip",
    "metadata.power_hook.constructor.receipt":
        "S46J.factory.metadata.power_hook.constructor.receipt",
    "source_read_hash.combined_candidate.owned_source_identity":
        "S46J.factory.source_read_hash.combined_candidate.owned_source_identity",
    "source_read_hash.combined_candidate.source_identity":
        "S46J.factory.source_read_hash.combined_candidate.source_identity",
    "git_blob_read.combined_candidate.owned_source_identity":
        "S46J.factory.git_blob_read.combined_candidate.owned_source_identity",
    "binding_verify.combined_candidate.source_identity":
        "S46J.factory.binding_verify.combined_candidate.source_identity",
    "ast_transform_compile.combined_candidate.attach_with_c":
        "S46J.factory.ast_transform_compile.combined_candidate.attach_with_c",
    "metadata.combined_candidate.metadata":
        "S46J.factory.metadata.combined_candidate.metadata",
    "binding_verify.combined_candidate.verify_bindings":
        "S46J.factory.binding_verify.combined_candidate.verify_bindings",
    "pool_lifecycle.combined_candidate.make_pools":
        "S46J.factory.pool_lifecycle.combined_candidate.make_pools",
    "unattributed_factory.combined_candidate.build_role":
        "S46J.factory.unattributed_factory.combined_candidate.build_role",
    "source_read_hash.c_candidate.owned_source_identity":
        "S46J.factory.source_read_hash.c_candidate.owned_source_identity",
    "git_blob_read.c_candidate.owned_source_identity":
        "S46J.factory.git_blob_read.c_candidate.owned_source_identity",
    "binding_verify.c_candidate.verify_bindings":
        "S46J.factory.binding_verify.c_candidate.verify_bindings",
    "ast_transform_compile.c_candidate.attach_with_c":
        "S46J.factory.ast_transform_compile.c_candidate.attach_with_c",
    "metadata.c_candidate.build_role.binding":
        "S46J.factory.metadata.c_candidate.build_role.binding",
    "pool_lifecycle.c_candidate.build_role.pool_construct":
        "S46J.factory.pool_lifecycle.c_candidate.build_role.pool_construct",
    "pool_lifecycle.selected_factory.build_role.pool_construct":
        "S46J.factory.pool_lifecycle.selected_factory.build_role.pool_construct",
    "pool_lifecycle.selected_factory.finalize_factory":
        "S46J.factory.pool_lifecycle.selected_factory.finalize_factory",
    "binding_verify.selected_factory.verify_bindings":
        "S46J.factory.binding_verify.selected_factory.verify_bindings",
    "metadata.selected_factory.build_role.binding":
        "S46J.factory.metadata.selected_factory.build_role.binding",
    "source_read_hash.selected_factory.owned_source_identity":
        "S46J.factory.source_read_hash.selected_factory.owned_source_identity",
}
_SITE_CATEGORY = {
    key: key.split(".", 1)[0] for key in FACTORY_TIMING_SITES
}
_ACTIVE: contextvars.ContextVar["FactoryTiming | None"] = contextvars.ContextVar(
    "s46j_factory_timing", default=None
)


class FactoryTimingError(RuntimeError):
    """A factory timing contract or source-binding check failed closed."""


class _NoopScope(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tracked_record(path: Path, repo: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    record: dict[str, Any] = {
        "path": path.relative_to(repo).as_posix(),
        "size_bytes": len(payload),
        "classification": "local_worktree_text",
        "hash_scope": "raw_bytes",
        "raw_sha256": pv.raw_sha256(payload),
    }
    try:
        oid = pv._git_blob_oid(repo, record["path"])
        blob = pv._read_git_blob(repo, oid)
    except (OSError, pv.ProvenanceError) as exc:
        if os.environ.get("SLURM_JOB_ID"):
            raise FactoryTimingError("S46J_FACTORY_COMMITTED_SOURCE_REQUIRED") from exc
        return record
    if pv.canonical_lf_bytes(payload) != pv.canonical_lf_bytes(blob):
        if os.environ.get("SLURM_JOB_ID"):
            raise FactoryTimingError("S46J_FACTORY_SOURCE_BLOB_MISMATCH")
        return record
    record.update(
        classification="tracked_text",
        hash_scope=pv.TRACKED_HASH_SCOPE,
        git_blob_oid=oid,
        canonical_lf_sha256=pv.canonical_lf_sha256(blob),
    )
    return record


def source_identity(*, extra_paths: tuple[str | Path, ...] = ()) -> dict[str, Any]:
    """Return raw and, when available, PV2 tracked identity for this seam.

    ``extra_paths`` is explicit so the parent factory can bind the call-site
    files it instruments without this helper silently broadening its source
    contract.  No source bytes are retained after this function returns.
    """

    repo = _repo_root()
    paths = (Path(__file__), Path(__file__).with_name('scope_inventory.py')) + tuple(Path(item) for item in extra_paths)
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.suffix != ".py":
            raise FactoryTimingError("S46J_FACTORY_SOURCE_PATH_INVALID")
        try:
            relative = resolved.relative_to(repo).as_posix()
        except ValueError as exc:
            raise FactoryTimingError("S46J_FACTORY_SOURCE_OUTSIDE_REPO") from exc
        records[relative] = _tracked_record(resolved, repo)
    return {
        "schema": FACTORY_TIMING_SCHEMA + ".owned_sources.v1",
        "records": records,
        "categories": list(FACTORY_TIMING_CATEGORIES),
        "sites": dict(FACTORY_TIMING_SITES),
        "site_categories": dict(_SITE_CATEGORY),
        "scope_label": FACTORY_TIMING_SCOPE_LABEL,
    }


def _code_identity(function: Any) -> dict[str, Any]:
    # H's package initializer imports the instrumented C/combined modules.
    # Resolve its existing seal implementation only after imports complete.
    from tools.cpu_s4_6h_combined_parallel.instrument import _code_identity as code_identity
    function = getattr(function, '__func__', function)
    if getattr(function, "__code__", None) is None:
        raise FactoryTimingError("S46J_FACTORY_BACKEND_CODE_MISSING")
    return {
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
        "code_identity": code_identity(function),
    }


def _backend_identity(recorder: Any) -> dict[str, Any]:
    scope = getattr(recorder, "scope", None)
    begin = getattr(recorder, "begin", None)
    end = getattr(recorder, "end", None)
    context_factory = getattr(recorder, "context", None)
    if not all(callable(item) for item in (scope, begin, end, context_factory)):
        raise FactoryTimingError("S46J_FACTORY_RECORDER_INTERFACE_INVALID")
    return {
        "recorder_type": f"{type(recorder).__module__}.{type(recorder).__qualname__}",
        "scope": _code_identity(scope),
        "begin": _code_identity(begin),
        "end": _code_identity(end),
        "context": _code_identity(context_factory),
        "owner_thread_id": int(getattr(recorder, "owner_thread_id", -1)),
    }


class FactoryTiming:
    """One bounded timing transaction owned by an existing scalar recorder."""

    __slots__ = (
        "recorder",
        "enabled",
        "_source_paths",
        "_source_seal",
        "_backend_seal",
        "_alias_paths",
        "_alias_seal",
        "_active_tokens",
        "_bound_recorder", "_bound_active", "_function_seals", "_functions", "_failures", "_scope_classes", "_bound_enabled",
    )

    def __init__(
        self,
        recorder: Any,
        enabled: bool = False,
        *,
        extra_source_paths: tuple[str | Path, ...] = (),
        instrumentation_aliases: tuple[str, ...] = (),
    ) -> None:
        from tools.cpu_s4_6h_combined_parallel.instrument import _seal
        if type(recorder) is not TimingRecorder or recorder.enabled:
            raise FactoryTimingError('S46J_FACTORY_EXACT_PROFILE_OFF_RECORDER_REQUIRED')
        self.recorder = recorder
        self._bound_recorder = recorder
        self._bound_active = _ACTIVE
        self._scope_classes = (_NoopScope, _DeferredScope)
        self.enabled = bool(enabled)
        self._bound_enabled = self.enabled
        self._source_paths = tuple(str(Path(item)) for item in extra_source_paths)
        self._alias_paths = tuple(str(item) for item in instrumentation_aliases)
        self._source_seal = source_identity(extra_paths=tuple(extra_source_paths))
        self._backend_seal = _backend_identity(recorder)
        self._alias_seal = self._alias_identity()
        self._active_tokens: list[contextvars.Token[FactoryTiming | None]] = []
        self._functions = {
            'module.scope': scope, 'module.active': active,
            'module.source_identity': source_identity, 'module._backend_identity': _backend_identity,
            'module._tracked_record': _tracked_record, 'module._code_identity': _code_identity,
            'helper.deferred_enter': _DeferredScope.__enter__, 'helper.deferred_exit': _DeferredScope.__exit__,
            'helper.noop_enter': _NoopScope.__enter__, 'helper.noop_exit': _NoopScope.__exit__,
            'helper.activate_wrapped': FactoryTiming.activate.__wrapped__,
            **{'method.'+name:getattr(FactoryTiming, name) for name in
               ('activate','scope','call','verify_binding','_alias_identity','_resolve_alias')},
            **{'recorder.'+name:getattr(getattr(recorder,name),'__func__',getattr(recorder,name))
               for name in ('scope','begin','end','context')},
            **{'alias.'+name:self._resolve_alias(name) for name in self._alias_paths},
        }
        self._function_seals = {name:_seal(function) for name,function in self._functions.items()}
        self._failures: list[str] = []
        if any(function is not scope for name,function in self._functions.items() if name.startswith('alias.')):
            raise FactoryTimingError('S46J_FACTORY_ALIAS_NOT_ACTUAL_SCOPE')

    @property
    def categories(self) -> tuple[str, ...]:
        return FACTORY_TIMING_CATEGORIES

    @property
    def source_sites(self) -> Mapping[str, str]:
        return dict(FACTORY_TIMING_SITES)

    @staticmethod
    def _resolve_alias(path: str) -> Any:
        module_name, separator, attribute = path.rpartition(".")
        if not separator or not module_name or not attribute:
            raise FactoryTimingError("S46J_FACTORY_ALIAS_PATH_INVALID")
        module = importlib.import_module(module_name)
        value = getattr(module, attribute, None)
        if not callable(value):
            raise FactoryTimingError("S46J_FACTORY_ALIAS_NOT_CALLABLE")
        return value

    def _alias_identity(self) -> dict[str, dict[str, Any]]:
        return {path: _code_identity(self._resolve_alias(path)) for path in self._alias_paths}

    def verify_binding(self) -> dict[str, Any]:
        """Recheck source and recorder backend identity at a boundary."""
        from tools.cpu_s4_6h_combined_parallel.instrument import verify_seal
        if (type(self) is not FactoryTiming or self.recorder is not self._bound_recorder
            or self.recorder.enabled is not False
            or _ACTIVE is not self._bound_active or self._scope_classes != (_NoopScope, _DeferredScope)
            or self.enabled is not self._bound_enabled):
            raise FactoryTimingError('S46J_FACTORY_RECORDER_OR_CONTEXTVAR_IDENTITY_CHANGED')
        for name, function in self._functions.items():
            kind, key = name.split('.', 1)
            actual = (globals()[key] if kind == 'module' else getattr(FactoryTiming,key) if kind == 'method'
                      else {'deferred_enter':_DeferredScope.__enter__,'deferred_exit':_DeferredScope.__exit__,
                            'noop_enter':_NoopScope.__enter__,'noop_exit':_NoopScope.__exit__,
                            'activate_wrapped':FactoryTiming.activate.__wrapped__}[key] if kind=='helper'
                      else self._resolve_alias(key) if kind == 'alias' else
                      getattr(getattr(self.recorder,key),'__func__',getattr(self.recorder,key)))
            if actual is not function:
                raise FactoryTimingError('S46J_FACTORY_FUNCTION_IDENTITY_CHANGED:'+name)
            verify_seal(self._function_seals[name])
        if self._failures:
            raise FactoryTimingError('S46J_FACTORY_TIMING_FAILED:'+repr(self._failures))
        current_sources = source_identity(extra_paths=tuple(self._source_paths))
        if current_sources != self._source_seal:
            raise FactoryTimingError("S46J_FACTORY_SOURCE_BINDING_CHANGED")
        current_backend = _backend_identity(self.recorder)
        if current_backend != self._backend_seal:
            raise FactoryTimingError("S46J_FACTORY_BACKEND_BINDING_CHANGED")
        if self._alias_identity() != self._alias_seal:
            raise FactoryTimingError("S46J_FACTORY_ALIAS_BINDING_CHANGED")
        return {
            "schema": FACTORY_TIMING_SCHEMA + ".binding.v1",
            "enabled": self.enabled,
            "scope_label": FACTORY_TIMING_SCOPE_LABEL,
            "categories": list(FACTORY_TIMING_CATEGORIES),
            "source_identity": copy.deepcopy(self._source_seal),
            "backend_identity": copy.deepcopy(self._backend_seal),
            "contextvar": {"name": _ACTIVE.name, "identity": id(_ACTIVE)},
            "instrumentation_aliases": copy.deepcopy(self._alias_seal),
        }

    @contextmanager
    def activate(self) -> Iterator["FactoryTiming"]:
        """Activate this transaction for the current thread/context only."""

        self.verify_binding()
        token = self._bound_active.set(self)
        self._active_tokens.append(token)
        primary = None
        try:
            yield self
        except BaseException as error:
            primary = error
            raise
        finally:
            if not self._active_tokens or self._active_tokens[-1] is not token:
                raise FactoryTimingError("S46J_FACTORY_CONTEXT_STACK_CORRUPTED")
            self._active_tokens.pop()
            self._bound_active.reset(token)
            if _ACTIVE is not self._bound_active:
                self._failures.append('CONTEXTVAR_REPLACED_DURING_TRANSACTION')
            if self._failures:
                if primary is not None:
                    primary.add_note('S46J_FACTORY_TIMING_FAILED:'+repr(self._failures))
                else:
                    raise FactoryTimingError('S46J_FACTORY_TIMING_FAILED:'+repr(self._failures))

    def scope(self, category: str, *, site: str | None = None) -> AbstractContextManager[Any]:
        """Return one fixed-site scope, or a true no-op when disabled."""

        if category not in _CATEGORY_SET:
            raise FactoryTimingError("S46J_FACTORY_CATEGORY_INVALID")
        site_key = category if site is None else str(site)
        if site_key not in FACTORY_TIMING_SITES:
            raise FactoryTimingError("S46J_FACTORY_SITE_INVALID")
        if _SITE_CATEGORY[site_key] != category:
            raise FactoryTimingError("S46J_FACTORY_SITE_CATEGORY_MISMATCH")
        if not self.enabled or self._failures:
            return _NoopScope()
        if _ACTIVE.get() is not self:
            raise FactoryTimingError("S46J_FACTORY_TIMING_NOT_ACTIVE")
        source_site = FACTORY_TIMING_SITES[site_key]
        context_factory = getattr(self.recorder, "context", None)
        if not callable(context_factory):
            raise FactoryTimingError("S46J_FACTORY_RECORDER_CONTEXT_INVALID")
        try:
            manager = self.recorder.scope(
                FACTORY_TIMING_SCOPE_LABEL,
                context=context_factory(0, occurrence_class="WINDOW_ONCE", source_site=source_site),
                common=True,
            )
        except BaseException as error:
            self._failures.append('SCOPE:'+type(error).__name__)
            return _NoopScope()
        return _DeferredScope(self, manager)

    def call(self, category: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a scalar-only factory call under one registered scope."""

        with self.scope(category):
            return function(*args, **kwargs)


class _DeferredScope(AbstractContextManager):
    """Preserve the body exception and let original pool cleanup finish first."""
    def __init__(self, owner, manager):
        self.owner, self.manager, self.entered = owner, manager, False

    def __enter__(self):
        try:
            value = self.manager.__enter__()
            self.entered = True
            return value
        except BaseException as error:
            self.owner._failures.append('ENTER:'+type(error).__name__)
            return None

    def __exit__(self, kind, value, traceback):
        if self.entered:
            try:
                self.manager.__exit__(kind, value, traceback)
            except BaseException as error:
                self.owner._failures.append('EXIT:'+type(error).__name__)
                if value is not None:
                    value.add_note('S46J_FACTORY_TIMER_EXIT_FAILED:'+type(error).__name__)
        return False

def active() -> FactoryTiming | None:
    """Return the current transaction for narrow instrumentation seams."""

    return _ACTIVE.get()


def scope(category: str, *, site: str | None = None) -> AbstractContextManager[Any]:
    """Use the active transaction from an existing factory seam.

    This keeps call-site instrumentation harmless when no S4-6J transaction is
    active.  The returned object never stores arrays or other runtime state.
    """

    timing = active()
    if timing is None:
        return _NoopScope()
    return timing.scope(category, site=site)


__all__ = [
    "FACTORY_TIMING_SCHEMA",
    "FACTORY_TIMING_SCOPE_LABEL",
    "FACTORY_TIMING_CATEGORIES",
    "FACTORY_TIMING_SITES",
    "FactoryTiming",
    "FactoryTimingError",
    "active",
    "scope",
    "source_identity",
]
