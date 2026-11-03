from __future__ import annotations

import pytest

from tools.cpu_s4_5b_steady_cost.timing import TimingRecorder
from tools.cpu_s4_6j_alpha_factory.factory_timing import (
    FACTORY_TIMING_CATEGORIES,
    FactoryTiming,
    FactoryTimingError,
    active,
)


def _timing(*, enabled: bool) -> FactoryTiming:
    recorder = TimingRecorder(enabled=False, common_shells_enabled=False)
    return FactoryTiming(recorder, enabled=enabled)


def test_disabled_scope_does_not_record() -> None:
    timing = _timing(enabled=False)
    with timing.activate():
        with timing.scope("metadata"):
            pass
    assert timing.recorder.snapshot()["records"] == []


def test_enabled_scope_and_nested_exclusive_clock() -> None:
    timing = _timing(enabled=True)
    with timing.activate():
        assert active() is timing
        with timing.scope("unattributed_factory"):
            with timing.scope("metadata"):
                pass
    snapshot = timing.recorder.snapshot()
    assert len(snapshot["records"]) == 2
    assert {row["context"]["source_site"] for row in snapshot["records"]} == {
        "S46J.factory.unattributed_factory",
        "S46J.factory.metadata",
    }
    assert snapshot["clock_closure"]["pass"] is True
    assert all(row["wall_exclusive_ns"] >= 0 for row in snapshot["records"])


def test_activation_restores_context_after_exception() -> None:
    timing = _timing(enabled=True)
    with pytest.raises(RuntimeError):
        with timing.activate():
            raise RuntimeError("expected")
    assert active() is None
    with pytest.raises(FactoryTimingError, match="NOT_ACTIVE"):
        timing.scope("metadata").__enter__()


def test_invalid_category_rejected() -> None:
    timing = _timing(enabled=False)
    with pytest.raises(FactoryTimingError, match="CATEGORY_INVALID"):
        timing.scope("arbitrary")
    assert len(FACTORY_TIMING_CATEGORIES) == 8


def test_source_binding_change_rejected() -> None:
    timing = _timing(enabled=False)
    timing._source_seal = {"tampered": True}  # type: ignore[attr-defined]
    with pytest.raises(FactoryTimingError, match="SOURCE_BINDING_CHANGED"):
        timing.verify_binding()


def test_backend_binding_change_rejected(monkeypatch) -> None:
    timing = _timing(enabled=False)
    monkeypatch.setattr(TimingRecorder, 'scope', lambda *args, **kwargs: None)
    with pytest.raises(FactoryTimingError, match="FUNCTION_IDENTITY_CHANGED"):
        timing.verify_binding()

def test_same_type_recorder_and_same_name_contextvar_replacements_rejected(monkeypatch):
    import contextvars
    from tools.cpu_s4_6j_alpha_factory import factory_timing as mod
    timing = _timing(enabled=False)
    timing.recorder = TimingRecorder(enabled=False, common_shells_enabled=False)
    with pytest.raises(FactoryTimingError, match='RECORDER_OR_CONTEXTVAR'):
        timing.verify_binding()
    timing.recorder = timing._bound_recorder
    monkeypatch.setattr(mod, '_ACTIVE', contextvars.ContextVar('s46j_factory_timing', default=None))
    with pytest.raises(FactoryTimingError, match='RECORDER_OR_CONTEXTVAR'):
        timing.verify_binding()

def test_alias_clone_with_same_code_is_rejected(monkeypatch):
    from types import FunctionType
    from tools.cpu_s4_6i_residual_parallel import power_hook
    from tools.cpu_s4_6j_alpha_factory import factory_timing as mod
    timing = FactoryTiming(TimingRecorder(enabled=False, common_shells_enabled=False),
        instrumentation_aliases=('tools.cpu_s4_6i_residual_parallel.power_hook.factory_scope',))
    clone = FunctionType(mod.scope.__code__, dict(mod.scope.__globals__), argdefs=mod.scope.__defaults__)
    clone.__kwdefaults__ = mod.scope.__kwdefaults__
    monkeypatch.setattr(power_hook, 'factory_scope', clone)
    with pytest.raises(FactoryTimingError, match='FUNCTION_IDENTITY_CHANGED'):
        timing.verify_binding()

def test_timer_exit_failure_is_deferred_until_original_cleanup_completes():
    from tools.cpu_s4_6j_alpha_factory.factory_timing import _DeferredScope
    timing = _timing(enabled=True)
    closed = []
    class BrokenClock:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            raise RuntimeError('clock end failed')
    with pytest.raises(FactoryTimingError, match='TIMING_FAILED'):
        with timing.activate():
            with _DeferredScope(timing, BrokenClock()):
                closed.append('power')
            with timing.scope('pool_lifecycle'):
                closed.append('original_three')
    assert closed == ['power','original_three']
    assert active() is None

def test_primary_exception_not_replaced_by_timer_failure():
    from tools.cpu_s4_6j_alpha_factory.factory_timing import _DeferredScope
    timing = _timing(enabled=True)
    class BrokenClock:
        def __enter__(self): pass
        def __exit__(self, *args): raise ValueError('clock error')
    with pytest.raises(KeyError, match='primary'):
        with timing.activate():
            with _DeferredScope(timing, BrokenClock()):
                raise KeyError('primary')

@pytest.mark.parametrize('name', ['source_identity','_backend_identity'])
def test_identity_helper_replacement_rejected(monkeypatch, name):
    from tools.cpu_s4_6j_alpha_factory import factory_timing as mod
    timing = _timing(enabled=False)
    monkeypatch.setattr(mod, name, lambda *args, **kwargs: {})
    with pytest.raises(FactoryTimingError, match='FUNCTION_IDENTITY_CHANGED'):
        timing.verify_binding()

def test_recorder_profile_replacement_rejected():
    timing = _timing(enabled=False)
    timing.recorder.enabled = True
    with pytest.raises(FactoryTimingError, match='RECORDER_OR_CONTEXTVAR'):
        timing.verify_binding()
