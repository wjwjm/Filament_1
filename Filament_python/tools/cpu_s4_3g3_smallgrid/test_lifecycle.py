"""Focused software-contract tests for the small-grid lifecycle candidate.

These tests exercise only metadata binding and retirement.  They deliberately
do not launch a propagation or construct a numba execution kernel.
"""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from .adapter import (
    ADAPTER_SCHEMA,
    BOUNDARY_CONTRACT,
    LINEAR_RETURN_FORMS,
    _resolve_resource_contract,
    build_role,
)
from .candidate import (
    CANDIDATE_SCHEMA,
    LOCAL_ENGINEERING_CONTRACT,
    POOL_CONTRACT,
    CandidateUnsupported,
    source_identity,
)
from .controlled_callsite import (
    ControlledLifetimeError,
    ControlledLifetimeIdentity,
    _lifecycle_token,
)


def _owner_for(field: np.ndarray) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    def producer() -> None:
        return None

    def nonlinear() -> None:
        return None

    context = SimpleNamespace(generation=11)
    owner = SimpleNamespace(
        _local=SimpleNamespace(context=context),
        _generation=7,
        _g_owner_generation=3,
        _g_linear_function=producer,
        _g_nonlinear_function=nonlinear,
        _g_linear_code=producer.__code__,
        _g_nonlinear_code=nonlinear.__code__,
    )
    capability = SimpleNamespace(
        _owner=owner,
        _field=field,
        _scope="pre",
        _thread_id=threading.get_ident(),
        _retired=False,
        _consumed=False,
        _site=0,
        _generation=3,
    )
    owner._g_pending_cap = capability
    owner._parameter_token = lambda kwargs: {"omega": str(kwargs["omega"])}
    return owner, context, capability


def test_smallgrid_lifecycle_token_records_identity_and_layout() -> None:
    field = np.zeros((2, 3, 4), dtype=np.complex128)
    owner, context, capability = _owner_for(field)

    token = _lifecycle_token(owner, field, {"omega": "frozen"})
    identity = token["lifetime_identity"]

    assert isinstance(identity, ControlledLifetimeIdentity)
    assert token["identity_kind"] == "controlled_lifetime_v1"
    assert token["shape"] == [2, 3, 4]
    assert token["dtype"] == field.dtype.str
    assert token["strides_bytes"] == [192, 64, 16]
    assert token["thread_id"] == threading.get_ident()
    assert token["producer_site"] == 0
    assert token["producer_function_ids"] == list(identity.producer_identity[:2])
    assert token["producer_code_ids"] == list(identity.code_identity)
    assert "content_" + "raw_sha256" not in token
    identity.validate(owner, field, context, {"omega": "frozen"})

    identity.retire(consumed=True)
    assert identity.as_record()["consumed"] is True
    with pytest.raises(ControlledLifetimeError, match="LIFETIME_IDENTITY_SCHEMA_OR_CONSUMPTION_INVALID"):
        identity.validate(owner, field, context, {"omega": "frozen"})


def test_smallgrid_lifecycle_rejects_new_field_object() -> None:
    field = np.zeros((2, 3, 4), dtype=np.complex128)
    owner, context, _ = _owner_for(field)
    token = _lifecycle_token(owner, field, {"omega": "frozen"})
    identity = token["lifetime_identity"]

    with pytest.raises(ControlledLifetimeError, match="LIFETIME_IDENTITY_FIELD_OBJECT_MISMATCH"):
        identity.validate(owner, field.copy(), context, {"omega": "frozen"})


def test_smallgrid_lifecycle_rejects_producer_code_rebind() -> None:
    field = np.zeros((2, 3, 4), dtype=np.complex128)
    owner, context, _ = _owner_for(field)
    token = _lifecycle_token(owner, field, {"omega": "frozen"})
    identity = token["lifetime_identity"]

    def replacement() -> None:
        return None

    owner._g_linear_code = replacement.__code__
    with pytest.raises(ControlledLifetimeError, match="LIFETIME_IDENTITY_.*CODE_MISMATCH"):
        identity.validate(owner, field, context, {"omega": "frozen"})


def test_smallgrid_source_schema_and_pool_contract_are_static() -> None:
    identity = source_identity()
    assert identity["schema"] == CANDIDATE_SCHEMA
    assert identity["lifecycle_identity"] == "controlled_lifetime_v1"
    assert identity["mathematics_unchanged"] is True
    assert POOL_CONTRACT["I0_SMALL"]["numba_pool"] == 24
    assert POOL_CONTRACT["G2_SMALL"]["numba_pool"] == 48
    assert POOL_CONTRACT["G3_SMALL"]["numba_pool"] == 48
    assert LOCAL_ENGINEERING_CONTRACT["numba_pool"] == 1
    assert LOCAL_ENGINEERING_CONTRACT["hpc_allowed"] is False
    assert ADAPTER_SCHEMA.endswith("adapter.v1")
    assert len(LINEAR_RETURN_FORMS) == 4
    assert BOUNDARY_CONTRACT["qualify"]["raw_reference_escape"] is False
    assert _resolve_resource_contract("G3_SMALL", local_engineering=True, pool_override=1)["numba_pool"] == 1
    with pytest.raises(ValueError, match="pool_override"):
        _resolve_resource_contract("G3_SMALL", local_engineering=False, pool_override=4)
    assert inspect.signature(build_role).parameters["capture_root"].kind is inspect.Parameter.KEYWORD_ONLY
