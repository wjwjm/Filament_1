"""Small-grid G3 lifecycle identity bridge.

The numerical call-site and the frozen F/E/D/R8 kernels remain the accepted
G3 implementation.  This module supplies the one new private handoff
representation used by the small-grid candidate.  It records ownership and
layout metadata in O(1); it deliberately does not hash or copy the field.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

import numpy as np

from tools.cpu_s4_3g import controlled_callsite as _legacy


PROPAGATE_RELATIVE_PATH = _legacy.PROPAGATE_RELATIVE_PATH
PROPAGATE_SOURCE_BLOB = _legacy.PROPAGATE_SOURCE_BLOB
PROPAGATE_CANONICAL_LF_SHA256 = _legacy.PROPAGATE_CANONICAL_LF_SHA256
CallsiteBindingError = _legacy.CallsiteBindingError
bind_propagator = _legacy.bind_propagator
source_binding = _legacy.source_binding
capability_type = _legacy.capability_type


class ControlledLifetimeError(RuntimeError):
    """A private lifecycle identity was used outside its admitted scope."""


class ControlledLifetimeIdentity:
    """Opaque, scalar-sized identity for one producer field handoff.

    The object keeps private references to the admitted owner and capability.
    ``as_record`` is the only serialisable surface and contains metadata only;
    in particular it has no content digest member.
    """

    __slots__ = (
        "kind",
        "owner",
        "capability",
        "call_context_id",
        "call_generation",
        "heun_generation",
        "propagation_generation",
        "scope",
        "thread_id",
        "producer_identity",
        "code_identity",
        "field_object_id",
        "data_pointer",
        "shape",
        "dtype",
        "strides",
        "parameter_token",
        "valid",
        "consumed",
    )

    def __init__(
        self,
        *,
        owner: Any,
        capability: Any,
        context: Any,
        field: np.ndarray,
        parameter_token: Any,
    ) -> None:
        shape, dtype, strides, pointer = _legacy._field_layout(field)
        producer = getattr(owner, "_g_linear_function", None)
        nonlinear = getattr(owner, "_g_nonlinear_function", None)
        producer_code = getattr(owner, "_g_linear_code", None)
        nonlinear_code = getattr(owner, "_g_nonlinear_code", None)
        if (
            not callable(producer)
            or not callable(nonlinear)
            or producer_code is None
            or nonlinear_code is None
        ):
            raise ControlledLifetimeError("LIFETIME_IDENTITY_PRODUCER_CODE_MISSING")
        self.kind = "controlled_lifetime_v1"
        self.owner = owner
        self.capability = capability
        self.call_context_id = int(id(context))
        self.call_generation = int(getattr(context, "generation", -1))
        self.heun_generation = int(getattr(owner, "_generation", -1))
        self.propagation_generation = int(getattr(owner, "_g_owner_generation", -1))
        self.scope = str(getattr(capability, "_scope", ""))
        self.thread_id = int(getattr(capability, "_thread_id", -1))
        self.producer_identity = (
            int(id(producer)),
            int(id(nonlinear)),
            int(getattr(capability, "_site", -1)),
        )
        self.code_identity = (int(id(producer_code)), int(id(nonlinear_code)))
        self.field_object_id = int(id(field))
        self.data_pointer = int(pointer)
        self.shape = tuple(int(item) for item in shape)
        self.dtype = str(dtype)
        self.strides = tuple(int(item) for item in strides)
        self.parameter_token = parameter_token
        self.valid = True
        self.consumed = False

    def retire(self, *, consumed: bool = False) -> None:
        self.valid = False
        self.consumed = bool(consumed)

    def validate(self, owner: Any, field: Any, context: Any, parameter_token: Any) -> None:
        if self.kind != "controlled_lifetime_v1" or self.consumed:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_SCHEMA_OR_CONSUMPTION_INVALID")
        if not self.valid or self.owner is not owner:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_OWNER_INVALID")
        capability = self.capability
        if capability is None or getattr(capability, "_owner", None) is not owner:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_CAPABILITY_OWNER_INVALID")
        if self.scope not in ("pre", "post") or str(getattr(capability, "_scope", "")) != self.scope:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_SCOPE_MISMATCH")
        if getattr(capability, "_retired", False) or getattr(capability, "_consumed", False):
            raise ControlledLifetimeError("LIFETIME_IDENTITY_CAPABILITY_RETIRED")
        if capability is not getattr(owner, "_g_pending_cap", None):
            raise ControlledLifetimeError("LIFETIME_IDENTITY_CAPABILITY_NOT_CURRENT")
        if int(getattr(context, "generation", -1)) != self.call_generation:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_CALL_GENERATION_MISMATCH")
        if int(id(context)) != self.call_context_id:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_CALL_CONTEXT_MISMATCH")
        if int(getattr(owner, "_generation", -1)) != self.heun_generation:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_HEUN_GENERATION_MISMATCH")
        if int(getattr(owner, "_g_owner_generation", -1)) != self.propagation_generation:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_PROPAGATION_GENERATION_MISMATCH")
        if int(getattr(capability, "_generation", -1)) != self.propagation_generation:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_CAPABILITY_GENERATION_MISMATCH")
        if self.thread_id != int(getattr(capability, "_thread_id", -1)) or self.thread_id != int(__import__("threading").get_ident()):
            raise ControlledLifetimeError("LIFETIME_IDENTITY_THREAD_MISMATCH")
        producer = getattr(owner, "_g_linear_function", None)
        nonlinear = getattr(owner, "_g_nonlinear_function", None)
        producer_code = getattr(owner, "_g_linear_code", None)
        nonlinear_code = getattr(owner, "_g_nonlinear_code", None)
        current_producer_identity = (
            int(id(producer)),
            int(id(nonlinear)),
            int(getattr(capability, "_site", -1)),
        )
        current_code_identity = (int(id(producer_code)), int(id(nonlinear_code)))
        if getattr(producer, "__code__", None) is not producer_code or getattr(nonlinear, "__code__", None) is not nonlinear_code:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_LIVE_CODE_MISMATCH")
        if current_producer_identity != self.producer_identity:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_PRODUCER_MISMATCH")
        if current_code_identity != self.code_identity:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_CODE_MISMATCH")
        if field is not getattr(capability, "_field", None) or id(field) != self.field_object_id:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_FIELD_OBJECT_MISMATCH")
        shape, dtype, strides, pointer = _legacy._field_layout(field)
        if (tuple(shape), str(dtype), tuple(strides), int(pointer)) != (
            self.shape,
            self.dtype,
            self.strides,
            self.data_pointer,
        ):
            raise ControlledLifetimeError("LIFETIME_IDENTITY_FIELD_LAYOUT_MISMATCH")
        if parameter_token != self.parameter_token:
            raise ControlledLifetimeError("LIFETIME_IDENTITY_PARAMETER_MISMATCH")

    def as_record(self) -> dict[str, Any]:
        return {
            "identity_kind": self.kind,
            "call_context_id": self.call_context_id,
            "call_generation": self.call_generation,
            "heun_generation": self.heun_generation,
            "propagation_generation": self.propagation_generation,
            "scope": self.scope,
            "thread_id": self.thread_id,
            "producer_function_ids": list(self.producer_identity[:2]),
            "producer_site": self.producer_identity[2],
            "producer_code_ids": list(self.code_identity),
            "field_object_id": self.field_object_id,
            "data_pointer": self.data_pointer,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "strides_bytes": list(self.strides),
            "valid": bool(self.valid),
            "consumed": bool(self.consumed),
        }


def _lifecycle_token(owner: Any, field: np.ndarray, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Create a typed metadata token and retain no field bytes."""
    context = getattr(getattr(owner, "_local", None), "context", None)
    capability = getattr(owner, "_g_pending_cap", None)
    if context is None or capability is None:
        raise ControlledLifetimeError("LIFETIME_IDENTITY_CONTEXT_OR_CAPABILITY_MISSING")
    parameter_token = owner._parameter_token(kwargs)
    identity = ControlledLifetimeIdentity(
        owner=owner,
        capability=capability,
        context=context,
        field=field,
        parameter_token=parameter_token,
    )
    return _metadata_token(identity, owner, field, parameter_token)


def _metadata_token(identity, owner, field, parameter_token):
    capability = owner._g_pending_cap
    shape, dtype, strides, pointer = _legacy._field_layout(field)
    return {
        "identity_kind": identity.kind,
        "lifetime_identity": identity,
        "owner_id": int(id(owner)),
        "capability_id": int(id(capability)),
        "call_context_id": identity.call_context_id,
        "call_generation": identity.call_generation,
        "heun_generation": identity.heun_generation,
        "propagation_generation": identity.propagation_generation,
        "scope": identity.scope,
        "thread_id": identity.thread_id,
        "producer_function_ids": list(identity.producer_identity[:2]),
        "producer_site": identity.producer_identity[2],
        "producer_code_ids": list(identity.code_identity),
        "shape": list(shape),
        "dtype": str(dtype),
        "strides_bytes": list(strides),
        "data_pointer": int(pointer),
        "object_id": int(id(field)),
        "parameter_token": parameter_token,
    }


def make_controlled_state_class(
    e_module: Any,
    *,
    candidate_schema: str,
    candidate_unsupported: type[BaseException],
    handoff_rejected: type[BaseException],
) -> type:
    """Build the private E state class with lifecycle-token semantics.

    The legacy builder supplies all frozen runtime checks and numerical
    dispatch.  The subclass changes only the private field token and the
    controlled hash-role accounting; strict/public methods remain inherited.
    """
    legacy_class = _legacy.make_controlled_state_class(
        e_module,
        candidate_schema=candidate_schema,
        candidate_unsupported=candidate_unsupported,
        handoff_rejected=handoff_rejected,
    )

    class _ControlledLifetimeState(legacy_class):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._g_lifecycle_identity: ControlledLifetimeIdentity | None = None
            self._stats.update(
                {
                    "schema": str(candidate_schema),
                    "candidate_name": "G3_SG_CONTROLLED_LIFECYCLE_V1",
                    "candidate_family": "CONTROLLED_LIFETIME_IDENTITY_V1",
                    "controlled_hash_role_order": [
                        "Omega_stage1",
                        "Omega_apply",
                        "Omega_stage2",
                    ],
                    "lifecycle_identity_created": 0,
                    "lifecycle_identity_validated": 0,
                    "lifecycle_identity_retired": 0,
                    "lifecycle_identity_rejected": 0,
                }
            )

        def _field_token(self, field: np.ndarray, kwargs: Mapping[str, Any]) -> dict[str, Any]:
            controlled = bool(getattr(self._local, "g_controlled", False))
            phase = str(getattr(self._local, "g_phase", ""))
            if not controlled:
                return super()._field_token(field, kwargs)
            if phase == "stage1":
                token = _lifecycle_token(self, field, kwargs)
                self._g_lifecycle_identity = token["lifetime_identity"]
                self._stats["lifecycle_identity_created"] += 1
                return token
            if phase == "apply":
                context = getattr(self._local, "context", None)
                stored = getattr(context, "field_token", None)
                identity = stored.get("lifetime_identity") if isinstance(stored, Mapping) else None
                if not isinstance(identity, ControlledLifetimeIdentity):
                    self._stats["lifecycle_identity_rejected"] += 1
                    self._reject("LIFETIME_IDENTITY_SCHEMA_MISSING")
                parameter_token = self._parameter_token(kwargs)
                try:
                    identity.validate(self, field, context, parameter_token)
                except ControlledLifetimeError as exc:
                    self._stats["lifecycle_identity_rejected"] += 1
                    self._reject(str(exc))
                self._stats["lifecycle_identity_validated"] += 1
                current = _metadata_token(identity, self, field, parameter_token)
                if set(stored) != set(current) or stored != current:
                    self._reject("LIFETIME_IDENTITY_TOKEN_SCHEMA_OR_METADATA_CHANGED")
                return current
            self._reject("LIFETIME_IDENTITY_PHASE_INVALID")

        def _set_controlled_hash_stats(self) -> None:
            expected = ["Omega_stage1", "Omega_apply", "Omega_stage2"]
            actual = [str(item.get("role")) for item in self._hash_records]
            if actual != expected:
                self._reject(f"CONTROLLED_HASH_ROLE_SEQUENCE:{actual}")
            context = getattr(self._local, "context", None)
            cap = self._g_pending_cap
            if context is None or cap is None:
                self._reject("CONTROLLED_HASH_FINAL_CONTEXT_OR_SCOPE_MISSING")
            for item, expected_role in zip(self._hash_records, expected):
                if item.get("source_role") != "Omega_Omega":
                    self._reject("CONTROLLED_HASH_SOURCE_ROLE_MISMATCH")
                if str(item.get("role")) != expected_role:
                    self._reject("CONTROLLED_HASH_ROLE_MISMATCH")
                if int(item.get("generation", -1)) != int(context.generation):
                    self._reject("CONTROLLED_HASH_GENERATION_MISMATCH")
                if str(item.get("scope")) != str(cap._scope):
                    self._reject("CONTROLLED_HASH_SCOPE_MISMATCH")
            logical_bytes = int(sum(int(item["logical_bytes"]) for item in self._hash_records))
            self._stats["hash_leaf_calls"] += len(actual)
            self._stats["hash_leaf_logical_bytes"] += logical_bytes
            self._stats["last_hash_leaf_calls"] = len(actual)
            self._stats["last_hash_leaf_logical_bytes"] = logical_bytes
            self._stats["last_controlled_hash_roles"] = list(actual)
            self._stats["last_controlled_hash_events"] = copy.deepcopy(self._hash_records)
            counts: dict[str, int] = {}
            for role in actual:
                counts[role] = int(counts.get(role, 0)) + 1
                self._stats["hash_leaf_role_counts"][role] = int(
                    self._stats["hash_leaf_role_counts"].get(role, 0)
                ) + 1
                self._stats["controlled_hash_role_counts"][role] = int(
                    self._stats["controlled_hash_role_counts"].get(role, 0)
                ) + 1
            self._stats["last_hash_leaf_role_counts"] = counts

        def _retire_lifecycle(self, *, consumed: bool = False) -> None:
            identity = self._g_lifecycle_identity
            if identity is not None and identity.valid:
                identity.retire(consumed=consumed)
                self._stats["lifecycle_identity_retired"] += 1
            self._g_lifecycle_identity = None

        def _g_controlled_full_heun(self, field, dz, kwargs, cap):
            previous = self._g_lifecycle_identity
            try:
                return super()._g_controlled_full_heun(field, dz, kwargs, cap)
            finally:
                if self._g_lifecycle_identity is not previous:
                    self._retire_lifecycle(consumed=bool(getattr(cap, "_consumed", False)))

        def _g_controlled_substep(self, *args: Any, **kwargs: Any) -> Any:
            capability = self._g_pending_cap
            result = super()._g_controlled_substep(*args, **kwargs)
            identity = self._g_lifecycle_identity
            if (
                identity is not None
                and capability is identity.capability
                and bool(getattr(capability, "_consumed", False))
            ):
                self._retire_lifecycle(consumed=True)
            return result

        def _g_before_nonlinear(self, field: Any) -> None:
            self._retire_lifecycle(consumed=True)
            return super()._g_before_nonlinear(field)

        def _g_abort_propagation(self, reason: str) -> None:
            self._retire_lifecycle(consumed=False)
            return super()._g_abort_propagation(reason)

        def _g_publish_result(self, value: Any) -> Any:
            self._retire_lifecycle(consumed=True)
            return super()._g_publish_result(value)

        def metrics(self) -> dict[str, Any]:
            result = super().metrics()
            result["schema"] = str(candidate_schema)
            result["candidate_name"] = "G3_SG_CONTROLLED_LIFECYCLE_V1"
            result["controlled_hash_role_order"] = [
                "Omega_stage1",
                "Omega_apply",
                "Omega_stage2",
            ]
            controlled_counts = result.get("controlled_hash_role_counts", {})
            result["controlled_E_stage1_hash_calls"] = int(controlled_counts.get("E_stage1", 0))
            result["controlled_E_apply_hash_calls"] = int(controlled_counts.get("E_apply", 0))
            result["active_lifecycle_identity"] = bool(self._g_lifecycle_identity is not None)
            if self._g_lifecycle_identity is not None:
                result["active_lifecycle_identity_record"] = self._g_lifecycle_identity.as_record()
            return result

    # The frozen G factory's compatibility gate accepts only the historical
    # private class-name prefix; the module identity and schema distinguish
    # this implementation from old G.  Keep the prefix while exposing the
    # lifecycle schema through metrics and the state contract.
    _ControlledLifetimeState.__name__ = "_ControlledHandoffState"
    _ControlledLifetimeState.__qualname__ = "_ControlledHandoffState"
    return _ControlledLifetimeState


__all__ = [
    "CallsiteBindingError",
    "ControlledLifetimeError",
    "ControlledLifetimeIdentity",
    "PROPAGATE_RELATIVE_PATH",
    "PROPAGATE_SOURCE_BLOB",
    "PROPAGATE_CANONICAL_LF_SHA256",
    "bind_propagator",
    "capability_type",
    "make_controlled_state_class",
    "source_binding",
]
