"""Predeclared scalar residual-cost units for the S4-6I measurement seam."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


REGISTRY_SCHEMA = "filament.cpu.s4_6i.residual_cost.registry.v1"
SCOPE_LABEL = "NONLINEAR_ARRAYS"
SCOPE_HELPER = "__cpu_s46i_cost_scope__"


@dataclass(frozen=True)
class CostUnit:
    name: str
    source_order: int
    context: str
    line_start: int
    line_end: int
    source_site: str
    selector: str
    eligible: bool = True
    scope_kind: str = "eligible_scope"
    branch_sensitive: bool = False


# These spans are source-bound anchors in the frozen propagate.py source.  The
# transform matches both the span and the named call/assignment selector, so a
# shifted or duplicated source fails closed instead of silently moving a cost
# scope to a different expression.
COST_UNITS: tuple[CostUnit, ...] = (
    CostUnit("prep_intensity", 10, "PREP", 690, 690, "propagate.py:690:inten_ion", "inten_ion"),
    CostUnit("photo_source_fallback", 20, "PREP", 790, 793, "propagate.py:790-793:photo_source_fallback", "photo_source_fallback", True, "eligible_scope", True),
    CostUnit("photo_source_cleanup", 30, "PREP", 797, 803, "propagate.py:797-803:photo_source_cleanup", "photo_source_cleanup"),
    CostUnit("ib_alpha", 40, "PREP", 814, 815, "propagate.py:814-815:ib_alpha", "ib_alpha"),
    CostUnit("alpha_ion_raw", 50, "PREP", 819, 819, "propagate.py:819:alpha_ion_raw", "alpha_ion_raw"),
    CostUnit("alpha_ion_applied_cleanup", 55, "PREP", 830, 833, "propagate.py:830-833:alpha_ion_applied_cleanup", "alpha_ion_applied_cleanup"),
    CostUnit("alpha_phase_assembly", 60, "PREP", 945, 955, "propagate.py:945-955:alpha_phase_assembly", "alpha_phase_assembly"),
    CostUnit("direct_ib_power", 70, "DEPOSITION_AND_QACC", 1118, 1118, "propagate.py:1118:direct_ib_power", "direct_ib_power"),
    CostUnit("q_ib", 80, "DEPOSITION_AND_QACC", 1119, 1119, "propagate.py:1119:q_ib:context_only", "q_ib", False, "context_only"),
    CostUnit("heat_q_per_z", 90, "HEAT_QACC", 1222, 1231, "propagate.py:1222-1231:heat_Q_per_z:context_only", "heat_q_per_z", False, "context_only"),
)

POINTWISE_HELPER_UNITS = (
    {
        "name": "q_ib_multiplication",
        "source_order": 81,
        "context": "DEPOSITION_AND_QACC",
        "source_site": "deposition.py:50-57:q_ib_multiplication",
        "eligible": True,
        "scope_kind": "eligible_scope",
    },
    {
        "name": "heat_q_pointwise",
        "source_order": 91,
        "context": "HEAT_QACC",
        "source_site": "heat.py:25-35:heat_Q_per_z",
        "eligible": True,
        "scope_kind": "eligible_scope",
    },
)

POWER_VOLUME_UNIT = {
    "name": "deposition_power_volume",
    "source_order": 100,
    "context": "DEPOSITION_AND_QACC",
    "source_site": "deposition.py:23:_power_volume",
    "contexts": ("q_ion", "q_ib", "direct_ion", "direct_ib"),
    "eligible": True,
    "scope_kind": "eligible_scope",
}

# Context and reduction clocks are excluded from candidate ranking. Their
# original operations remain independent of the eligible pointwise scopes.
AUXILIARY_UNITS = (
    CostUnit("initial_intensity", 110, "INITIALIZATION", 489, 489, "propagate.py:489:intensity_initial", "aux:intensity", False, "outside_body"),
    CostUnit("ion_recursion", 120, "PREP", 767, 773, "propagate.py:767-773:ion_recursion", "aux:evolve_rho_time", False, "noncandidate"),
    CostUnit("ion_operator_correction", 130, "PREP", 821, 827, "propagate.py:821-827:operator_correction", "aux:operator_correct_scalar", False, "noncandidate"),
    CostUnit("qmap_energy_reductions", 140, "DEPOSITION", 1122, 1128, "propagate.py:1122-1128:qmap_energy", "aux:interval_energy_from_q", False, "noncandidate"),
    CostUnit("deposition_ledger", 150, "DEPOSITION", 1135, 1153, "propagate.py:1135-1153:deposition_ledger", "aux:ledger", False, "noncandidate"),
    CostUnit("thermalization_and_sink", 160, "HEAT_QACC", 1157, 1182, "propagate.py:1157-1182:thermalization_and_sink", "aux:thermalize_interval", False, "noncandidate"),
    CostUnit("slow_state_and_POST", 170, "HEAT_QACC", 1183, 1217, "propagate.py:1183-1217:slow_state_and_POST", "aux:slow_state", False, "noncandidate"),
    CostUnit("qacc_commit", 180, "HEAT_QACC", 1232, 1232, "propagate.py:1232:Qacc_commit", "aux:Qacc", False, "noncandidate"),
    CostUnit("qacc_raman_commit", 190, "HEAT_QACC", 1235, 1235, "propagate.py:1235:Qacc_raman_commit", "aux:Qacc", False, "noncandidate"),
    CostUnit("saved_intensity", 200, "SAVED_DIAGNOSTICS", 1294, 1294, "propagate.py:1294:intensity_saved", "aux:intensity", False, "noncandidate"),
)

REDUCTION_UNITS = tuple(
    dict(name=name, source_site=site, source_order=order, context="REDUCTION", eligible=False,
         scope_kind="noncandidate", **({"contexts": contexts} if contexts else {}))
    for name, site, order, contexts in (
        ("deposition_time_sum", "deposition.py:42:time_sum", 210, ("q_ion", "q_ib")),
        ("deposition_direct_sum", "deposition.py:131:independent_3d_sum", 220, ("direct_ion", "direct_ib")),
        ("heat_time_sum", "heat.py:36:time_sum", 230, ()),
    )
)

PARENT_LABELS = (
    "PROP_ENTRY_INIT", "INTERVAL_PREP_AND_OTHER_ARRAYS", "INTERVAL_ENERGY_BOOKKEEPING",
    "INTERVAL_DEPOSITION_AND_QACC", "INTERVAL_POST_COMMIT", "INTERVAL_SAVED_DIAGNOSTICS",
    "PROP_FINALIZE_EXIT",
)


def validate_registry() -> None:
    names = [unit.name for unit in COST_UNITS]
    orders = [unit.source_order for unit in COST_UNITS]
    if len(set(names)) != len(names) or len(set(orders)) != len(orders):
        raise RuntimeError("S46I_COST_REGISTRY_NOT_DISJOINT")
    if orders != sorted(orders):
        raise RuntimeError("S46I_COST_REGISTRY_SOURCE_ORDER")
    spans = sorted((unit.line_start, unit.line_end, unit.name) for unit in COST_UNITS)
    for left, right in zip(spans, spans[1:]):
        if left[1] >= right[0]:
            raise RuntimeError("S46I_COST_REGISTRY_SPAN_OVERLAP")


validate_registry()


def registry_snapshot() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "scope_label": SCOPE_LABEL,
        "scope_helper": SCOPE_HELPER,
        "units": tuple(asdict(unit) for unit in COST_UNITS),
        "power_volume": dict(POWER_VOLUME_UNIT),
        "pointwise_helper_units": tuple(dict(row) for row in POINTWISE_HELPER_UNITS),
        "noncandidate_units": tuple(asdict(unit) for unit in AUXILIARY_UNITS) + REDUCTION_UNITS,
        "common_parent_labels": PARENT_LABELS,
        "eligible_units": tuple(
            row["name"] for row in (
                tuple(asdict(unit) for unit in COST_UNITS)
                + POINTWISE_HELPER_UNITS
                + (POWER_VOLUME_UNIT,)
            ) if bool(row.get("eligible", False))
        ),
        "pure_pointwise_sites": {
            "propagate": tuple(
                asdict(unit)["name"] for unit in COST_UNITS if unit.eligible
            ),
            "deposition_power_volume": {
                "source_site": POWER_VOLUME_UNIT["source_site"],
                "contexts": tuple(POWER_VOLUME_UNIT["contexts"]),
            },
            "deposition_q_ib": "deposition.py:50-57:q_ib_multiplication",
            "heat": "heat.py:25-35:heat_Q_per_z",
            "outer_context_only": ("q_ib", "heat_q_per_z"),
        },
        "disjoint_exclusive_ranking": True,
        "scientific_arrays_retained": False,
    }


__all__ = [
    "COST_UNITS",
    "POWER_VOLUME_UNIT",
    "POINTWISE_HELPER_UNITS",
    "REGISTRY_SCHEMA",
    "SCOPE_HELPER",
    "SCOPE_LABEL",
    "CostUnit",
    "registry_snapshot",
    "validate_registry",
]
