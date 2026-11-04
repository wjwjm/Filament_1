"""Fixed source sites for the S4-6K five-chain scalar cost overlay.

The registry is deliberately smaller than the S4-6I inventory.  The alpha
and power kernels already have parallel implementations owned by the 6J
hook, so this list contains only the remaining pure-pointwise work plus
context-only phase selectors/parents needed to prove clock closure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


REGISTRY_SCHEMA = "filament.cpu.s4_6k.five_chain_cost.registry.v1"
SCOPE_LABEL = "NONLINEAR_ARRAYS"
SCOPE_HELPER = "__cpu_s46k_cost_scope__"


@dataclass(frozen=True)
class CostSite:
    name: str
    unit: str
    source_site: str
    source_order: int
    eligible: bool
    scope_kind: str
    parent: bool = False
    branch_sensitive: bool = False
    expected_calls_per_interval: int | None = None
    phase_related: bool = False


# ``source_order`` is the fixed report order, not an execution-time counter.
# Four phase children are eligible arithmetic children.  The selector/zero
# branches and scalar alpha_R assignment remain separately visible but are not
# ranked.  The phase parent is context-only and never added to a candidate.
SITES: tuple[CostSite, ...] = (
    CostSite(
        "prep_intensity",
        "prep_intensity",
        "propagate.py:690:inten_ion",
        10,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
    ),
    CostSite(
        "photo_source_fallback",
        "photo_source_fallback",
        "propagate.py:790-793:photo_source_fallback",
        20,
        True,
        "eligible_scope",
        branch_sensitive=True,
    ),
    CostSite(
        "photo_source_cleanup",
        "photo_source_cleanup",
        "propagate.py:797-803:photo_source_cleanup",
        30,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
    ),
    CostSite(
        "ib_alpha",
        "ib_alpha",
        "propagate.py:814-815:ib_alpha",
        40,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
    ),
    CostSite(
        "alpha_ion_raw",
        "alpha_ion_raw",
        "propagate.py:819:alpha_ion_raw",
        50,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
    ),
    CostSite(
        "alpha_phase_assembly",
        "alpha_phase_assembly",
        "propagate.py:947-955:alpha_phase_assembly",
        60,
        False,
        "context_only",
        parent=True,
        expected_calls_per_interval=1,
        phase_related=True,
    ),
    CostSite(
        "alpha_ion_applied",
        "alpha_phase_assembly",
        "propagate.py:947:alpha_ion_applied",
        61,
        False,
        "context_only",
        expected_calls_per_interval=1,
        phase_related=True,
    ),
    CostSite(
        "alpha_R",
        "alpha_phase_assembly",
        "propagate.py:948:alpha_R",
        62,
        False,
        "context_only",
        expected_calls_per_interval=1,
        phase_related=True,
    ),
    CostSite(
        "alpha_total",
        "alpha_phase_assembly",
        "propagate.py:949:alpha_total",
        63,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
        phase_related=True,
    ),
    CostSite(
        "kerr",
        "alpha_phase_assembly",
        "propagate.py:952:kerr",
        64,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
        phase_related=True,
    ),
    CostSite(
        "rawplasma",
        "alpha_phase_assembly",
        "propagate.py:953:rawplasma",
        65,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
        phase_related=True,
    ),
    CostSite(
        "plasma",
        "alpha_phase_assembly",
        "propagate.py:954:plasma",
        66,
        False,
        "context_only",
        expected_calls_per_interval=1,
        phase_related=True,
    ),
    CostSite(
        "phase",
        "alpha_phase_assembly",
        "propagate.py:955:phase",
        67,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
        phase_related=True,
    ),
    CostSite(
        "direct_ib_power",
        "direct_ib_power",
        "propagate.py:1118:direct_ib_power",
        70,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
    ),
    CostSite(
        "q_ib_multiplication",
        "q_ib_multiplication",
        "deposition.py:50-57:q_ib_multiplication",
        80,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
    ),
    CostSite(
        "heat_q_pointwise",
        "heat_q_pointwise",
        "heat.py:25-35:heat_Q_per_z",
        90,
        True,
        "eligible_scope",
        expected_calls_per_interval=1,
    ),
)


def validate_registry() -> None:
    names = [site.name for site in SITES]
    source_sites = [site.source_site for site in SITES]
    orders = [site.source_order for site in SITES]
    if len(set(names)) != len(names):
        raise RuntimeError("S46K_COST_REGISTRY_DUPLICATE_NAME")
    if len(set(source_sites)) != len(source_sites):
        raise RuntimeError("S46K_COST_REGISTRY_DUPLICATE_SOURCE_SITE")
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise RuntimeError("S46K_COST_REGISTRY_SOURCE_ORDER")
    phase_children = [site for site in SITES if site.unit == "alpha_phase_assembly" and not site.parent]
    if [site.name for site in phase_children if site.eligible] != [
        "alpha_total",
        "kerr",
        "rawplasma",
        "phase",
    ]:
        raise RuntimeError("S46K_PHASE_ELIGIBLE_CHILDREN")
    if sum(site.parent for site in SITES if site.unit == "alpha_phase_assembly") != 1:
        raise RuntimeError("S46K_PHASE_PARENT_COUNT")


validate_registry()


def registry_snapshot() -> dict[str, Any]:
    """Return a JSON-safe, stable inventory consumed by cost analysis."""

    sites = [asdict(site) for site in SITES]
    return {
        "schema": REGISTRY_SCHEMA,
        "scope_label": SCOPE_LABEL,
        "scope_helper": SCOPE_HELPER,
        "sites": sites,
        "eligible_sites": [site["name"] for site in sites if site["eligible"]],
        "parallel_excluded": {
            "alpha_ion_applied_cleanup": "propagate.py:830-833:alpha_ion",
            "power_volume": "owned by AlphaParallelHook/PowerVolumeHook",
        },
        "phase_ranking": {
            "parent": "alpha_phase_assembly",
            "eligible_children": [
                site["name"]
                for site in sites
                if site["unit"] == "alpha_phase_assembly"
                and site["eligible"]
                and not site["parent"]
            ],
            "selectors_noneligible": [
                site["name"]
                for site in sites
                if site["unit"] == "alpha_phase_assembly"
                and not site["eligible"]
                and not site["parent"]
            ],
        },
        "scientific_arrays_retained": False,
        "common_scope": True,
        "profile": False,
    }


__all__ = [
    "CostSite",
    "REGISTRY_SCHEMA",
    "SCOPE_HELPER",
    "SCOPE_LABEL",
    "SITES",
    "registry_snapshot",
    "validate_registry",
]
