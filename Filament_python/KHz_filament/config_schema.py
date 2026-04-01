from __future__ import annotations

"""Configuration schema semantics for normalization/validation.

This module is intentionally lightweight: it centralizes field semantics,
legacy aliases, and key rule sets used by ``config_normalize``.
"""

from typing import Dict, Set


TOP_LEVEL_SECTIONS: Set[str] = {
    "grid",
    "beam",
    "propagation",
    "ionization",
    "heat",
    "run",
    "raman",
}


BEAM_DERIVED_FIELDS: Dict[str, str] = {
    "energy_J": "single-pulse energy [J]",
    "P0_peak": "peak power at pulse center [W]",
    "I0_peak": "legacy alias of peak power (historical misname) [W]",
    "E0_peak": "peak electric field amplitude [V/m]",
}


RATE_ALIAS_MAP: Dict[str, str] = {
    # historical shorthand
    "ppt_talebpour_i": "ppt_talebpour_i_lut",
    "ppt_talebpour_i_full": "ppt_talebpour_i_full_reference",
    "popruzhenko_atom_i": "popruzhenko_atom_i_lut",
    "popruzhenko_atom_i_full": "popruzhenko_atom_i_full_reference",
    # normalization helpers
    "none": "off",
    "zero": "off",
}


REMOVED_RATES: Set[str] = {
    "ppt_e",
    "ppt_i",
    "ppt_i_legacy",
    "adk_e",
    "powerlaw",
    "mpa",
}

