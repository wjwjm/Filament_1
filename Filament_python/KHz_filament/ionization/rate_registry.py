from __future__ import annotations

REMOVED_RATES = {"ppt_e", "ppt_i", "ppt_i_legacy", "adk_e", "powerlaw", "mpa"}

RATE_ALIAS_MAP = {
    "ppt_talebpour_i": "ppt_talebpour_i_lut",
    "ppt_talebpour_i_full": "ppt_talebpour_i_full_reference",
    "popruzhenko_atom_i": "popruzhenko_atom_i_lut",
    "popruzhenko_atom_i_full": "popruzhenko_atom_i_full_reference",
}

SUPPORTED_RATES = {
    "mpa_fact",
    "ppt_talebpour_i_legacy",
    "ppt_talebpour_i_full_reference",
    "ppt_talebpour_i_lut",
    "popruzhenko_atom_i_legacy",
    "popruzhenko_atom_i_full_reference",
    "popruzhenko_atom_i_lut",
    "off",
}

LEGACY_MODEL_REMOVED = {"ppt", "ppt_cycleavg", "adk", "powerlaw", "mpa"}
