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
    "E0_peak": "peak electric field amplitude [V/m]",
}


TRANSVERSE_PROFILE_TYPES: Set[str] = {
    "gaussian",
    "flat_top_cosine",
    "super_gaussian",
}


NONLINEAR_SWITCH_FIELDS: Set[str] = {
    "use_electronic_kerr",
    "use_raman_phase",
    "use_raman_full_operator",
    "use_plasma_phase",
    "use_ionization_loss",
    "use_raman_absorption",
    "use_ionization_solver",
}


HEAT_HR3B_FIELDS: Dict[str, str] = {
    "hr3b_enabled": "enable authoritative HR-3B post-acoustic delta_n_th state",
    "rho0": "ambient dry-air mass density [kg/m^3]",
    "Cv": "dry-air constant-volume specific heat [J/(kg K)]",
}


HEAT_HR3C_FIELDS: Dict[str, str] = {
    "D_th": "authoritative HR-3C transverse thermal diffusivity [m^2/s]",
    "hr3c_enabled": "enable transactional HR-3C state lifecycle",
    "hr3c_batch_intervals": "HR-3C disk streaming batch size [intervals]",
    "resume_hr3c": "explicitly resume an existing HR-3C state manifest",
}


HEAT_HR4_FIELDS: Dict[str, str] = {
    "hr4_enabled": "enable HR-4A contract validation only; no runner integration",
    "chi": "must equal authoritative HR-3C D_th [m^2/s]",
    "nu": "frozen HR-4 kinematic viscosity [m^2/s]",
    "gravity_x": "frozen x gravity component [m/s^2]",
    "gravity_y": "frozen y gravity component [m/s^2]",
    "x_min": "frozen HR-4 transverse x lower bound [m]",
    "x_max": "frozen HR-4 transverse x upper bound [m]",
    "y_min": "frozen HR-4 transverse y lower bound [m]",
    "y_max": "frozen HR-4 transverse y upper bound [m]",
    "dx": "provisional HR-4 transverse spacing [m]",
    "dy": "provisional HR-4 transverse spacing [m]",
    "dt_hydro": "provisional HR-4 fixed hydro timestep [s]",
    "advection_scheme": "frozen first-order local upwind",
    "diffusion_scheme": "frozen explicit central finite difference",
    "time_integrator": "frozen explicit unsplit Forward Euler",
    "grid_layout": "frozen collocated three-field layout",
    "boundary_delta_n": "frozen ambient Dirichlet zero boundary",
    "boundary_velocity": "frozen open outflow / ambient inflow boundary",
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
