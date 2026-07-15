from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .constants import eps0, c0
from .config_schema import RATE_ALIAS_MAP, REMOVED_RATES, TRANSVERSE_PROFILE_TYPES


def E0_from_energy(U: float, w0: float, tau_fwhm: float, n0: float) -> float:
    import math
    tau = tau_fwhm / math.sqrt(2.0 * math.log(2.0))
    space = math.pi * w0**2 / 2.0
    time = math.sqrt(math.pi / 2.0) * tau
    pref = 0.5 * eps0 * c0 * n0
    return float((U / (pref * space * time)) ** 0.5)


def E0_from_peak_power(P0_peak: float, w0: float, n0: float) -> float:
    import math
    pref = 0.5 * eps0 * c0 * n0
    area_eff = math.pi * (w0 ** 2) / 2.0
    return float((P0_peak / (pref * area_eff)) ** 0.5)


def _to_float(v: Any) -> Any:
    try:
        return float(v)
    except Exception:
        return v


def _normalize_beam(beam: Dict[str, Any], *, grid: Dict[str, Any]) -> None:
    for k in ("w0", "tau_fwhm", "n0", "energy_J", "P0_peak", "E0_peak"):
        if k in beam:
            beam[k] = _to_float(beam[k])

    if beam.get("I0_peak", None) is not None:
        raise ValueError("beam.I0_peak has been removed; please use beam.P0_peak instead.")

    profile = beam.get("transverse_profile", None)
    if profile is not None:
        if not isinstance(profile, dict):
            raise ValueError("beam.transverse_profile must be an object when provided.")
        canonical = dict(profile)
        profile_type = str(canonical.get("type", "")).strip().lower()
        if profile_type not in TRANSVERSE_PROFILE_TYPES:
            allowed = ", ".join(sorted(TRANSVERSE_PROFILE_TYPES))
            raise ValueError(f"unknown beam.transverse_profile.type={profile_type!r}; allowed: {allowed}")
        if "radius_m" not in canonical:
            raise ValueError("beam.transverse_profile.radius_m is required when a profile is provided.")
        canonical["type"] = profile_type
        canonical["radius_m"] = float(_to_float(canonical["radius_m"]))
        if canonical["radius_m"] <= 0.0:
            raise ValueError("beam.transverse_profile.radius_m must be positive.")
        if profile_type == "flat_top_cosine":
            if "edge_start_fraction" not in canonical:
                raise ValueError("flat_top_cosine requires beam.transverse_profile.edge_start_fraction.")
            canonical["edge_start_fraction"] = float(_to_float(canonical["edge_start_fraction"]))
            if not 0.0 < canonical["edge_start_fraction"] < 1.0:
                raise ValueError("flat_top_cosine edge_start_fraction must satisfy 0 < value < 1.")
        if profile_type == "super_gaussian" and "order" in canonical:
            canonical["order"] = float(_to_float(canonical["order"]))
            if canonical["order"] <= 0.0:
                raise ValueError("super_gaussian order must be positive.")
        beam["transverse_profile"] = canonical

    has_energy = beam.get("energy_J", None) is not None
    has_p0 = beam.get("P0_peak", None) is not None
    active_count = int(has_energy) + int(has_p0)
    if active_count > 1:
        raise ValueError("beam.energy_J / beam.P0_peak are mutually exclusive; keep only one.")
    if has_energy and float(beam["energy_J"]) <= 0.0:
        raise ValueError("beam.energy_J must be positive when provided.")
    if has_p0 and float(beam["P0_peak"]) <= 0.0:
        raise ValueError("beam.P0_peak must be positive when provided.")

    if "Twin" not in grid:
        tau_fwhm = beam.get("tau_fwhm", None)
        if tau_fwhm is not None:
            grid["Twin"] = 8.0 * float(tau_fwhm)

    # Derived sources are normalized only after the discrete transverse field
    # has been built.  This avoids applying the Gaussian analytic area to a
    # non-Gaussian profile, and makes P0_peak exact on the actual grid.
    if has_p0:
        beam["E0_peak"] = 0.0
        beam["_norm_source"] = "P0_peak"
    elif has_energy:
        beam["E0_peak"] = 0.0
        beam["_norm_source"] = "energy_J"
    elif float(beam.get("E0_peak", 0.0) or 0.0) > 0.0:
        beam["_norm_source"] = "E0_peak_direct"


def _normalize_species(ion: Dict[str, Any]) -> None:
    species = ion.get("species", None)
    if not isinstance(species, list):
        return

    total = 0.0
    for sp in species:
        if not isinstance(sp, dict):
            continue
        frac = max(0.0, float(_to_float(sp.get("fraction", 1.0))))
        sp["fraction"] = frac
        total += frac

        rate_raw = str(sp.get("rate", "") or "").strip().lower().replace("ppt-i", "ppt_i")
        if rate_raw in REMOVED_RATES:
            raise ValueError(
                f"[ionization] species.rate='{rate_raw}' removed; use ppt_talebpour_i_legacy / "
                "ppt_talebpour_i_full_reference / ppt_talebpour_i_lut / "
                "popruzhenko_atom_i_full_reference / popruzhenko_atom_i_lut / mpa_fact / off"
            )
        if rate_raw:
            sp["rate"] = RATE_ALIAS_MAP.get(rate_raw, rate_raw)

    if total > 0.0:
        for sp in species:
            if isinstance(sp, dict):
                sp["fraction"] = float(sp["fraction"] / total)


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw config dict into a single canonical representation."""
    out = deepcopy(raw or {})
    out["grid"] = dict(out.get("grid", {}))
    out["beam"] = dict(out.get("beam", {}))
    out["ionization"] = dict(out.get("ionization", {}))

    _normalize_beam(out["beam"], grid=out["grid"])
    _normalize_species(out["ionization"])
    return out
