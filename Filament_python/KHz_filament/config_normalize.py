from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .constants import eps0, c0
from .config_schema import NONLINEAR_SWITCH_FIELDS, RATE_ALIAS_MAP, REMOVED_RATES, TRANSVERSE_PROFILE_TYPES


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


def _normalize_nonlinear_switches(propagation: Dict[str, Any]) -> None:
    """Validate optional Phase-2 propagation switches without inventing defaults."""
    for name in NONLINEAR_SWITCH_FIELDS:
        if name not in propagation or propagation[name] is None:
            continue
        if not isinstance(propagation[name], bool):
            raise ValueError(f"propagation.{name} must be true, false, or omitted for legacy compatibility.")


def _normalize_linear_precision(propagation: Dict[str, Any]) -> None:
    strategy = str(propagation.get("linear_precision_strategy", "baseline_complex64") or "baseline_complex64").lower()
    allowed = ("baseline_complex64", "orthonormal_fft", "mixed_precision", "unitary_projection")
    if strategy not in allowed:
        raise ValueError(f"propagation.linear_precision_strategy must be one of {allowed}.")
    propagation["linear_precision_strategy"] = strategy


def _normalize_heat(heat: Dict[str, Any]) -> None:
    """Validate explicit HR-3B parameters without touching legacy heat terms."""
    for name in ("rho0", "Cv", "f_rep", "D_gas", "gamma_heat"):
        if name in heat:
            heat[name] = _to_float(heat[name])
    if "hr3b_enabled" in heat and not isinstance(heat["hr3b_enabled"], bool):
        raise ValueError("heat.hr3b_enabled must be true or false.")
    for name in ("rho0", "Cv"):
        if name in heat and float(heat[name]) <= 0.0:
            raise ValueError(f"heat.{name} must be positive for HR-3B.")


def _normalize_raman(raman: Dict[str, Any]) -> None:
    """Validate the explicit Isaacs rotational-Raman parameterization.

    Validation intentionally runs on the raw mapping before dataclass defaults
    are applied.  The legacy ``rot_sinexp`` defaults are therefore untouched.
    """
    model = str(raman.get("model", "rot_sinexp") or "rot_sinexp").lower()
    convention = str(raman.get("operator_convention", "legacy") or "legacy").lower()
    if convention not in ("legacy", "isaacs_eq27"):
        raise ValueError("raman.operator_convention must be 'legacy' or 'isaacs_eq27'.")
    raman["operator_convention"] = convention
    sampling = str(raman.get("iir_sampling", "legacy_right_hold") or "legacy_right_hold").lower()
    allowed_sampling = ("legacy_right_hold", "left_hold", "trapezoidal", "exact_piecewise_linear")
    if sampling not in allowed_sampling:
        raise ValueError(f"raman.iir_sampling must be one of {allowed_sampling}.")
    raman["iir_sampling"] = sampling
    operator_mode = str(raman.get("operator_mode", "legacy_split") or "legacy_split").lower()
    allowed_modes = (
        "legacy_split", "split_energy_closed", "full_isaacs_eq27",
        "full_isaacs_eq27_complete", "historical_fr_mixture",
    )
    if operator_mode not in allowed_modes:
        raise ValueError(f"raman.operator_mode must be one of {allowed_modes}.")
    raman["operator_mode"] = operator_mode
    if operator_mode == "historical_fr_mixture":
        # The historical pre-April f_R mixture is a phase-only split operator.
        # It reuses the historical rot_sinexp kernel parameterization (T2/T_R)
        # and the legacy_right_hold IIR, while absorption stays on the current
        # comparison path (omega_R/Gamma_R/n_R) unchanged.
        if model != "rot_sinexp":
            raise ValueError("historical_fr_mixture requires raman.model='rot_sinexp'.")
        if raman.get("method", "iir") not in (None, "iir"):
            raise ValueError("historical_fr_mixture requires raman.method='iir'.")
        if sampling != "legacy_right_hold":
            raise ValueError("historical_fr_mixture requires raman.iir_sampling='legacy_right_hold'.")
        if raman.get("f_R") is None:
            raise ValueError("historical_fr_mixture requires raman.f_R.")
        f_R = float(_to_float(raman["f_R"]))
        if not (0.0 <= f_R < 1.0):
            raise ValueError("historical_fr_mixture requires 0 <= raman.f_R < 1.")
        for name in ("T2", "T_R"):
            if raman.get(name) is None:
                raise ValueError(f"historical_fr_mixture requires raman.{name}.")
            if float(_to_float(raman[name])) <= 0.0:
                raise ValueError(f"historical_fr_mixture requires raman.{name} > 0.")
    integrator = str(raman.get("operator_integrator", "heun") or "heun").lower()
    if integrator not in ("euler", "heun"):
        raise ValueError("raman.operator_integrator must be 'euler' or 'heun'.")
    raman["operator_integrator"] = integrator
    split_order = str(raman.get("nonlinear_split_order", "after_other") or "after_other").lower()
    if split_order not in ("after_other", "before_other", "strang"):
        raise ValueError("raman.nonlinear_split_order must be 'after_other', 'before_other', or 'strang'.")
    raman["nonlinear_split_order"] = split_order
    if model != "isaacs_rot_sinexp":
        return
    forbidden = ("f_R", "T_R", "T2", "Omega_R", "tau2")
    if any(raman.get(name) is not None for name in forbidden):
        raise ValueError(
            "isaacs_rot_sinexp uses explicit n_R, omega_R and Gamma_R.\n"
            "f_R/T_R/T2/Omega_R/tau2 must be omitted or null."
        )
    for name, lower, inclusive in (("n_R", 0.0, False), ("omega_R", 0.0, False), ("Gamma_R", 0.0, True)):
        if name not in raman or raman[name] is None:
            raise ValueError(f"raman.{name} is required for isaacs_rot_sinexp.")
        value = float(_to_float(raman[name]))
        valid = value >= lower if inclusive else value > lower
        if not valid:
            relation = ">=" if inclusive else ">"
            raise ValueError(f"raman.{name} must be {relation} {lower:g} for isaacs_rot_sinexp.")
        raman[name] = value


def _validate_raman_operator_coupling(raman: Dict[str, Any], propagation: Dict[str, Any]) -> None:
    mode = str(raman.get("operator_mode", "legacy_split") or "legacy_split").lower()
    legacy_absorption = bool(raman.get("absorption", False)) or bool(propagation.get("use_raman_absorption", False))
    absorption_model = str(raman.get("absorption_model", "") or "").lower()
    use_split_phase = bool(propagation.get("use_raman_phase", False))
    use_full_operator = bool(propagation.get("use_raman_full_operator", False))
    full_modes = ("full_isaacs_eq27", "full_isaacs_eq27_complete")
    if mode in full_modes and legacy_absorption:
        raise ValueError(
            f"{mode} includes Raman energy exchange and rejects legacy Raman absorption."
        )
    if mode in full_modes and use_split_phase:
        raise ValueError(
            f"{mode} rejects propagation.use_raman_phase=true; use use_raman_full_operator."
        )
    if mode not in full_modes and use_full_operator:
        raise ValueError(
            "propagation.use_raman_full_operator=true requires raman.operator_mode="
            "full_isaacs_eq27 or full_isaacs_eq27_complete."
        )
    if mode == "split_energy_closed" and (legacy_absorption or absorption_model == "conv_deriv"):
        raise ValueError("split_energy_closed rejects legacy conv_deriv/absorption feedback.")


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw config dict into a single canonical representation."""
    out = deepcopy(raw or {})
    out["grid"] = dict(out.get("grid", {}))
    out["beam"] = dict(out.get("beam", {}))
    out["propagation"] = dict(out.get("propagation", {}))
    out["ionization"] = dict(out.get("ionization", {}))
    out["heat"] = dict(out.get("heat", {}))
    out["raman"] = dict(out.get("raman", {}))

    _normalize_beam(out["beam"], grid=out["grid"])
    _normalize_species(out["ionization"])
    _normalize_nonlinear_switches(out["propagation"])
    _normalize_linear_precision(out["propagation"])
    _normalize_heat(out["heat"])
    _normalize_raman(out["raman"])
    _validate_raman_operator_coupling(out["raman"], out["propagation"])
    return out
