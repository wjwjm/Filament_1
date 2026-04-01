from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as _np

from ..device import xp
from .common import ION_LUT_SCHEMA_VERSION, W_CAP_DEFAULT, _ION_LUT_MEMORY_CACHE, _as_real_like, _to_numpy
from .models_ppt import cycle_average_ppt_talebpour_full_from_I
from .models_popruzhenko import cycle_average_popruzhenko_atom_full_from_I


def _ion_rate_table_defaults(ion_conf):
    cfg = dict(getattr(ion_conf, "rate_table", None) or {})
    cfg.setdefault("enabled", True)
    cfg.setdefault("reuse_cache", True)
    cfg.setdefault("cache_dir", "cache/rate_tables")
    cfg.setdefault("rebuild_if_missing", True)
    cfg.setdefault("force_rebuild", False)
    cfg.setdefault("save_tables", True)
    cfg.setdefault("I_min_SI", 1e8)
    cfg.setdefault("I_max_SI", 1e19)
    cfg.setdefault("n_samples", 3000)
    cfg.setdefault("spacing", "log")
    cfg.setdefault("interp_mode", "loglog")
    cfg.setdefault("ref_cycle_avg_samples", 64)
    cfg.setdefault("popruzhenko_sum_tol", 1e-6)
    cfg.setdefault("popruzhenko_max_terms", 256)
    cfg.setdefault("max_rel_error_target", 0.03)
    cfg.setdefault("filament_I_min_SI", 1e13)
    cfg.setdefault("filament_I_max_SI", 1e15)
    return cfg


def _canonical_table_metadata(model_name, species_name, species_params, omega0_SI, n0, table_cfg):
    keys = ("Ip_eV", "Ip_eV_eff", "Z", "Zeff", "l", "m", "rate", "reference_model")
    species_meta = {k: species_params.get(k) for k in keys if k in species_params}
    return {
        "schema_version": ION_LUT_SCHEMA_VERSION,
        "model_name": str(model_name),
        "species_name": str(species_name),
        "species_params": species_meta,
        "omega0_SI": float(omega0_SI),
        "n0": float(n0),
        "I_min_SI": float(table_cfg["I_min_SI"]),
        "I_max_SI": float(table_cfg["I_max_SI"]),
        "n_samples": int(table_cfg["n_samples"]),
        "spacing": str(table_cfg["spacing"]),
        "interp_mode": str(table_cfg["interp_mode"]),
        "ref_cycle_avg_samples": int(table_cfg["ref_cycle_avg_samples"]),
        "popruzhenko_sum_tol": float(table_cfg["popruzhenko_sum_tol"]),
        "popruzhenko_max_terms": int(table_cfg["popruzhenko_max_terms"]),
    }


def _table_signature(metadata: dict) -> str:
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _table_path(cache_dir: str, species_name: str, signature: str) -> str:
    safe_species = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(species_name))
    return os.path.join(cache_dir, f"ionlut_{safe_species}_{signature[:16]}.npz")


def _get_reference_evaluator(model_name: str, species_params: dict, omega0_SI: float, n0: float, reference_opts: dict):
    model = str(model_name).lower()
    ref_samples = int(reference_opts.get("cycle_avg_samples_ref", reference_opts.get("ref_cycle_avg_samples", 64)))
    pop_tol = float(reference_opts.get("sum_rel_tol", reference_opts.get("popruzhenko_sum_tol", 1e-6)))
    pop_max_terms = int(reference_opts.get("max_terms", reference_opts.get("popruzhenko_max_terms", 256)))
    W_cap = float(reference_opts.get("W_cap", W_CAP_DEFAULT))

    if model == "ppt_talebpour_i_full_reference":
        from .runtime import _talebpour_defaults

        name = str(species_params.get("name", ""))
        Ip_use, Zeff_use = _talebpour_defaults(name=name, Ip_eV=species_params.get("Ip_eV"), Ip_eV_eff=species_params.get("Ip_eV_eff"), Zeff=species_params.get("Zeff"))
        l = int(species_params.get("l", 0))
        m = int(species_params.get("m", 0))
        return lambda I_SI: cycle_average_ppt_talebpour_full_from_I(I_SI, n0=n0, omega0_SI=omega0_SI, Ip_eV=Ip_use, Zeff=Zeff_use, l=l, m=m, samples=ref_samples, max_terms=pop_max_terms, sum_rel_tol=pop_tol, W_cap=W_cap)
    if model == "popruzhenko_atom_i_full_reference":
        Ip = float(species_params.get("Ip_eV", 15.6))
        Z = int(species_params.get("Z", 1))
        return lambda I_SI: cycle_average_popruzhenko_atom_full_from_I(I_SI, n0=n0, omega0_SI=omega0_SI, Ip_eV=Ip, Z=Z, samples=ref_samples, max_terms=pop_max_terms, sum_rel_tol=pop_tol, W_cap=W_cap)
    raise ValueError(f"[ionization] unsupported reference model for LUT build: {model_name}")


def build_rate_table(model_name, species_params, omega0_SI, n0, I_min_SI, I_max_SI, n_samples, spacing="log", reference_opts=None):
    reference_opts = dict(reference_opts or {})
    if str(spacing).lower() == "log":
        I_grid = _np.logspace(_np.log10(float(I_min_SI)), _np.log10(float(I_max_SI)), int(n_samples))
    else:
        I_grid = _np.linspace(float(I_min_SI), float(I_max_SI), int(n_samples))
    ref_eval = _get_reference_evaluator(model_name, species_params, omega0_SI, n0, reference_opts)
    W_grid = _to_numpy(ref_eval(xp.asarray(I_grid))).astype(_np.float64, copy=False)
    W_grid = _np.nan_to_num(_np.clip(W_grid, 0.0, float(reference_opts.get("W_cap", W_CAP_DEFAULT))), nan=0.0)
    return {
        "I_grid": I_grid.astype(_np.float64, copy=False),
        "W_grid": W_grid.astype(_np.float64, copy=False),
        "model_name": str(model_name),
        "species_name": str(species_params.get("name", "species")),
        "omega0_SI": float(omega0_SI),
        "n0": float(n0),
        "build_opts": dict(reference_opts),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": ION_LUT_SCHEMA_VERSION,
    }


def eval_rate_from_table(I_SI, table, method="loglog"):
    method = str(method or "loglog").lower()
    I_grid = xp.asarray(table["I_grid"])
    W_grid = xp.asarray(table["W_grid"])
    I_in = xp.asarray(I_SI)
    I_min = float(_to_numpy(I_grid[0]))
    I_max = float(_to_numpy(I_grid[-1]))
    I_clip = xp.clip(_as_real_like(I_in), max(I_min, 1e-300), max(I_max, I_min))
    logI = xp.log(I_clip)
    logI_grid = xp.log(I_grid)
    if method in ("loglog", "log-linear-logw", "logi-logw"):
        eps_w = 1e-300
        logW_grid = xp.log(xp.maximum(W_grid, eps_w))
        logW = xp.interp(logI, logI_grid, logW_grid)
        W = xp.exp(logW)
        W = xp.where(I_in <= 0.0, xp.asarray(0.0, dtype=W.dtype), W)
        return xp.nan_to_num(W, nan=0.0, posinf=float(W_CAP_DEFAULT), neginf=0.0)
    if method in ("loglinear", "logi-linearw"):
        W = xp.interp(logI, logI_grid, W_grid)
        W = xp.where(I_in <= 0.0, xp.asarray(0.0, dtype=W.dtype), W)
        return xp.nan_to_num(xp.clip(W, 0.0, None), nan=0.0, posinf=float(W_CAP_DEFAULT), neginf=0.0)
    raise ValueError(f"Unknown LUT interpolation method: {method}")


def validate_rate_table(reference_eval, lut_eval, I_test_grid, filament_I_min_SI=1e13, filament_I_max_SI=1e15):
    I_test = xp.asarray(I_test_grid)
    W_ref = _as_real_like(reference_eval(I_test))
    W_lut = _as_real_like(lut_eval(I_test))
    denom = xp.maximum(xp.abs(W_ref), 1e-30)
    rel = _as_real_like(xp.abs(W_lut - W_ref) / denom)
    hi_mask = I_test >= (0.5 * float(_to_numpy(I_test[-1])))
    fil_mask = (I_test >= float(filament_I_min_SI)) & (I_test <= float(filament_I_max_SI))

    def _stat(mask):
        if bool(xp.any(mask)):
            vals = rel[mask]
            return float(_to_numpy(xp.max(vals))), float(_to_numpy(xp.median(vals)))
        return 0.0, 0.0

    hi_max, hi_median = _stat(hi_mask)
    fil_max, fil_median = _stat(fil_mask)
    return {
        "max_relative_error": float(_to_numpy(xp.max(rel))),
        "median_relative_error": float(_to_numpy(xp.median(rel))),
        "high_intensity_max_relative_error": hi_max,
        "high_intensity_median_relative_error": hi_median,
        "filament_max_relative_error": fil_max,
        "filament_median_relative_error": fil_median,
    }


def _load_table_npz(path: str):
    data = _np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"].item()))
    return {
        "I_grid": _np.asarray(data["I_grid"], dtype=_np.float64),
        "W_grid": _np.asarray(data["W_grid"], dtype=_np.float64),
        "metadata": metadata,
        "model_name": str(data["model_name"].item()),
        "species_name": str(data["species_name"].item()),
        "build_timestamp": str(data["build_timestamp"].item()),
        "schema_version": str(data["schema_version"].item()),
        "validation": json.loads(str(data["validation_json"].item())) if "validation_json" in data.files else {},
    }


def _save_table_npz(path: str, table: dict, metadata: dict, validation: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _np.savez_compressed(
        path,
        I_grid=_np.asarray(table["I_grid"], dtype=_np.float64),
        W_grid=_np.asarray(table["W_grid"], dtype=_np.float64),
        metadata_json=json.dumps(metadata, sort_keys=True, ensure_ascii=False),
        model_name=str(table["model_name"]),
        species_name=str(table["species_name"]),
        build_timestamp=str(table.get("build_timestamp", datetime.now(timezone.utc).isoformat())),
        schema_version=str(table.get("schema_version", ION_LUT_SCHEMA_VERSION)),
        validation_json=json.dumps(validation, sort_keys=True, ensure_ascii=False),
    )


def prepare_ionization_lut_for_species(species_params, omega0_SI: float, n0: float, rate_table_cfg: dict):
    rate = str(species_params.get("rate", "")).lower()
    reference_model = str(species_params.get("reference_model", "")).lower()
    if rate == "ppt_talebpour_i_lut" and not reference_model:
        reference_model = "ppt_talebpour_i_full_reference"
    if rate == "popruzhenko_atom_i_lut" and not reference_model:
        reference_model = "popruzhenko_atom_i_full_reference"
    if reference_model not in ("ppt_talebpour_i_full_reference", "popruzhenko_atom_i_full_reference"):
        raise ValueError(f"[ionization] LUT rate requires valid reference_model, got: {reference_model}")

    metadata = _canonical_table_metadata(reference_model, str(species_params.get("name", "species")), species_params, omega0_SI, n0, rate_table_cfg)
    signature = _table_signature(metadata)
    cache_dir = str(rate_table_cfg.get("cache_dir", "cache/rate_tables"))
    path = _table_path(cache_dir, metadata["species_name"], signature)
    mem_key = (path, signature)
    if bool(rate_table_cfg.get("reuse_cache", True)) and mem_key in _ION_LUT_MEMORY_CACHE and not bool(rate_table_cfg.get("force_rebuild", False)):
        return _ION_LUT_MEMORY_CACHE[mem_key]

    hit = False
    cache_reason = ""
    if bool(rate_table_cfg.get("reuse_cache", True)) and os.path.exists(path) and not bool(rate_table_cfg.get("force_rebuild", False)):
        loaded = _load_table_npz(path)
        loaded_meta = loaded.get("metadata", {})
        if loaded_meta == metadata:
            table = loaded
            hit = True
        else:
            mismatch = [k for k in sorted(metadata.keys()) if loaded_meta.get(k) != metadata.get(k)]
            cache_reason = f"metadata mismatch fields={mismatch}"
    if hit:
        _ION_LUT_MEMORY_CACHE[mem_key] = table
        return table

    ref_opts = {
        "cycle_avg_samples_ref": int(rate_table_cfg["ref_cycle_avg_samples"]),
        "popruzhenko_sum_tol": float(rate_table_cfg["popruzhenko_sum_tol"]),
        "popruzhenko_max_terms": int(rate_table_cfg["popruzhenko_max_terms"]),
    }
    table = build_rate_table(reference_model, species_params, omega0_SI, n0, rate_table_cfg["I_min_SI"], rate_table_cfg["I_max_SI"], rate_table_cfg["n_samples"], rate_table_cfg["spacing"], ref_opts)
    I_test = _np.logspace(_np.log10(float(rate_table_cfg["I_min_SI"])), _np.log10(float(rate_table_cfg["I_max_SI"])), 256)
    ref_eval = _get_reference_evaluator(reference_model, species_params, omega0_SI, n0, ref_opts)
    lut_eval = lambda I: eval_rate_from_table(I, table, method=rate_table_cfg.get("interp_mode", "loglog"))
    validation = validate_rate_table(ref_eval, lut_eval, I_test, filament_I_min_SI=rate_table_cfg.get("filament_I_min_SI", 1e13), filament_I_max_SI=rate_table_cfg.get("filament_I_max_SI", 1e15))
    if bool(rate_table_cfg.get("save_tables", True)):
        _save_table_npz(path, table, metadata, validation)

    table["metadata"] = metadata
    table["validation"] = validation
    if cache_reason:
        table["cache_reason"] = cache_reason
    _ION_LUT_MEMORY_CACHE[mem_key] = table
    return table
