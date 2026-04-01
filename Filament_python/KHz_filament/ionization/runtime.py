from __future__ import annotations

import math

from ..device import xp
from .common import DRHO_FRAC, W_CAP_DEFAULT, _as_real_like, _g, _minmax_inplace
from .lut import _ion_rate_table_defaults, eval_rate_from_table, prepare_ionization_lut_for_species
from .models_ppt import _W_mpa_factorial, cycle_average_ppt_talebpour_full_from_I, cycle_average_ppt_talebpour_legacy_from_I
from .models_popruzhenko import cycle_average_popruzhenko_atom_full_from_I
from .rate_registry import LEGACY_MODEL_REMOVED, RATE_ALIAS_MAP, REMOVED_RATES, SUPPORTED_RATES


def _removed_rate_error(rate: str, *, context: str) -> ValueError:
    repl = "ppt_talebpour_i_legacy / ppt_talebpour_i_full_reference / ppt_talebpour_i_lut / popruzhenko_atom_i_legacy / popruzhenko_atom_i_full_reference / popruzhenko_atom_i_lut / mpa_fact / off"
    hint = "该旧模型为 |E| 域模型，现已下线；请改用 I 域模型并传入强度 I。" if str(rate).lower() in ("ppt_e", "adk_e") else "请改用保留模型。"
    return ValueError(f"[ionization] {context}: rate='{rate}' 已被移除。建议替代: {repl}。{hint}")


def _resolve_rate(sp, ion_conf):
    r = str(_g(sp, "rate", None) or "").lower().replace("ppt-i", "ppt_i")
    if r:
        if r in REMOVED_RATES:
            raise _removed_rate_error(r, context="species.rate")
        if r in ("none", "zero"):
            return "off"
        if r in RATE_ALIAS_MAP:
            print(f"[ionization] rate alias '{r}' -> '{RATE_ALIAS_MAP[r]}'")
            return RATE_ALIAS_MAP[r]
        if r in SUPPORTED_RATES:
            return r
        raise ValueError(f"[ionization] 未识别的 rate='{r}'。")

    m = str(_g(sp, "model", getattr(ion_conf, "model", "off"))).lower()
    cyc = bool(_g(sp, "cycle_avg", getattr(ion_conf, "cycle_avg", False)))
    if m in ("none", "off", "zero", ""):
        return "off"
    if m in ("mpa_fact", "mpa_factorial", "multiphoton_factorial"):
        return "mpa_fact"
    if m in LEGACY_MODEL_REMOVED:
        raise _removed_rate_error(m, context=f"legacy model='{m}' with cycle_avg={cyc}")
    raise ValueError(f"[ionization] 无法从 legacy model='{m}' 推断可用电离模型")


def _talebpour_defaults(name: str, Ip_eV: float | None, Ip_eV_eff: float | None, Zeff: float | None):
    n = str(name).upper()
    if n == "O2":
        return float(Ip_eV_eff if Ip_eV_eff is not None else 12.55), float(Zeff if Zeff is not None else 0.53)
    if n == "N2":
        return float(Ip_eV if Ip_eV is not None else 15.576), float(Zeff if Zeff is not None else 0.9)
    return float(Ip_eV_eff if Ip_eV_eff is not None else (Ip_eV if Ip_eV is not None else 15.6)), float(Zeff if Zeff is not None else 1.0)


def _ion_model_family(rate: str) -> str:
    return {
        "ppt_talebpour_i_legacy": "legacy_simplified",
        "ppt_talebpour_i_full_reference": "talebpour_molecule_full",
        "ppt_talebpour_i_lut": "talebpour_molecule_lut",
        "popruzhenko_atom_i_full_reference": "popruzhenko_atom",
        "popruzhenko_atom_i_lut": "popruzhenko_atom_lut",
    }.get(str(rate).lower(), "other")


def make_Wfunc(model_or_conf, ion_conf, omega0_SI: float, n0: float):
    W_cap_global = float(getattr(ion_conf, "W_cap", W_CAP_DEFAULT))
    species = getattr(ion_conf, "species", None)
    assert species and isinstance(species, (list, tuple)) and len(species) > 0, "简化后的 ionization 需要提供非空的 species[]"

    _phase_count = int(getattr(ion_conf, "cycle_avg_samples", 64))
    rate_table_cfg = _ion_rate_table_defaults(ion_conf)
    W_scale_def = float(getattr(ion_conf, "W_scale", 1.0))

    def _mk_W_by_rate(rate, sp, W_cap):
        rate = str(rate).lower()
        W_scale = float(_g(sp, "W_scale", W_scale_def))

        if rate in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full_reference", "ppt_talebpour_i_lut"):
            name = str(_g(sp, "name", "")).strip()
            Ip_use, Zeff_use = _talebpour_defaults(name=name, Ip_eV=_g(sp, "Ip_eV", None), Ip_eV_eff=_g(sp, "Ip_eV_eff", None), Zeff=_g(sp, "Zeff", None))
            l, m = int(_g(sp, "l", 0)), int(_g(sp, "m", 0))
            max_terms = int(_g(sp, "max_terms", 4096))
            sum_rel_tol = float(_g(sp, "sum_rel_tol", 1e-9))

            def W_ref(I_SI, rate=rate):
                if rate in ("ppt_talebpour_i_full_reference", "ppt_talebpour_i_lut"):
                    W_eval = cycle_average_ppt_talebpour_full_from_I(I_SI, n0=n0, omega0_SI=omega0_SI, Ip_eV=Ip_use, Zeff=Zeff_use, l=l, m=m, samples=_phase_count, max_terms=max_terms, sum_rel_tol=sum_rel_tol, W_cap=W_cap)
                else:
                    W_eval = cycle_average_ppt_talebpour_legacy_from_I(I_SI, n0=n0, Ip_eV=Ip_use, Zeff=Zeff_use, l=l, m=m, samples=_phase_count, W_cap=W_cap)
                if W_scale != 1.0:
                    W_eval = W_eval * W_scale
                return xp.nan_to_num(xp.clip(W_eval, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)

            W_runtime = W_ref
            if rate == "ppt_talebpour_i_lut" and bool(rate_table_cfg.get("enabled", True)):
                table = prepare_ionization_lut_for_species(dict(sp), omega0_SI=omega0_SI, n0=n0, rate_table_cfg=rate_table_cfg)
                interp_mode = rate_table_cfg.get("interp_mode", "loglog")

                def W_runtime(I_SI, table=table, interp_mode=interp_mode):
                    W_eval = eval_rate_from_table(I_SI, table, method=interp_mode)
                    if W_scale != 1.0:
                        W_eval = W_eval * W_scale
                    return xp.nan_to_num(xp.clip(W_eval, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)

            return W_ref, W_runtime, "uses_I"

        if rate in ("popruzhenko_atom_i_full_reference", "popruzhenko_atom_i_lut", "popruzhenko_atom_i_legacy"):
            Ip = float(_g(sp, "Ip_eV", 15.6))
            Z = int(_g(sp, "Z", 1))
            max_terms = int(_g(sp, "max_terms", rate_table_cfg.get("popruzhenko_max_terms", 256)))
            sum_rel_tol = float(_g(sp, "sum_rel_tol", rate_table_cfg.get("popruzhenko_sum_tol", 1e-6)))

            def W_ref(I_SI):
                W_eval = cycle_average_popruzhenko_atom_full_from_I(I_SI, n0=n0, omega0_SI=omega0_SI, Ip_eV=Ip, Z=Z, samples=_phase_count, max_terms=max_terms, sum_rel_tol=sum_rel_tol, W_cap=W_cap)
                if W_scale != 1.0:
                    W_eval = W_eval * W_scale
                return xp.nan_to_num(xp.clip(W_eval, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)

            W_runtime = W_ref
            if rate == "popruzhenko_atom_i_lut" and bool(rate_table_cfg.get("enabled", True)):
                table = prepare_ionization_lut_for_species(dict(sp), omega0_SI=omega0_SI, n0=n0, rate_table_cfg=rate_table_cfg)
                interp_mode = rate_table_cfg.get("interp_mode", "loglog")

                def W_runtime(I_SI, table=table, interp_mode=interp_mode):
                    W_eval = eval_rate_from_table(I_SI, table, method=interp_mode)
                    if W_scale != 1.0:
                        W_eval = W_eval * W_scale
                    return xp.nan_to_num(xp.clip(W_eval, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)

            return W_ref, W_runtime, "uses_I"

        if rate == "mpa_fact":
            ell = int(_g(sp, "ell", 8))
            Imp = float(_g(sp, "I_mp", 1e18))

            def W_ref(I_SI):
                W = _W_mpa_factorial(I_SI, omega0_SI, Imp, ell, W_cap)
                if W_scale != 1.0:
                    W = W * W_scale
                return xp.nan_to_num(xp.clip(W, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)

            return W_ref, W_ref, "uses_I"

        def W_off(inp):
            return xp.zeros_like(inp, dtype=(xp.float32 if xp.asarray(inp).dtype == xp.complex64 else xp.float64))

        return W_off, W_off, "none"

    entries, flags, species_meta = [], set(), []
    for sp in species:
        name = str(_g(sp, "name", f"sp{len(species_meta)}"))
        frac = float(_g(sp, "fraction", 1.0))
        W_cap = float(_g(sp, "W_cap", W_cap_global))
        rate = _resolve_rate(sp, ion_conf)
        W_ref, W_runtime, tag = _mk_W_by_rate(rate, sp, W_cap)
        entries.append((frac, W_runtime, tag))
        species_meta.append({"name": name, "fraction": frac, "W_s": W_ref, "W_runtime": W_runtime, "tag": tag, "rate": rate, "family": _ion_model_family(rate)})
        flags.add(tag)

    def Wfunc(inp):
        Wtot = None
        for frac, W_s, _tag in entries:
            Ws = W_s(inp)
            Wtot = Ws * frac if Wtot is None else (Wtot + frac * Ws)
        cap = float(getattr(ion_conf, "W_cap", W_cap_global))
        return xp.nan_to_num(xp.clip(Wtot, 0.0, cap), nan=0.0, posinf=cap, neginf=0.0)

    Wfunc._expects = "uses_I" if "uses_I" in flags else ("uses_E" if "uses_E" in flags else "none")
    Wfunc._species_entries = tuple(species_meta)
    Wfunc._species_fractions = tuple([m["fraction"] for m in species_meta])
    Wfunc._time_mode = str(getattr(ion_conf, "time_mode", "")).lower()
    Wfunc._integrator = str(getattr(ion_conf, "integrator", "rk4")).lower()
    return Wfunc


def _ion_input_domain(ion_conf):
    species = getattr(ion_conf, "species", None)
    assert species and len(species) > 0, "要求提供 species[]"
    for sp in species:
        _ = _resolve_rate(sp, ion_conf)
    return "I"


def prepare_ionization_lut_cache(ion_conf, omega0_SI: float, n0: float):
    species = getattr(ion_conf, "species", None) or []
    table_cfg = _ion_rate_table_defaults(ion_conf)
    if not bool(table_cfg.get("enabled", True)):
        return []
    built = []
    for sp in species:
        rate = _resolve_rate(sp, ion_conf)
        sp_local = dict(sp)
        sp_local["rate"] = rate
        if rate in ("ppt_talebpour_i_lut", "popruzhenko_atom_i_lut"):
            built.append(prepare_ionization_lut_for_species(sp_local, omega0_SI=omega0_SI, n0=n0, rate_table_cfg=table_cfg))
    return built


def evolve_rho_time(input_array, dt: float, N0: float, beta_rec: float, Wfunc, *, quasi_static_time: bool = False, time_stat: str = "peak", mean_clip_frac: float = 1e-3, expects: str | None = None):
    tm = str(getattr(Wfunc, "_time_mode", "")).lower()
    if tm in ("qs_peak", "qs_mean", "qs_mean_esq"):
        quasi_static_time = True
        time_stat = {"qs_peak": "peak", "qs_mean": "mean", "qs_mean_esq": "mean_esq"}[tm]
    elif tm == "full":
        quasi_static_time = False

    inp = _as_real_like(input_array, like=input_array)
    Nt = int(inp.shape[0])
    rdtype = xp.float32 if inp.dtype == xp.complex64 else xp.float64
    expects = (expects or getattr(Wfunc, "_expects", "E")).upper()

    sp_entries = getattr(Wfunc, "_species_entries", None)
    if sp_entries and len(sp_entries) > 0:
        fracs = xp.asarray([max(0.0, float(s["fraction"])) for s in sp_entries], dtype=rdtype)
        ssum = float(fracs.sum())
        fracs = (fracs / ssum) if (xp.isfinite(ssum) and ssum > 0.0) else (xp.ones_like(fracs) / float(len(sp_entries)))
        N0_j_list = [float(N0) * float(fracs[j]) for j in range(len(sp_entries))]

        if not quasi_static_time:
            Wt_list = [s.get("W_runtime", s["W_s"])(inp).astype(rdtype, copy=False) for s in sp_entries]
            beta_list = [float(beta_rec) * float(N0_j) for N0_j in N0_j_list]
            u_list = [xp.zeros_like(inp, dtype=rdtype) for _ in sp_entries]
            mode = str(getattr(Wfunc, "_integrator", "rk4")).lower()
            for it in range(Nt - 1):
                if mode == "rk4":
                    for j in range(len(sp_entries)):
                        u = u_list[j][it]
                        W1, W4 = Wt_list[j][it], Wt_list[j][it + 1]
                        W2 = 0.5 * (W1 + W4)
                        betaN = beta_list[j]
                        k1 = W1 * (1.0 - u) - betaN * (u * u)
                        u2 = u + 0.5 * dt * k1
                        k2 = W2 * (1.0 - u2) - betaN * (u2 * u2)
                        u3 = u + 0.5 * dt * k2
                        k3 = W2 * (1.0 - u3) - betaN * (u3 * u3)
                        u4 = u + dt * k3
                        k4 = W4 * (1.0 - u4) - betaN * (u4 * u4)
                        u_next = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                        _minmax_inplace(u_next, 0.0, 1.0)
                        u_list[j][it + 1] = u_next
                else:
                    Wmax_frame = max(float(xp.max(Wt_list[j][it])) for j in range(len(sp_entries)))
                    max_du_dt = max(Wmax_frame + max(beta_list), 0.0)
                    if (max_du_dt <= 0.0) or (not math.isfinite(max_du_dt)):
                        n_sub = 1
                    else:
                        val = dt * max_du_dt / max(DRHO_FRAC, 1e-6)
                        n_sub = int(val) if val == int(val) else int(val) + 1
                        n_sub = 1 if n_sub < 1 else (1000 if n_sub > 1000 else n_sub)
                    dt_sub = dt / n_sub
                    for _ in range(n_sub):
                        for j in range(len(sp_entries)):
                            u = u_list[j][it]
                            du = dt_sub * (Wt_list[j][it] * (1.0 - u) - beta_list[j] * (u * u))
                            u = u + du
                            _minmax_inplace(u, 0.0, 1.0)
                            u_list[j][it] = u
                    for j in range(len(sp_entries)):
                        u_list[j][it + 1] = u_list[j][it]

            rho_list = [(u_list[j] * float(N0_j_list[j])).astype(rdtype, copy=False) for j in range(len(sp_entries))]
            rho_sum = rho_list[0]
            for j in range(1, len(sp_entries)):
                rho_sum = rho_sum + rho_list[j]
            Wt_total = Wt_list[0] * float(fracs[0])
            for j in range(1, len(sp_entries)):
                Wt_total = Wt_total + Wt_list[j] * float(fracs[j])
            return rho_sum, Wt_total

        stat = str(time_stat).lower()
        Wc_list = []
        if stat in ("peak", "max"):
            for s in sp_entries:
                Wc_list.append(xp.max(s.get("W_runtime", s["W_s"])(inp).astype(rdtype, copy=False), axis=0))
        elif stat in ("mean", "avg"):
            for s in sp_entries:
                Wt_full = s.get("W_runtime", s["W_s"])(inp).astype(rdtype, copy=False)
                if mean_clip_frac and mean_clip_frac > 0.0:
                    peak_inp = xp.max(inp, axis=0) + 1e-30
                    mask = inp >= (mean_clip_frac * peak_inp)[None, ...]
                    Wc = (Wt_full * mask).sum(axis=0) / inp.shape[0]
                else:
                    Wc = xp.mean(Wt_full, axis=0)
                Wc_list.append(Wc.astype(rdtype, copy=False))
        elif stat in ("mean_esq", "rms_e", "e_rms"):
            if expects == "E":
                E_rms = xp.sqrt(xp.mean(inp * inp, axis=0) + 0.0)
                for s in sp_entries:
                    Wc_list.append(s.get("W_runtime", s["W_s"])(E_rms).astype(rdtype, copy=False))
            else:
                I_mean = xp.mean(inp, axis=0)
                for s in sp_entries:
                    Wc_list.append(s.get("W_runtime", s["W_s"])(I_mean).astype(rdtype, copy=False))
        else:
            raise ValueError("time_stat 只能是 'peak' | 'mean' | 'mean_Esq'")

        t_idx = (xp.arange(1, Nt + 1, dtype=rdtype) * float(dt))[:, None, None]
        rho_list = [float(N0_j_list[j]) * (1.0 - xp.exp(-Wc_list[j][None, ...] * t_idx)) for j in range(len(sp_entries))]
        rho_sum = rho_list[0]
        for j in range(1, len(sp_entries)):
            rho_sum = rho_sum + rho_list[j]
        Wt_total = Wc_list[0] * float(fracs[0])
        for j in range(1, len(sp_entries)):
            Wt_total = Wt_total + Wc_list[j] * float(fracs[j])
        Wt_total = xp.broadcast_to(Wt_total[None, ...], inp.shape).astype(rdtype, copy=False)
        return xp.asarray(rho_sum, dtype=rdtype), Wt_total

    if not quasi_static_time:
        Wt = Wfunc(inp)
        u = xp.zeros_like(inp, dtype=rdtype)
        betaN0 = float(beta_rec) * float(N0)
        for it in range(Nt - 1):
            Wmax = float(xp.max(Wt[it]))
            max_du_dt = Wmax + betaN0
            if (max_du_dt <= 0.0) or (not xp.isfinite(max_du_dt)):
                n_sub = 1
            else:
                val = dt * max_du_dt / max(DRHO_FRAC, 1e-6)
                n_sub = int(val) if val == int(val) else int(val) + 1
                n_sub = 1 if n_sub < 1 else (1000 if n_sub > 1000 else n_sub)
            dt_sub = dt / n_sub
            u_t = u[it]
            for _ in range(n_sub):
                du = dt_sub * (Wt[it] * (1.0 - u_t) - betaN0 * (u_t * u_t))
                u_t = u_t + du
                _minmax_inplace(u_t, 0.0, 1.0)
            u[it + 1] = u_t
        return (u * float(N0)).astype(rdtype, copy=False), Wt

    stat = str(time_stat).lower()
    Wt_full = Wfunc(inp).astype(rdtype, copy=False)
    if stat in ("peak", "max"):
        Wc = xp.max(Wt_full, axis=0)
    elif stat in ("mean", "avg"):
        if mean_clip_frac and mean_clip_frac > 0.0:
            peak_inp = xp.max(inp, axis=0) + 1e-30
            mask = inp >= (mean_clip_frac * peak_inp)[None, ...]
            cnt = xp.maximum(mask.sum(axis=0), 1)
            Wc = (Wt_full * mask).sum(axis=0) / cnt
        else:
            Wc = xp.mean(Wt_full, axis=0)
    elif stat in ("mean_esq", "rms_e", "e_rms"):
        Wc = Wfunc(xp.sqrt(xp.mean(inp * inp, axis=0) + 0.0) if expects == "E" else xp.mean(inp, axis=0)).astype(rdtype, copy=False)
    else:
        raise ValueError("time_stat 只能是 'peak' | 'mean' | 'mean_Esq'")

    Wt = xp.broadcast_to(Wc[None, ...], inp.shape)
    t_idx = (xp.arange(1, Nt + 1, dtype=rdtype) * float(dt))[:, None, None]
    rho_t = float(N0) * (1.0 - xp.exp(-Wc[None, ...] * t_idx))
    return rho_t.astype(rdtype, copy=False), Wt
