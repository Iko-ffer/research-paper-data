#!/usr/bin/env python3
"""
10 s non-overlapping chunks: compare single2 (V1+V6 hedge) vs V1 / V6 / V16-focused ICA.

Maternal + uterine from one native 6 ch ICA pass per chunk; fetal = best-scoring route
(or score-weighted blend when top two are close).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from abdominal_bss_separate import (
    branch_ecg,
    preprocess_common_for_pipeline,
    sliding_bss_three_outputs,
    spatial_maternal_proxy,
)
from chunked_multiroute_bss import (
    _bandpass,
    _chunk_start_indices,
    _refine_route_composite_score,
    _route_to_matrix,
    adaptive_v1_v6_combine,
    adaptive_v1_v6_combine_best_polarity,
    fetal_route_quality,
    single2_bandpass_from_kwargs,
    spatial_fetal_proxy,
)


FETAL_COMPARE_ROUTES: Tuple[str, ...] = ("single2", "v1_focus", "v6_focus", "v16_band")


def single2_fetal_trace(x6: np.ndarray, fs: float, bss_kwargs: Optional[dict] = None) -> np.ndarray:
    """
    V1/V6 hedge + 5–40 Hz bandpass (``single2`` core).

    When ``bss_kwargs`` is present and ``single2_dual_polarity`` is true (default), choose
    ``v1+k*v6`` vs ``v1-k*v6`` by ``fetal_route_quality`` vs the spatial fetal proxy.
    """
    c0 = np.asarray(x6[:, 0], dtype=np.float64)
    c5 = np.asarray(x6[:, 5], dtype=np.float64)
    use_dual = bool(bss_kwargs.get("single2_dual_polarity", True)) if bss_kwargs else False
    if use_dual and bss_kwargs is not None:
        x0 = preprocess_common_for_pipeline(x6, fs, bss_kwargs)
        fb = tuple(bss_kwargs.get("fetal_band", (17.0, 42.0)))
        mb = bss_kwargs.get("maternal_ecg_band") or bss_kwargs.get("ecg_band") or (1.0, 45.0)
        xf = _bandpass(x0, fs, fb[0], fb[1])
        xm = branch_ecg(x0, fs, low=float(mb[0]), high=float(mb[1]))
        prox_f = spatial_fetal_proxy(xf)
        prox_m = spatial_maternal_proxy(xm)
        use_v2 = bool(bss_kwargs.get("single2_polarity_v2", True))
        s2_band, s2_ord = single2_bandpass_from_kwargs(bss_kwargs)
        return adaptive_v1_v6_combine_best_polarity(
            c0,
            c5,
            fs,
            prox_f,
            prox_m=prox_m,
            fetal_band=fb,
            use_v2=use_v2,
            single2_band_hz=s2_band,
            single2_bandpass_order=s2_ord,
        )
    return adaptive_v1_v6_combine(c0, c5, fs)


def _pop_hybrid_only_bss_keys(kw: dict) -> None:
    for _hk in (
        "single2_dual_polarity",
        "single2_polarity_v2",
        "single2_band_hz",
        "single2_bandpass_order",
        "hybrid_fetal_hedge_ica_obs",
        "hybrid_fetal_ica_gating",
        "hybrid_single2_gate_margin",
        "hybrid_fetal_score_fusion",
        "hybrid_fetal_blend_score_margin",
        "hybrid_maternal_score_fusion",
        "hybrid_maternal_blend_score_margin",
        "refine_fhr_window_sec",
        "refine_mhr_penalize_fetal_boost",
        "refine_neighbor_bridge_sec",
        "refine_bad_mode",
        "refine_use_fhr_stability",
        "hybrid_v16_refine",
        "fetal_routes",
        "chunk_sec",
        "hybrid_ica_gate_min_peaks",
        "hybrid_ica_gate_peak_ratio",
    ):
        kw.pop(_hk, None)


def _ica_once_on_chunk(
    x_in: np.ndarray,
    fs: float,
    chunk_sec: float,
    bss_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One ICA window spanning the whole chunk (window == hop == chunk length)."""
    kw = dict(bss_kwargs or {})
    _pop_hybrid_only_bss_keys(kw)
    kw["verbose"] = False
    kw["window_sec"] = float(chunk_sec)
    kw["hop_sec"] = float(chunk_sec)
    out = sliding_bss_three_outputs(x_in, fs, **kw)
    L = x_in.shape[0]
    m = np.asarray(out["maternal_ecg"], dtype=np.float64).ravel()[:L]
    f = np.asarray(out["fetal_ecg"], dtype=np.float64).ravel()[:L]
    u = np.asarray(out["uterine_abdominal"], dtype=np.float64).ravel()[:L]
    return m, f, u


def _score_fetal(
    f: np.ndarray,
    fs: float,
    prox_q: np.ndarray,
    fb: Tuple[float, float],
    m: Optional[np.ndarray],
    bss_kwargs: dict,
) -> float:
    base = fetal_route_quality(f, fs, prox_f_precomputed=prox_q, fetal_band=fb)
    if m is None:
        return base
    return _refine_route_composite_score(f, m, fs, prox_q, fb, bss_kwargs)


def pick_best_fetal_route(
    seg6: np.ndarray,
    fs: float,
    chunk_sec: float,
    bss_kwargs: dict,
    routes: Tuple[str, ...] = FETAL_COMPARE_ROUTES,
    *,
    blend_score_margin: float = 0.0,
) -> Tuple[np.ndarray, str, float, Dict[str, float]]:
    """
    Compare ``single2`` and V1/V6/V16 ICA routes; return fetal trace + route label + score map.
    """
    L = int(seg6.shape[0])
    fb = tuple(bss_kwargs.get("fetal_band", (17.0, 42.0)))
    x0q = preprocess_common_for_pipeline(seg6, fs, bss_kwargs)
    xf_q = _bandpass(x0q, fs, fb[0], fb[1])
    prox_q = spatial_fetal_proxy(xf_q)

    scored: List[Tuple[str, np.ndarray, float, Optional[np.ndarray]]] = []
    for route in routes:
        try:
            if route == "single2":
                f = single2_fetal_trace(seg6, fs, bss_kwargs)
                m_ref = None
            else:
                x_in = _route_to_matrix(seg6, fs, route, bss_kwargs)
                _, f, _ = _ica_once_on_chunk(x_in, fs, chunk_sec, bss_kwargs)
                m_ref = None
            if len(f) < L:
                continue
            f = f[:L].copy()
            sc = _score_fetal(f, fs, prox_q, fb, m_ref, bss_kwargs)
            scored.append((route, f, sc, m_ref))
        except Exception:
            continue

    if not scored:
        f = single2_fetal_trace(seg6, fs, bss_kwargs)
        return f[:L], "single2_fallback", 0.0, {"single2_fallback": 0.0}

    scored.sort(key=lambda x: -x[2])
    best_route, best_f, best_sc, _ = scored[0]
    score_map = {r: float(s) for r, _, s, _ in scored}

    if len(scored) >= 2 and blend_score_margin > 0:
        r2, f2, s2, _ = scored[1]
        if best_sc - s2 <= blend_score_margin and s2 > 0:
            w1 = best_sc / (best_sc + s2 + 1e-9)
            w2 = 1.0 - w1
            fused = w1 * best_f + w2 * f2[:L]
            tag = f"blend:{best_route}+{r2}"
            return fused, tag, float(best_sc), score_map

    return best_f, best_route, float(best_sc), score_map


def _fetal_peak_metrics(f: np.ndarray, fs: float) -> Tuple[int, float, float]:
    """Peak count, BPM std, median BPM on fetal-like RR (for gating ICA vs single2)."""
    from scipy.signal import find_peaks

    s = np.asarray(f, dtype=np.float64).ravel() - np.median(f)
    std = float(np.std(s)) + 1e-9
    peaks, _ = find_peaks(
        np.abs(s),
        distance=max(3, int(fs * 0.3)),
        prominence=std * 0.5,
    )
    if len(peaks) < 4:
        return len(peaks), 0.0, 0.0
    rr = np.diff(peaks) / fs
    bpm = 60.0 / rr
    ok = (bpm >= 95.0) & (bpm <= 200.0) & np.isfinite(bpm)
    if ok.sum() < 3:
        return len(peaks), 0.0, 0.0
    b = bpm[ok]
    return int(len(peaks)), float(np.std(b)), float(np.median(b))


def _ica_peak_ok_for_gate(
    f_ica: np.ndarray,
    f_s2: np.ndarray,
    fs: float,
    bss_kwargs: dict,
) -> bool:
    """ICA trace must have enough peaks vs single2 before we prefer it over single2."""
    n_i, std_i, _ = _fetal_peak_metrics(f_ica, fs)
    n_s, std_s, _ = _fetal_peak_metrics(f_s2, fs)
    min_peaks = int(bss_kwargs.get("hybrid_ica_gate_min_peaks", 12))
    ratio = float(bss_kwargs.get("hybrid_ica_gate_peak_ratio", 0.55))
    if n_i < min_peaks:
        return False
    if n_s >= min_peaks and n_i < int(ratio * n_s):
        return False
    if std_s >= 3.0 and std_i < 2.5:
        return False
    return True


def pick_fetal_chunk_ica_gated(
    seg6: np.ndarray,
    fs: float,
    chunk_sec: float,
    f_native: np.ndarray,
    bss_kwargs: dict,
    prox_q: np.ndarray,
    fb: Tuple[float, float],
    *,
    f_s2_ref: Optional[np.ndarray] = None,
    fetal_routes: Tuple[str, ...] = FETAL_COMPARE_ROUTES,
    ica_override_margin: float = 0.15,
    single2_rescue_threshold: float = 2.05,
) -> Tuple[np.ndarray, str, float, Dict[str, float]]:
    """
    Fetal IC from hedge-augmented band ICA vs single2 — hard pick only (no waveform blend).

    When ``f_s2_ref`` is set (whole-trace single2 slice), gating compares ICA to that reference
    instead of re-running single2 on the chunk alone (matches RUN-123345 behaviour).
    """
    L = int(seg6.shape[0])
    f_ica = np.asarray(f_native, dtype=np.float64).ravel()[:L]
    if f_s2_ref is not None:
        f_s2 = np.asarray(f_s2_ref, dtype=np.float64).ravel()[:L]
    else:
        f_s2 = single2_fetal_trace(seg6, fs, bss_kwargs)[:L]
    sc_ica = _score_fetal(f_ica, fs, prox_q, fb, None, bss_kwargs)
    sc_s2 = _score_fetal(f_s2, fs, prox_q, fb, None, bss_kwargs)
    s2_ok = _single2_chunk_has_plausible_peaks(f_s2, fs)
    scores: Dict[str, float] = {"ica_split": float(sc_ica), "single2": float(sc_s2)}
    gate_margin = float(bss_kwargs.get("hybrid_single2_gate_margin", 0.0))

    ica_ok = _ica_peak_ok_for_gate(f_ica, f_s2, fs, bss_kwargs)
    scores["ica_peak_ok"] = float(1.0 if ica_ok else 0.0)
    n_i, _, _ = _fetal_peak_metrics(f_ica, fs)
    n_s, _, _ = _fetal_peak_metrics(f_s2, fs)
    scores["ica_n_peaks"] = float(n_i)
    scores["single2_n_peaks"] = float(n_s)

    ica_margin = float(bss_kwargs.get("hybrid_ica_override_margin", ica_override_margin))
    if s2_ok and sc_s2 >= sc_ica - gate_margin:
        return f_s2, "single2", float(sc_s2), scores
    if ica_ok and sc_ica >= sc_s2 + ica_margin:
        return f_ica, "ica_split", float(sc_ica), scores
    if ica_ok and (not s2_ok) and sc_ica >= sc_s2 - gate_margin:
        return f_ica, "ica_split", float(sc_ica), scores
    if s2_ok:
        return f_s2, "single2", float(sc_s2), scores

    if sc_s2 < float(single2_rescue_threshold) and not s2_ok:
        ica_routes = tuple(r for r in ("v16_band", "v6_focus") if r in fetal_routes)
        candidates: List[Tuple[float, str, np.ndarray]] = []
        for route in ica_routes:
            try:
                x_in = _route_to_matrix(seg6, fs, route, bss_kwargs)
                _, f_r, _ = _ica_once_on_chunk(x_in, fs, chunk_sec, bss_kwargs)
                f_r = f_r[:L]
                sc = _score_fetal(f_r, fs, prox_q, fb, None, bss_kwargs)
                scores[route] = float(sc)
                if sc <= sc_s2 + float(ica_override_margin):
                    continue
                candidates.append((float(sc), route, f_r.copy()))
            except Exception:
                continue
        if candidates:
            candidates.sort(key=lambda t: -t[0])
            sc_b, route_b, f_b = candidates[0]
            return f_b, route_b, float(sc_b), scores

    if ica_ok and sc_ica >= sc_s2:
        return f_ica, "ica_split", float(sc_ica), scores
    return f_s2, "single2", float(sc_s2), scores


def _fetal_chunk_amplitude_ok(
    f_new: np.ndarray,
    f_ref: np.ndarray,
    max_ratio: float = 3.0,
) -> bool:
    """Reject ICA/rescue chunks whose robust scale diverges from the single2 reference."""
    a = np.asarray(f_ref, dtype=np.float64).ravel()
    b = np.asarray(f_new, dtype=np.float64).ravel()
    sa = float(np.std(a)) + 1e-12
    sb = float(np.std(b)) + 1e-12
    ratio = float(max_ratio)
    return sb <= ratio * sa and sa <= ratio * sb


def _single2_chunk_has_plausible_peaks(f_s2: np.ndarray, fs: float) -> bool:
    """If single2 trace has enough fetal-like peaks, skip ICA replacement for this chunk."""
    from scipy.signal import find_peaks

    s = np.asarray(f_s2, dtype=np.float64).ravel() - np.median(f_s2)
    std = float(np.std(s)) + 1e-9
    peaks, _ = find_peaks(
        np.abs(s),
        distance=max(3, int(fs * 0.28)),
        prominence=std * 0.5,
    )
    if len(peaks) < 4:
        return False
    rr = np.diff(peaks) / fs
    bpm = 60.0 / rr
    m = np.isfinite(bpm) & (bpm >= 95.0) & (bpm <= 200.0)
    return bool(m.sum() >= 3 and float(np.median(bpm[m])) > 0)


def hybrid_v16_chunked_bss(
    x6: np.ndarray,
    fs: float,
    *,
    chunk_sec: float = 10.0,
    bss_kwargs: Optional[dict] = None,
    fetal_routes: Tuple[str, ...] = FETAL_COMPARE_ROUTES,
    progress_callback: Optional[Callable[..., None]] = None,
    ica_override_margin: float = 0.15,
    single2_rescue_threshold: float = 2.05,
) -> dict:
    """
    Non-overlapping ``chunk_sec`` tiles. Per tile: native 6 ch ICA for m/u; fetal from route compare.
    """
    bss_kwargs = dict(bss_kwargs or {})
    n = int(x6.shape[0])
    chunk = max(256, int(chunk_sec * fs))
    hop = chunk  # no overlap

    hedge_ica_obs = bool(bss_kwargs.get("hybrid_fetal_hedge_ica_obs", False))
    maternal = np.zeros(n, dtype=np.float64)
    # Always single2 underlay (RUN-123345); hedge mode overwrites chunks when ICA wins gating.
    fetal = single2_fetal_trace(x6, fs, bss_kwargs)
    uterine = np.zeros(n, dtype=np.float64)
    chunk_meta: List[dict] = []
    fb = tuple(bss_kwargs.get("fetal_band", (17.0, 42.0)))
    x0_full = preprocess_common_for_pipeline(x6, fs, bss_kwargs)
    prox_full = spatial_fetal_proxy(_bandpass(x0_full, fs, fb[0], fb[1]))

    starts = _chunk_start_indices(n, chunk, hop)
    n_chunks = len(starts)
    ica_routes = tuple(r for r in ("v16_band", "v6_focus") if r in fetal_routes)

    for ci, start in enumerate(starts):
        end = min(start + chunk, n)
        seg_len = end - start
        seg = x6[start:end].copy()
        chunk_dur_sec = seg_len / fs
        prox_q = prox_full[start:end]

        m, f_native, u = _ica_once_on_chunk(seg, fs, chunk_dur_sec, bss_kwargs)

        if hedge_ica_obs:
            f_s2_ref = fetal[start:end].copy()
            best_f, best_route, best_sc, route_scores = pick_fetal_chunk_ica_gated(
                seg,
                fs,
                chunk_dur_sec,
                f_native,
                bss_kwargs,
                prox_q,
                fb,
                f_s2_ref=f_s2_ref,
                fetal_routes=fetal_routes,
                ica_override_margin=float(ica_override_margin),
                single2_rescue_threshold=float(single2_rescue_threshold),
            )
        else:
            f_s2 = fetal[start:end].copy()
            score_s2 = _score_fetal(f_s2, fs, prox_q, fb, None, bss_kwargs)
            best_route = "single2"
            best_f = f_s2
            best_sc = score_s2
            route_scores = {"single2": float(score_s2)}

            if score_s2 < float(single2_rescue_threshold) and not _single2_chunk_has_plausible_peaks(
                f_s2, fs
            ):
                candidates: List[Tuple[float, float, str, np.ndarray]] = []
                for route in ica_routes:
                    try:
                        x_in = _route_to_matrix(seg, fs, route, bss_kwargs)
                        _, f_ica, _ = _ica_once_on_chunk(x_in, fs, chunk_dur_sec, bss_kwargs)
                        f_ica = f_ica[:seg_len]
                        sc = _score_fetal(f_ica, fs, prox_q, fb, None, bss_kwargs)
                        route_scores[route] = float(sc)
                        if sc <= score_s2 + float(ica_override_margin):
                            continue
                        a = f_s2.astype(np.float64)
                        b = f_ica.astype(np.float64)
                        if float(np.std(a)) > 1e-9 and float(np.std(b)) > 1e-9:
                            cr = abs(float(np.corrcoef(a, b)[0, 1]))
                        else:
                            cr = 0.0
                        candidates.append((float(sc), cr, route, f_ica.copy()))
                    except Exception:
                        continue
                if candidates:
                    candidates.sort(key=lambda t: (t[1], t[0]), reverse=True)
                    best_sc, _cr, best_route, best_f = candidates[0]

        sl = slice(start, end)
        maternal[sl] = m[:seg_len]
        uterine[sl] = u[:seg_len]
        f_ref = fetal[start:end]
        if best_route != "single2" and not _fetal_chunk_amplitude_ok(best_f, f_ref):
            best_route = "single2"
            best_f = f_ref
        replace_fetal = (not hedge_ica_obs) or (best_route != "single2")
        ramp_n = max(
            0,
            min(int(0.25 * fs), seg_len // 4),
        ) if replace_fetal and best_route not in ("single2", "ica_split") else 0
        if replace_fetal:
            if ramp_n > 0 and start > 0:
                a = np.linspace(0.0, 1.0, ramp_n, endpoint=False)
                fetal[sl.start : sl.start + ramp_n] = (
                    (1.0 - a) * fetal[sl.start : sl.start + ramp_n] + a * best_f[:ramp_n]
                )
                fetal[sl.start + ramp_n : sl.stop] = best_f[ramp_n:seg_len]
            else:
                fetal[sl] = best_f[:seg_len]

        chunk_meta.append(
            {
                "start_s": start / fs,
                "end_s": end / fs,
                "fetal_route": best_route,
                "fetal_score": float(best_sc),
                "route_scores": route_scores,
            }
        )

        if progress_callback:
            progress_callback(ci + 1, n_chunks, start, best_route, float(best_sc))

    x0 = preprocess_common_for_pipeline(x6, fs, bss_kwargs)
    mb = bss_kwargs.get("maternal_ecg_band") or bss_kwargs.get("ecg_band") or (1.0, 45.0)
    fb = bss_kwargs.get("fetal_band", (17.0, 42.0))
    x_m = branch_ecg(x0, fs, low=mb[0], high=mb[1])

    return {
        "maternal_ecg": maternal,
        "fetal_ecg": fetal,
        "uterine_abdominal": uterine,
        "aux": {
            "x_ecg_branch": x_m,
            "x_fetal_band_branch": _bandpass(x0, fs, fb[0], fb[1]),
            "x_uterine_branch": np.zeros_like(x0),
            "maternal_robust_ref": np.zeros(n),
            "uterine_envelope_physical": np.zeros(n),
            "fs": fs,
            "chunk_starts": starts,
            "chunk_meta": chunk_meta,
            "hybrid_v16": True,
        },
    }


def refine_hybrid_fetal_segment(
    x6_seg: np.ndarray,
    fs: float,
    chunk_sec: float,
    bss_kwargs: dict,
    routes: Tuple[str, ...] = FETAL_COMPARE_ROUTES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Bad-window refine: native m/u + gated fetal (hedge ICA obs or legacy routes)."""
    L = int(x6_seg.shape[0])
    dur = L / fs
    win = min(chunk_sec, dur)
    m, f_native, u = _ica_once_on_chunk(x6_seg, fs, win, bss_kwargs)
    fb = tuple(bss_kwargs.get("fetal_band", (17.0, 42.0)))
    x0q = preprocess_common_for_pipeline(x6_seg, fs, bss_kwargs)
    prox_q = spatial_fetal_proxy(_bandpass(x0q, fs, fb[0], fb[1]))
    if bool(bss_kwargs.get("hybrid_fetal_hedge_ica_obs", False)):
        f, route, score, _ = pick_fetal_chunk_ica_gated(
            x6_seg,
            fs,
            win,
            f_native,
            bss_kwargs,
            prox_q,
            fb,
            fetal_routes=routes,
            ica_override_margin=float(bss_kwargs.get("hybrid_ica_override_margin", 0.15)),
            single2_rescue_threshold=float(bss_kwargs.get("hybrid_single2_rescue_threshold", 2.05)),
        )
    else:
        f, route, score, _ = pick_best_fetal_route(
            x6_seg, fs, win, bss_kwargs, routes=routes, blend_score_margin=0.0
        )
    return m[:L], f[:L], u[:L], float(score), route
