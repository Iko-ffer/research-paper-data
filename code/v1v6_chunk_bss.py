#!/usr/bin/env python3
"""
10 s non-overlapping chunks: single ICA on V1/V6 + hedge observations.

Primary fetal output = fetal IC from one FastICA per chunk; tiered rescue (alt IC,
re-hedge, single2, v6_focus). FHR jump refinement calls ``refine_v1v6_segment``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from abdominal_bss_separate import (
    _bandpass,
    _fastica_on_observation,
    _match_sign,
    _zscore_matrix_cols,
    build_v1v6_ica_observation,
    pick_three_sources,
    pca_single_band_stack,
    preprocess_common_for_pipeline,
    score_components,
    spatial_maternal_proxy,
    uterine_contraction_envelope,
    v1v6_fetal_ic_candidates,
)
from chunked_multiroute_bss import _chunk_start_indices, _route_to_matrix
from hybrid_v16_chunk_bss import (
    FETAL_COMPARE_ROUTES,
    _fetal_chunk_amplitude_ok,
    _fetal_peak_metrics,
    _ica_once_on_chunk,
    _pop_hybrid_only_bss_keys,
    _score_fetal,
    _single2_chunk_has_plausible_peaks,
    single2_fetal_trace,
)


def _v1v6_ica_core(
    x6_seg: np.ndarray,
    fs: float,
    bss_kwargs: dict,
    *,
    include_fetal_hedge_plus: bool = True,
    include_fetal_hedge_minus: bool = True,
    include_maternal_hedge_minus: bool = True,
    rng_seed: Optional[int] = None,
) -> dict:
    """One V1/V6 observation stack + PCA + FastICA on a segment."""
    x0 = preprocess_common_for_pipeline(x6_seg, fs, bss_kwargs)
    mb = bss_kwargs.get("maternal_ecg_band") or bss_kwargs.get("ecg_band") or (1.0, 45.0)
    fb = bss_kwargs.get("fetal_band", (17.0, 42.0))
    hedge_bp = tuple(bss_kwargs.get("single2_band_hz", (5.0, 40.0)))

    obs, meta = build_v1v6_ica_observation(
        x0,
        fs,
        maternal_band=tuple(mb),
        fetal_band=tuple(fb),
        hedge_band=hedge_bp,
        include_fetal_hedge_plus=include_fetal_hedge_plus,
        include_fetal_hedge_minus=include_fetal_hedge_minus,
        include_maternal_hedge_minus=include_maternal_hedge_minus,
    )
    prox_m = meta["prox_m"]
    prox_f = meta["prox_f"]
    fetal_spec = (float(fb[0]), min(float(fb[1]), 0.48 * fs))

    rs = int(rng_seed) if rng_seed is not None else 0
    try:
        z = pca_single_band_stack(obs, n_keep=6, random_state=rs)
        seg_ica = _zscore_matrix_cols(z)
    except Exception:
        seg_ica = _zscore_matrix_cols(obs[:, : min(6, obs.shape[1])])

    from numpy.random import RandomState

    rng = RandomState(rs)
    S = _fastica_on_observation(
        seg_ica,
        bss=str(bss_kwargs.get("bss", "fastica")),
        rng=rng,
        fastica_max_iter=int(bss_kwargs.get("fastica_max_iter", 400)),
        fastica_tol=float(bss_kwargs.get("fastica_tol", 2e-3)),
    )
    sm, sf, su, _ = score_components(
        S,
        fs,
        prox_m=prox_m,
        prox_f=prox_f,
        fetal_spec_band=fetal_spec,
        maternal_penalize_fetal_proxy=float(bss_kwargs.get("maternal_penalize_fetal_proxy", 1.75)),
        fetal_penalize_maternal_proxy=float(bss_kwargs.get("fetal_penalize_maternal_proxy", 1.35)),
    )
    im, iff, iu = pick_three_sources(
        S,
        sm,
        sf,
        su,
        prox_m=prox_m,
        prox_f=prox_f,
        ica_fetal_proxy_margin=float(bss_kwargs.get("ica_fetal_proxy_margin", 0.12)),
    )
    return {
        "S": S,
        "sm": sm,
        "sf": sf,
        "su": su,
        "im": int(im),
        "iff": int(iff),
        "iu": int(iu),
        "prox_m": prox_m,
        "prox_f": prox_f,
        "meta": meta,
        "x0": x0,
    }


def _chunk_fetal_ok(
    f: np.ndarray,
    fs: float,
    prox_f: np.ndarray,
    fb: Tuple[float, float],
    m: Optional[np.ndarray],
    bss_kwargs: dict,
) -> Tuple[bool, float]:
    """Waveform quality gate before accepting primary fetal IC."""
    sc = _score_fetal(f, fs, prox_f, fb, m, bss_kwargs)
    thr = float(bss_kwargs.get("chunk_quality_threshold", 0.35))
    n_peaks, _, med_bpm = _fetal_peak_metrics(f, fs)
    min_peaks = int(bss_kwargs.get("v1v6_chunk_min_peaks", 10))
    if sc < thr:
        return False, float(sc)
    if n_peaks < min_peaks:
        return False, float(sc)
    if med_bpm > 0 and (med_bpm < 95.0 or med_bpm > 200.0):
        return False, float(sc)
    return True, float(sc)


def pick_v1v6_fetal_with_rescue(
    x6_seg: np.ndarray,
    fs: float,
    chunk_sec: float,
    bss_kwargs: dict,
    prox_q: np.ndarray,
    fb: Tuple[float, float],
    *,
    prev_f: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float, int, Dict[str, float]]:
    """
    Returns m, f, u, route, score, rescue_level, route_scores.
    rescue_level: 0=primary ICA, 1=alt IC, 2=re-hedge ICA, 3=single2, 4=v6_focus.
    """
    L = int(x6_seg.shape[0])
    route_scores: Dict[str, float] = {}
    core = _v1v6_ica_core(x6_seg, fs, bss_kwargs, rng_seed=42)
    S, sm, sf = core["S"], core["sm"], core["sf"]
    im, iu = core["im"], core["iu"]
    prox_f = core["prox_f"]
    m = S[:, im].copy()
    u = S[:, core["iu"]].copy()

    def _finalize(f_raw: np.ndarray, route: str, level: int, sc: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float, int, Dict[str, float]]:
        f_out = np.asarray(f_raw, dtype=np.float64).ravel()[:L]
        if prev_f is not None and len(prev_f) >= 8:
            f_out = _match_sign(prev_f[-min(len(prev_f), L) :], f_out)
        return m[:L], f_out, u[:L], route, float(sc), int(level), route_scores

    iff0 = core["iff"]
    f0 = S[:, iff0].copy()
    ok0, sc0 = _chunk_fetal_ok(f0, fs, prox_q, fb, m, bss_kwargs)
    route_scores["v1v6_ica"] = float(sc0)
    if ok0:
        return _finalize(f0, "v1v6_ica", 0, sc0)

    # L1: alternate fetal IC from same decomposition
    best_l1: Optional[Tuple[float, int, np.ndarray]] = None
    for j in v1v6_fetal_ic_candidates(S, sf, im, iu, max_candidates=4):
        if j == iff0:
            continue
        fj = S[:, j].copy()
        okj, scj = _chunk_fetal_ok(fj, fs, prox_q, fb, m, bss_kwargs)
        route_scores[f"ica_ic{j}"] = float(scj)
        if okj and (best_l1 is None or scj > best_l1[0]):
            best_l1 = (float(scj), int(j), fj)
    if best_l1 is not None:
        sc_b, j_b, f_b = best_l1
        return _finalize(f_b, f"ica_ic{j_b}", 1, sc_b)

    # L2: re-run ICA with single hedge polarity (plus only)
    try:
        core2 = _v1v6_ica_core(
            x6_seg,
            fs,
            bss_kwargs,
            include_fetal_hedge_plus=True,
            include_fetal_hedge_minus=False,
            rng_seed=43,
        )
        S2 = core2["S"]
        im2, iu2 = core2["im"], core2["iu"]
        for j in v1v6_fetal_ic_candidates(S2, core2["sf"], im2, iu2, max_candidates=3):
            fj = S2[:, j].copy()
            okj, scj = _chunk_fetal_ok(fj, fs, prox_q, fb, m, bss_kwargs)
            route_scores[f"rehedge_ic{j}"] = float(scj)
            if okj:
                return _finalize(fj, "v1v6_rehedge", 2, scj)
    except Exception:
        pass

    # L3: single2 rescue
    f_s2 = single2_fetal_trace(x6_seg, fs, bss_kwargs)[:L]
    sc_s2 = _score_fetal(f_s2, fs, prox_q, fb, m, bss_kwargs)
    route_scores["single2_rescue"] = float(sc_s2)
    s2_ok = _single2_chunk_has_plausible_peaks(f_s2, fs)
    if s2_ok and sc_s2 >= float(bss_kwargs.get("chunk_quality_threshold", 0.35)):
        return _finalize(f_s2, "single2_rescue", 3, sc_s2)

    # L4: v6_focus ICA (legacy virtual input)
    if "v6_focus" in tuple(bss_kwargs.get("fetal_routes", FETAL_COMPARE_ROUTES)):
        try:
            x_in = _route_to_matrix(x6_seg, fs, "v6_focus", bss_kwargs)
            kw_l = dict(bss_kwargs)
            kw_l["separation_mode"] = "standard"
            kw_l["use_pca_maternal_fetal_stack"] = True
            _, f_v6, _ = _ica_once_on_chunk(x_in, fs, chunk_sec, kw_l)
            f_v6 = f_v6[:L]
            sc_v6 = _score_fetal(f_v6, fs, prox_q, fb, m, bss_kwargs)
            route_scores["v6_focus_rescue"] = float(sc_v6)
            if sc_v6 > sc_s2 and _fetal_chunk_amplitude_ok(f_v6, f_s2):
                return _finalize(f_v6, "v6_focus_rescue", 4, sc_v6)
        except Exception:
            pass

    if s2_ok:
        return _finalize(f_s2, "single2_rescue", 3, sc_s2)
    if best_l1 is not None:
        sc_b, j_b, f_b = best_l1
        return _finalize(f_b, f"ica_ic{j_b}", 1, sc_b)
    return _finalize(f0, "v1v6_ica", 0, sc0)


def v1v6_chunked_bss(
    x6: np.ndarray,
    fs: float,
    *,
    chunk_sec: float = 10.0,
    bss_kwargs: Optional[dict] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> dict:
    """Non-overlapping chunks with V1/V6 single ICA + rescue ladder per chunk."""
    bss_kwargs = dict(bss_kwargs or {})
    bss_kwargs["separation_mode"] = "v1v6_single_ica"
    n = int(x6.shape[0])
    chunk = max(256, int(chunk_sec * fs))
    hop = chunk

    maternal = np.zeros(n, dtype=np.float64)
    fetal = np.zeros(n, dtype=np.float64)
    uterine = np.zeros(n, dtype=np.float64)
    chunk_meta: List[dict] = []

    fb = tuple(bss_kwargs.get("fetal_band", (17.0, 42.0)))
    x0_full = preprocess_common_for_pipeline(x6, fs, bss_kwargs)
    prox_full = _bandpass(x0_full, fs, fb[0], fb[1])
    prox_full = 0.5 * (prox_full[:, 0] + prox_full[:, 5])

    starts = _chunk_start_indices(n, chunk, hop)
    prev_f: Optional[np.ndarray] = None

    for ci, start in enumerate(starts):
        end = min(start + chunk, n)
        seg = x6[start:end].copy()
        seg_len = end - start
        chunk_dur_sec = seg_len / fs
        prox_q = prox_full[start:end]

        m, f, u, route, score, rescue_level, route_scores = pick_v1v6_fetal_with_rescue(
            seg,
            fs,
            chunk_dur_sec,
            bss_kwargs,
            prox_q,
            fb,
            prev_f=prev_f,
        )
        prev_f = f.copy()

        sl = slice(start, end)
        maternal[sl] = m[:seg_len]
        fetal[sl] = f[:seg_len]
        uterine[sl] = u[:seg_len]

        chunk_meta.append(
            {
                "start_s": start / fs,
                "end_s": end / fs,
                "fetal_route": route,
                "fetal_score": float(score),
                "rescue_level": int(rescue_level),
                "route_scores": route_scores,
            }
        )
        if progress_callback is not None:
            progress_callback(ci + 1, len(starts), start, route, float(score))

    # Physics-first uterine envelope (C3/C4)
    u_phys = uterine_contraction_envelope(x0_full, fs)
    uterine = 0.93 * u_phys + 0.07 * (uterine - np.mean(uterine))

    # Maternal spatial blend
    x_m = _bandpass(x0_full, fs, float(bss_kwargs.get("maternal_ecg_band", (1.0, 45.0))[0]),
                    min(float(bss_kwargs.get("maternal_ecg_band", (1.0, 45.0))[1]), 0.48 * fs))
    m_sp = spatial_maternal_proxy(x_m)
    wmix = float(bss_kwargs.get("maternal_spatial_weight", 0.88))
    maternal = (1.0 - wmix) * maternal + wmix * m_sp

    post = bss_kwargs.get("fetal_post_band", (15.0, 45.0))
    fetal = _bandpass(fetal.reshape(-1, 1), fs, post[0], min(post[1], 0.48 * fs)).ravel()

    return {
        "maternal_ecg": maternal,
        "fetal_ecg": fetal,
        "uterine_abdominal": uterine,
        "fs": fs,
        "aux": {
            "chunk_meta": chunk_meta,
            "v1v6_single_ica": True,
        },
    }


def refine_v1v6_segment(
    x6_seg: np.ndarray,
    fs: float,
    chunk_sec: float,
    bss_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """FHR-jump bad window: full rescue ladder on segment."""
    fb = tuple(bss_kwargs.get("fetal_band", (17.0, 42.0)))
    x0 = preprocess_common_for_pipeline(x6_seg, fs, bss_kwargs)
    prox_q = _bandpass(x0, fs, fb[0], fb[1])
    prox_q = 0.5 * (prox_q[:, 0] + prox_q[:, 5])
    m, f, u, route, score, _, _ = pick_v1v6_fetal_with_rescue(
        x6_seg, fs, chunk_sec, bss_kwargs, prox_q, fb
    )
    return m, f, u, float(score), route


def _pop_v1v6_only_keys(kw: dict) -> None:
    _pop_hybrid_only_bss_keys(kw)
    for k in ("v1v6_chunk_min_peaks", "v1v6_single_ica", "v1v6_single_ica_refine"):
        kw.pop(k, None)
