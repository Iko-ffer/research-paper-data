#!/usr/bin/env python3
"""
Chunked (10s-style) overlap-add BSS + optional multi-route virtual-channel inputs.

Runs ``sliding_bss_three_outputs`` on time chunks to bound memory and latency,
with Hann overlap-add. Bad-quality chunks can retry alternate PCA-from-virtual stacks.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from sklearn.decomposition import PCA

from abdominal_bss_separate import (
    detect_ecg_peaks,
    sliding_bss_three_outputs,
    spatial_fetal_proxy,
    spatial_maternal_proxy,
    branch_ecg,
    preprocess_common_for_pipeline,
    _bandpass,
)


def _v1_v6_hedge_baseline(v1: np.ndarray, v6: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Baseline-removed V1/V6 and positive hedge scale ``k`` (single2-style)."""
    bl_win = int(fs * 0.8)
    if bl_win % 2 == 0:
        bl_win += 1
    bl_win = max(3, min(bl_win, len(v1) // 2 * 2 - 1))
    v1_c = v1 - medfilt(v1, bl_win)
    v6_c = v6 - medfilt(v6, bl_win)
    m_peaks, _ = find_peaks(v1_c, distance=int(fs * 0.5), prominence=np.std(v1_c) * 1.2 + 1e-9)
    win = int(fs * 0.02)
    v1_amps, v6_amps = [], []
    for p in m_peaks:
        a, b = max(0, p - win), min(len(v1_c), p + win)
        v1_amps.append(np.max(v1_c[a:b]))
        v6_amps.append(np.abs(np.min(v6_c[a:b])))
    k = float(np.median(v1_amps) / (np.median(v6_amps) + 1e-9)) if v1_amps else 1.0
    return v1_c, v6_c, k


def _v1_v6_bandpass_hedge(
    v1_c: np.ndarray,
    v6_c: np.ndarray,
    k: float,
    sign: float,
    fs: float,
    band_hz: Tuple[float, float] = (5.0, 40.0),
    order: int = 2,
) -> np.ndarray:
    """``v1_c + sign * k * v6_c`` then bandpass (default 5–40 Hz, same as single2)."""
    f_raw = v1_c + float(sign) * v6_c * float(k)
    nyq = 0.5 * fs
    lo = max(float(band_hz[0]), 0.5)
    hi = min(float(band_hz[1]), nyq - 1.0)
    ord_n = max(2, min(int(order), 6))
    b, a = butter(ord_n, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, f_raw)


def single2_bandpass_from_kwargs(
    bss_kwargs: Optional[Dict[str, object]],
) -> Tuple[Tuple[float, float], int]:
    """Resolve single2 hedge bandpass (Hz) and Butterworth order from pipeline kwargs."""
    kw = bss_kwargs or {}
    band = kw.get("single2_band_hz", (5.0, 40.0))
    if isinstance(band, (list, tuple)) and len(band) >= 2:
        bp = (float(band[0]), float(band[1]))
    else:
        bp = (5.0, 40.0)
    return bp, max(2, int(kw.get("single2_bandpass_order", 2)))


def adaptive_v1_v6_combine(
    v1: np.ndarray,
    v6: np.ndarray,
    fs: float,
    band_hz: Tuple[float, float] = (5.0, 40.0),
    bandpass_order: int = 2,
) -> np.ndarray:
    """V1 + K*V6 hedge with fixed ``+`` sign (legacy / multiroute virtual channel)."""
    v1_c, v6_c, k = _v1_v6_hedge_baseline(v1, v6, fs)
    return _v1_v6_bandpass_hedge(
        v1_c, v6_c, k, 1.0, fs, band_hz=band_hz, order=bandpass_order
    )


def _abs_corr(a: np.ndarray, b: np.ndarray, max_samples: int = 4000) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    if n < 32:
        return 0.0
    step = max(1, int(np.ceil(n / float(max_samples))))
    a = a[:n:step]
    b = b[:n:step]
    a = a - np.mean(a)
    a /= np.std(a) + 1e-9
    b = b - np.mean(b)
    b /= np.std(b) + 1e-9
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return abs(c) if np.isfinite(c) else 0.0


def _hedge_polarity_metrics(
    sig: np.ndarray,
    fs: float,
    prox_f: np.ndarray,
    prox_m: np.ndarray,
    fetal_band: Tuple[float, float],
) -> Dict[str, float]:
    fe = np.asarray(sig, dtype=np.float64).ravel()
    prox_f = np.asarray(prox_f, dtype=np.float64).ravel()
    prox_m = np.asarray(prox_m, dtype=np.float64).ravel()
    n = min(len(fe), len(prox_f), len(prox_m))
    if n < 32:
        return {
            "fetal_q": 0.0,
            "corr_maternal": 1.0,
            "n_peaks": 0.0,
            "peak_rate_hz": 0.0,
            "fetal_bpm_frac": 0.0,
            "maternal_bpm_frac": 0.0,
            "composite": 0.0,
        }
    fe = fe[:n]
    prox_f = prox_f[:n]
    prox_m = prox_m[:n]
    fq = float(
        fetal_route_quality(fe, fs, prox_f_precomputed=prox_f, fetal_band=fetal_band)
    )
    corr_m = _abs_corr(fe, prox_m)
    peaks = detect_ecg_peaks(fe, fs, 85.0, 215.0, height_factor=1.0, prominence_factor=0.08)
    n_peaks = int(len(peaks))
    dur = n / fs
    peak_rate = n_peaks / (dur + 1e-9)
    fetal_bpm_frac = 0.0
    maternal_bpm_frac = 0.0
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        bpm = 60.0 / rr
        ok = np.isfinite(bpm)
        bpm = bpm[ok]
        if bpm.size:
            fetal_bpm_frac = float(np.mean((bpm >= 105.0) & (bpm <= 190.0)))
            maternal_bpm_frac = float(np.mean((bpm >= 55.0) & (bpm <= 105.0)))
    penalty = (
        1.85 * corr_m
        + 2.4 * maternal_bpm_frac
        + 0.55 * max(0.0, peak_rate - 2.15)
    )
    bonus = 1.35 * fetal_bpm_frac
    if n_peaks < 4:
        penalty += 0.45
    composite = fq * max(0.12, 1.0 - 1.65 * corr_m) + bonus - penalty
    return {
        "fetal_q": fq,
        "corr_maternal": corr_m,
        "n_peaks": float(n_peaks),
        "peak_rate_hz": float(peak_rate),
        "fetal_bpm_frac": fetal_bpm_frac,
        "maternal_bpm_frac": maternal_bpm_frac,
        "composite": float(composite),
    }


def _hedge_polarity_disqualified(
    m: Dict[str, float],
    other: Dict[str, float],
) -> Tuple[bool, str]:
    """Reject hedge candidates dominated by maternal-rate peaks or maternal proxy."""
    if m["peak_rate_hz"] > 2.25 and m["fetal_bpm_frac"] < 0.38:
        return True, "high_peak_rate_low_fetal_bpm"
    if other["n_peaks"] > 3 and m["n_peaks"] > 3.2 * other["n_peaks"] and m["fetal_bpm_frac"] < 0.42:
        return True, "peak_count_surge_vs_other"
    if m["corr_maternal"] > 0.48 and m["corr_maternal"] > other["corr_maternal"] + 0.07:
        return True, "maternal_proxy_corr"
    if m["maternal_bpm_frac"] > 0.52 and m["maternal_bpm_frac"] > other["maternal_bpm_frac"] + 0.14:
        return True, "maternal_bpm_fraction"
    return False, ""


def pick_v1_v6_hedge_polarity(
    v1: np.ndarray,
    v6: np.ndarray,
    fs: float,
    prox_f: np.ndarray,
    prox_m: np.ndarray,
    *,
    fetal_band: Tuple[float, float] = (17.0, 42.0),
    single2_band_hz: Tuple[float, float] = (5.0, 40.0),
    single2_bandpass_order: int = 2,
    use_v2: bool = True,
    low_margin_fallback_plus: float = 0.10,
    minus_win_margin: float = 0.08,
) -> Tuple[np.ndarray, int, Dict[str, object]]:
    """
    Build V1±K·V6 hedge (bandpass on hedge sum) and pick sign.

    v2 (default): fetal proxy score + maternal-proxy suppression + BPM/peak-rate gates.
    v1 (legacy): higher ``fetal_route_quality`` only.
    """
    v1_c, v6_c, k = _v1_v6_hedge_baseline(v1, v6, fs)
    fp = _v1_v6_bandpass_hedge(
        v1_c, v6_c, k, 1.0, fs, band_hz=single2_band_hz, order=single2_bandpass_order
    )
    fm = _v1_v6_bandpass_hedge(
        v1_c, v6_c, k, -1.0, fs, band_hz=single2_band_hz, order=single2_bandpass_order
    )
    prox_f = np.asarray(prox_f, dtype=np.float64).ravel()
    prox_m = np.asarray(prox_m, dtype=np.float64).ravel()
    n = min(len(fp), len(fm), len(prox_f), len(prox_m))
    info: Dict[str, object] = {"hedge_k": float(k), "polarity_version": "v2" if use_v2 else "v1"}
    if n < 128:
        info["polarity_note"] = "short_segment_default_plus"
        return fp, 1, info
    fb_hi = min(float(fetal_band[1]), 0.48 * fs - 1.0)
    fb = (float(fetal_band[0]), fb_hi)
    pf = prox_f[:n]
    pm = prox_m[:n]
    fp_n = fp[:n]
    fm_n = fm[:n]

    if not use_v2:
        try:
            sc_p = fetal_route_quality(fp_n, fs, prox_f_precomputed=pf, fetal_band=fb)
            sc_m = fetal_route_quality(fm_n, fs, prox_f_precomputed=pf, fetal_band=fb)
        except Exception:
            info["polarity_note"] = "score_failed_default_plus"
            return fp, 1, info
        sign = 1 if sc_p >= sc_m else -1
        info.update(
            score_plus=float(sc_p),
            score_minus=float(sc_m),
            chosen_sign=sign,
            polarity_margin=float(abs(sc_p - sc_m)),
        )
        return (fp if sign > 0 else fm), sign, info

    mp = _hedge_polarity_metrics(fp_n, fs, pf, pm, fb)
    mm = _hedge_polarity_metrics(fm_n, fs, pf, pm, fb)
    dq_p, rq_p = _hedge_polarity_disqualified(mp, mm)
    dq_m, rq_m = _hedge_polarity_disqualified(mm, mp)
    info.update(
        score_plus=mp["fetal_q"],
        score_minus=mm["fetal_q"],
        composite_plus=mp["composite"],
        composite_minus=mm["composite"],
        corr_maternal_plus=mp["corr_maternal"],
        corr_maternal_minus=mm["corr_maternal"],
        fetal_bpm_frac_plus=mp["fetal_bpm_frac"],
        fetal_bpm_frac_minus=mm["fetal_bpm_frac"],
        maternal_bpm_frac_plus=mp["maternal_bpm_frac"],
        maternal_bpm_frac_minus=mm["maternal_bpm_frac"],
        peak_rate_plus=mp["peak_rate_hz"],
        peak_rate_minus=mm["peak_rate_hz"],
        disqualified_plus=dq_p,
        disqualified_minus=dq_m,
    )

    if dq_p and not dq_m:
        sign, note = -1, f"plus_disqualified:{rq_p}"
    elif dq_m and not dq_p:
        sign, note = 1, f"minus_disqualified:{rq_m}"
    elif dq_p and dq_m:
        if mm["composite"] > mp["composite"] + minus_win_margin:
            sign, note = -1, "both_disqualified_composite_minus"
        else:
            sign, note = 1, "both_disqualified_default_plus"
    else:
        margin = float(mm["composite"] - mp["composite"])
        info["composite_margin"] = margin
        if margin >= minus_win_margin:
            sign, note = -1, "composite_minus"
        elif margin <= -minus_win_margin:
            sign, note = 1, "composite_plus"
        else:
            sign, note = 1, "low_margin_default_plus"
        if sign < 0 and abs(margin) < low_margin_fallback_plus:
            sign, note = 1, "low_margin_default_plus"

    info["chosen_sign"] = sign
    info["polarity_margin"] = float(abs(mp["composite"] - mm["composite"]))
    info["polarity_note"] = note
    trace = fp if sign > 0 else fm
    return trace, sign, info


def adaptive_v1_v6_combine_best_polarity(
    v1: np.ndarray,
    v6: np.ndarray,
    fs: float,
    prox_f: np.ndarray,
    *,
    prox_m: Optional[np.ndarray] = None,
    fetal_band: Tuple[float, float] = (17.0, 42.0),
    use_v2: bool = True,
    single2_band_hz: Tuple[float, float] = (5.0, 40.0),
    single2_bandpass_order: int = 2,
) -> np.ndarray:
    """
    Hedge ``v1 ± k*v6`` then bandpass; pick sign (v2: fetal score + maternal suppression).

    Fetal QRS polarity relative to V1/V6 geometry can flip; a fixed ``+`` hedge can fail to cancel
    maternal energy and leaves fetal-like content in the maternal path downstream.
    """
    if prox_m is None:
        prox_m = np.zeros_like(prox_f, dtype=np.float64)
    trace, _sign, _info = pick_v1_v6_hedge_polarity(
        v1,
        v6,
        fs,
        prox_f,
        prox_m,
        fetal_band=fetal_band,
        single2_band_hz=single2_band_hz,
        single2_bandpass_order=single2_bandpass_order,
        use_v2=use_v2,
    )
    return trace


def diagnose_single2_v1v6_polarity(
    x6: np.ndarray,
    fs: float,
    bss_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Report V1±K·V6 hedge polarity choice (same logic as ``pick_v1_v6_hedge_polarity``).

    Helps cohort review when V1/V6 are in-phase vs opposite across subjects.
    """
    from scipy.signal import find_peaks

    bss_kwargs = dict(bss_kwargs or {})
    x6 = np.asarray(x6, dtype=np.float64)
    c0 = x6[:, 0]
    c5 = x6[:, 5]
    fb = tuple(bss_kwargs.get("fetal_band", (17.0, 42.0)))
    mb = bss_kwargs.get("maternal_ecg_band") or bss_kwargs.get("ecg_band") or (1.0, 45.0)
    x0 = preprocess_common_for_pipeline(x6, fs, bss_kwargs)
    xf = _bandpass(x0, fs, fb[0], fb[1])
    xm = branch_ecg(x0, fs, low=float(mb[0]), high=float(mb[1]))
    prox_f = spatial_fetal_proxy(xf)
    prox_m = spatial_maternal_proxy(xm)
    use_v2 = bool(bss_kwargs.get("single2_polarity_v2", True))
    s2_band, s2_ord = single2_bandpass_from_kwargs(bss_kwargs)
    _trace, sign, info = pick_v1_v6_hedge_polarity(
        c0,
        c5,
        fs,
        prox_f,
        prox_m,
        fetal_band=fb,
        single2_band_hz=s2_band,
        single2_bandpass_order=s2_ord,
        use_v2=use_v2,
    )
    n = min(len(_trace), len(prox_f))
    out: Dict[str, object] = dict(info)
    out["duration_sec"] = float(n / fs) if fs > 0 else 0.0
    out["single2_band_hz_low"] = s2_band[0]
    out["single2_band_hz_high"] = s2_band[1]
    out["single2_bandpass_order"] = s2_ord
    v1_c, v6_c, k = _v1_v6_hedge_baseline(c0, c5, fs)
    fp = _v1_v6_bandpass_hedge(v1_c, v6_c, k, 1.0, fs, band_hz=s2_band, order=s2_ord)[:n]
    fm = _v1_v6_bandpass_hedge(v1_c, v6_c, k, -1.0, fs, band_hz=s2_band, order=s2_ord)[:n]
    chosen = fp if sign > 0 else fm

    def _peak_count(sig: np.ndarray) -> int:
        s = sig - np.median(sig)
        std = float(np.std(s)) + 1e-12
        peaks, _ = find_peaks(np.abs(s), distance=max(3, int(fs * 0.28)), prominence=std * 0.5)
        return int(len(peaks))

    def _plausible(sig: np.ndarray) -> bool:
        s = sig - np.median(sig)
        std = float(np.std(s)) + 1e-12
        peaks, _ = find_peaks(np.abs(s), distance=max(3, int(fs * 0.28)), prominence=std * 0.5)
        if len(peaks) < 4:
            return False
        rr = np.diff(peaks) / fs
        bpm = 60.0 / rr
        m = np.isfinite(bpm) & (bpm >= 95.0) & (bpm <= 200.0)
        return bool(m.sum() >= 3 and float(np.median(bpm[m])) > 0)

    out["n_peaks_plus"] = _peak_count(fp)
    out["n_peaks_minus"] = _peak_count(fm)
    out["n_peaks_chosen"] = _peak_count(chosen)
    out["plausible_peaks_chosen"] = _plausible(chosen)
    return out


def _route_to_matrix(
    x6: np.ndarray,
    fs: float,
    route: str,
    bss_kwargs: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """Build (n,6) matrix for BSS: native 6ch or PCA-compress augmented stack."""
    n = x6.shape[0]
    c0, c1, c2, c3, c4, c5 = [x6[:, i] for i in range(6)]

    extras: List[np.ndarray] = []
    if route == "native":
        return x6.astype(np.float64, copy=False)
    if route == "v1_focus":
        z = np.zeros_like(x6, dtype=np.float64)
        z[:, 0] = np.asarray(c0, dtype=np.float64)
        return z
    if route == "v6_focus":
        z = np.zeros_like(x6, dtype=np.float64)
        z[:, 5] = np.asarray(c5, dtype=np.float64)
        return z
    if route == "v16_band":
        extras.append(adaptive_v1_v6_combine(c0, c5, fs))
    if route == "bipolar_ladder":
        extras.extend(
            [
                c0 - c1,
                c0 - c2,
                c5 - c4,
                c5 - c3,
            ]
        )
    if route == "v16_plus_ladder":
        extras.append(adaptive_v1_v6_combine(c0, c5, fs))
        extras.extend([c0 - c1, c0 - c2, c5 - c4, c5 - c3])
    if route == "z5_vref":
        kw = bss_kwargs or {}
        w0 = float(kw.get("vref_weight_ch0", 1.0))
        w5 = float(kw.get("vref_weight_ch5", 1.0))
        extras.append(w0 * np.asarray(c0, dtype=np.float64) + w5 * np.asarray(c5, dtype=np.float64) - np.asarray(c1, dtype=np.float64))

    if not extras:
        return x6.astype(np.float64, copy=False)

    X = np.column_stack([x6] + extras)
    X = X - np.mean(X, axis=0, keepdims=True)
    n_comp = min(6, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    z = pca.fit_transform(X)
    if z.shape[1] < 6:
        pad = np.zeros((n, 6 - z.shape[1]), dtype=np.float64)
        z = np.hstack([z, pad])
    return z.astype(np.float64)


def fetal_route_quality(
    fetal_ecg: np.ndarray,
    fs: float,
    x6_for_proxy: Optional[np.ndarray] = None,
    *,
    prox_f_precomputed: Optional[np.ndarray] = None,
    fetal_band: Tuple[float, float] = (17.0, 42.0),
    corr_max_samples: int = 4000,
    bss_kwargs: Optional[Dict] = None,
) -> float:
    """
    Unsupervised score: fetal peaks plausibility + corr with spatial fetal proxy.
    Higher is better.

    When ``prox_f_precomputed`` is set (same length as fetal trace), skips
    rebuilding the proxy from ``x6_for_proxy`` — used for multi-route chunk scoring.
    Correlation uses a strided subsample for speed; peak detection uses full length.
    """
    if prox_f_precomputed is not None:
        prox = np.asarray(prox_f_precomputed, dtype=np.float64).ravel()
    else:
        if x6_for_proxy is None:
            raise ValueError("fetal_route_quality: need x6_for_proxy or prox_f_precomputed")
        x0 = preprocess_common_for_pipeline(x6_for_proxy, fs, bss_kwargs)
        xf = _bandpass(x0, fs, fetal_band[0], fetal_band[1])
        prox = spatial_fetal_proxy(xf)
        prox = np.asarray(prox, dtype=np.float64).ravel()

    prox = prox - np.mean(prox)
    prox /= np.std(prox) + 1e-9
    fe_raw = np.asarray(fetal_ecg, dtype=np.float64).ravel()
    n = min(len(fe_raw), len(prox))
    if n < 32:
        return 0.0
    fe_raw = fe_raw[:n]
    prox = prox[:n]
    step = max(1, int(np.ceil(n / float(corr_max_samples))))
    fe_s = fe_raw[::step]
    prox_s = prox[::step]
    a = fe_s - np.mean(fe_s)
    a /= np.std(a) + 1e-9
    b = prox_s - np.mean(prox_s)
    b /= np.std(b) + 1e-9
    corr = float(np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    if not np.isfinite(corr):
        corr = 0.0

    peaks = detect_ecg_peaks(fe_raw, fs, 95.0, 210.0, height_factor=1.0, prominence_factor=0.08)
    if len(peaks) < 3:
        return corr * 0.5
    rr = np.diff(peaks) / fs
    bpm = 60.0 / rr
    frac_ok = float(np.mean((bpm >= 100) & (bpm <= 190)))
    rate_score = min(1.0, len(peaks) / (len(fe_raw) / fs * 2.5 + 1e-9))
    return corr * 1.2 + frac_ok * 1.5 + rate_score * 0.3


def _chunk_start_indices(n: int, chunk: int, hop: int) -> List[int]:
    if n <= 0 or chunk <= 0:
        return []
    if n <= chunk:
        return [0]
    starts: List[int] = []
    s = 0
    while s + chunk < n:
        starts.append(s)
        s += hop
    last = n - chunk
    if not starts or starts[-1] != last:
        starts.append(last)
    # dedupe ordered
    out: List[int] = []
    for x in starts:
        if not out or x != out[-1]:
            out.append(x)
    return out


def sliding_bss_three_outputs_quiet(x6: np.ndarray, fs: float, **kwargs) -> dict:
    kw = dict(kwargs)
    kw["verbose"] = False
    return sliding_bss_three_outputs(x6, fs, **kw)


def chunked_multiroute_bss(
    x6: np.ndarray,
    fs: float,
    *,
    chunk_sec: float = 10.0,
    chunk_hop_sec: float = 8.0,
    multiroute: bool = True,
    quality_threshold: float = 0.35,
    routes_good: Tuple[str, ...] = ("native",),
    routes_fallback: Tuple[str, ...] = (
        "native",
        "v16_band",
        "bipolar_ladder",
        "v16_plus_ladder",
        "z5_vref",
    ),
    bss_kwargs: Optional[dict] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> dict:
    """
    Overlap-add sliding BSS on long records.

    Parameters
    ----------
    chunk_sec, chunk_hop_sec
        Outer window and hop (e.g. 10s / 8s => 2s overlap).
    multiroute
        If True, retry fallback routes on chunks whose native ``fetal_route_quality`` is below
        ``quality_threshold`` (default: on, same strategy as RUN-20260515-151221). Set False for speed.
    """
    bss_kwargs = dict(bss_kwargs or {})
    n = int(x6.shape[0])
    chunk = max(256, int(chunk_sec * fs))
    hop = max(1, int(chunk_hop_sec * fs))
    if hop >= chunk:
        hop = max(1, chunk // 5)

    maternal_acc = np.zeros(n, dtype=np.float64)
    fetal_acc = np.zeros(n, dtype=np.float64)
    uterine_acc = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)

    starts = _chunk_start_indices(n, chunk, hop)
    n_chunks = len(starts)

    for ci, start in enumerate(starts):
        end = min(start + chunk, n)
        seg_len = end - start
        seg = np.zeros((chunk, 6), dtype=np.float64)
        seg[:seg_len] = x6[start:end]

        routes = routes_good if not multiroute else routes_fallback
        best_out: Optional[dict] = None
        best_score = -1e18
        tried_native_score: Optional[float] = None
        single_route = len(routes) <= 1
        fb = bss_kwargs.get("fetal_band", (17.0, 42.0))
        prox_for_q: Optional[np.ndarray] = None
        if not single_route:
            seg_active = seg[:seg_len]
            x0q = preprocess_common_for_pipeline(seg_active, fs, bss_kwargs)
            xf_q = _bandpass(x0q, fs, fb[0], fb[1])
            prox_for_q = spatial_fetal_proxy(xf_q)

        for ri, route in enumerate(routes):
            x_in = _route_to_matrix(seg, fs, route, bss_kwargs)
            try:
                out = sliding_bss_three_outputs_quiet(x_in[:seg_len], fs, **bss_kwargs)
                m = np.asarray(out["maternal_ecg"], dtype=np.float64).ravel()
                f = np.asarray(out["fetal_ecg"], dtype=np.float64).ravel()
                if len(m) < seg_len:
                    continue
                m = m[:seg_len]
                f = f[:seg_len]
                u = np.asarray(out["uterine_abdominal"], dtype=np.float64).ravel()[:seg_len]
                if single_route:
                    score = 0.0
                else:
                    score = fetal_route_quality(
                        f, fs, prox_f_precomputed=prox_for_q, fetal_band=fb
                    )
                if route == "native":
                    tried_native_score = score
                if score > best_score:
                    best_score = score
                    best_out = {
                        "maternal_ecg": m.copy(),
                        "fetal_ecg": f.copy(),
                        "uterine_abdominal": u.copy(),
                        "route": route,
                        "score": score,
                    }
            except Exception:
                continue

            if multiroute and route == "native" and tried_native_score is not None and tried_native_score >= quality_threshold:
                break

        if best_out is None:
            raise RuntimeError(f"BSS failed on chunk starting {start}")

        m = best_out["maternal_ecg"]
        f = best_out["fetal_ecg"]
        u = best_out["uterine_abdominal"]
        win = np.hanning(seg_len)
        sl = slice(start, end)
        maternal_acc[sl] += m * win
        fetal_acc[sl] += f * win
        uterine_acc[sl] += u * win
        wsum[sl] += win

        if progress_callback:
            progress_callback(ci + 1, n_chunks, start, best_out.get("route", "?"), float(best_score))

    wsum[wsum < 1e-12] = 1.0
    maternal = maternal_acc / wsum
    fetal = fetal_acc / wsum
    uterine = uterine_acc / wsum

    x0 = preprocess_common_for_pipeline(x6, fs, bss_kwargs)
    mb = bss_kwargs.get("maternal_ecg_band") or bss_kwargs.get("ecg_band")
    if mb is None:
        mb = (1.0, 45.0)
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
        },
    }


def _refine_route_composite_score(
    f: np.ndarray,
    m: np.ndarray,
    fs: float,
    prox_q: np.ndarray,
    fb: Tuple[float, float],
    bss_kwargs: dict,
) -> float:
    """
    Bad-segment multiroute score: fetal_route_quality minus weighted fetal–maternal correlation
    in the fetal band (down-ranks routes whose fetal trace tracks maternal QRS in that band).
    """
    base = fetal_route_quality(f, fs, prox_f_precomputed=prox_q, fetal_band=fb)
    wt = float(bss_kwargs.get("refine_penalize_fetal_maternal_corr", 0.0))
    if wt <= 0.0 or len(f) < 64:
        return base
    lo, hi = float(fb[0]), min(float(fb[1]), 0.45 * fs)
    L = min(int(len(f)), int(len(m)))
    if L < 64:
        return base
    mfb = _bandpass(m[:L].reshape(-1, 1), fs, lo, hi, order=3).ravel()
    fz = f[:L].astype(np.float64) - np.mean(f[:L])
    fz = fz / (float(np.std(fz)) + 1e-9)
    mfb = mfb - np.mean(mfb)
    mfb = mfb / (float(np.std(mfb)) + 1e-9)
    try:
        cr = float(np.abs(np.corrcoef(fz, mfb)[0, 1]))
    except Exception:
        cr = 0.0
    if not np.isfinite(cr):
        cr = 0.0
    return base - wt * cr


def refine_segment_pick_best_route(
    x6_seg: np.ndarray,
    fs: float,
    bss_kwargs: dict,
    routes: Tuple[str, ...] = (
        "native",
        "v16_band",
        "bipolar_ladder",
        "v16_plus_ladder",
        "z5_vref",
    ),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Run sliding BSS on a short slice with several virtual-input routes; pick highest
    composite score (fetal_route_quality minus maternal–fetal-band correlation penalty when enabled).
    """
    best_score = -1e18
    best: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    best_route = "none"
    L = int(x6_seg.shape[0])
    if L < 64:
        raise ValueError("segment too short for refine")
    score_kw = dict(bss_kwargs)
    kw = dict(bss_kwargs)
    kw.pop("refine_penalize_fetal_maternal_corr", None)
    fb = kw.get("fetal_band", (17.0, 42.0))
    x0q = preprocess_common_for_pipeline(x6_seg, fs, kw)
    xf_q = _bandpass(x0q, fs, fb[0], fb[1])
    prox_q = spatial_fetal_proxy(xf_q)
    for route in routes:
        x_in = _route_to_matrix(x6_seg, fs, route, kw)
        try:
            out = sliding_bss_three_outputs_quiet(x_in, fs, **kw)
            m = np.asarray(out["maternal_ecg"], dtype=np.float64).ravel()
            f = np.asarray(out["fetal_ecg"], dtype=np.float64).ravel()
            u = np.asarray(out["uterine_abdominal"], dtype=np.float64).ravel()
            if len(f) < L:
                continue
            m, f, u = m[:L], f[:L], u[:L]
            sc = _refine_route_composite_score(f, m, fs, prox_q, fb, score_kw)
            if sc > best_score:
                best_score = sc
                best = (m.copy(), f.copy(), u.copy())
                best_route = route
        except Exception:
            continue
    if best is None:
        raise RuntimeError("refine_segment_pick_best_route: all routes failed")
    m, f, u = best
    return m, f, u, float(best_score), best_route
