#!/usr/bin/env python3
"""
Integrated six-channel abdominal signal pipeline.

Combines BSS separation (abdominal_bss_separate), gold-standard FHR alignment
(single2), six-channel energy envelopes with spike suppression, and a
four-panel QC figure per segment.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, hilbert, medfilt

from abdominal_bss_separate import (
    detect_ecg_peaks,
    instantaneous_hr_from_peaks,
    load_csv_6ch,
    preprocess_common_for_pipeline,
    sliding_bss_three_outputs,
    subtract_fetal_band_leakage_from_maternal,
)

from chunked_multiroute_bss import chunked_multiroute_bss, diagnose_single2_v1v6_polarity
from fhr_bad_refinement import apply_fhr_bad_segment_refinement, detect_fhr_jump_intervals
from hybrid_v16_chunk_bss import FETAL_COMPARE_ROUTES, hybrid_v16_chunked_bss
from v1v6_chunk_bss import v1v6_chunked_bss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    fs_hint: float = 500.0
    gold_fs: float = 5.0
    window_sec: float = 10.0
    overlap: float = 0.75
    template_cancel: bool = True
    blend_maternal: float = 0.88
    blend_fetal: float = 0.42
    uterine_envelope_weight: float = 0.93
    max_sync_scan_sec: int = 120
    plot_preview_sec: float = 10.0
    energy_band_hz: Tuple[float, float] = (0.08, 10.0)
    energy_fast_smooth_sec: float = 0.35
    energy_slow_smooth_sec: float = 2.0
    energy_hampel_window_sec: float = 2.0
    energy_hampel_n_sigmas: float = 4.0
    energy_downsample_hz: float = 5.0
    # Half-width (samples on energy grid) for rolling ``|Δ fused|`` contraction proxy (~0.2 s/sample @ 5 Hz).
    energy_fused_osc_halfwin: int = 2
    quiet: bool = False
    # V1/V6 single ICA (7-col obs + hedge); 10 s chunks with rescue ladder (preferred over hybrid).
    use_v1v6_single_ica: bool = False
    # Hybrid V1/V6: 10 s non-overlapping chunks, single2 vs focused ICA for fetal.
    use_hybrid_v16_fetal: bool = True
    fetal_chunk_routes: Tuple[str, ...] = FETAL_COMPARE_ROUTES
    hybrid_ica_override_margin: float = 0.15
    hybrid_single2_rescue_threshold: float = 2.05
    # V1/V6 hedge: pick ``+k`` vs ``-k`` on V6 using fetal proxy score (recommended on non-代淼 subjects).
    single2_dual_polarity: bool = True
    # v2: fetal score + maternal-proxy suppression + BPM/peak-rate gates (see pick_v1_v6_hedge_polarity).
    single2_polarity_v2: bool = True
    # Single2 hedge bandpass (fetal path only; ``preprocess_common`` HP stays ~0.35 Hz for maternal).
    single2_band_hz: Tuple[float, float] = (5.0, 40.0)
    single2_bandpass_order: int = 2
    # Before FHR ``find_peaks``: rolling median detrend (suppress UC baseline wander on fECG).
    fecg_fhr_detrend_median_sec: float = 0.0
    # Optional extra bandpass on fECG used only for peak-pick (e.g. 8–40 Hz, order 4).
    fecg_fhr_bandpass_hz: Optional[Tuple[float, float]] = None
    fecg_fhr_bandpass_order: int = 4
    # Peak-pick BPM windows (keep a gap between maternal and fetal to reduce cross-leak).
    mhr_peak_bpm_hz: Tuple[float, float] = (45.0, 100.0)
    fhr_peak_bpm_hz: Tuple[float, float] = (110.0, 170.0)
    # Minimum BPM gap: fetal_min must be >= maternal_max + gap (enforced if windows overlap).
    peak_bpm_guard_gap_bpm: float = 8.0
    # Legacy full 6 ch overlap-add chunks (off when hybrid is on).
    use_chunked: bool = False
    chunk_sec: float = 10.0
    chunk_hop_sec: float = 10.0
    # Match quality-oriented runs (e.g. RUN-20260515-151221): try fallback virtual-input routes
    # when native chunk score is below chunk_quality_threshold. For speed use CLI --no-multiroute.
    multiroute: bool = True
    chunk_quality_threshold: float = 0.35
    # Inner sliding BSS hop (hybrid: one ICA per chunk => hop == window == chunk_sec).
    bss_inner_hop_sec: Optional[float] = 10.0
    # FHR on gold grid: causal mean of instantaneous BPM over fhr_smooth_sec
    fhr_output_hz: float = 5.0
    fhr_smooth_sec: float = 2.5
    fetal_band_hz: Tuple[float, float] = (17.0, 42.0)
    maternal_ecg_band_hz: Tuple[float, float] = (1.0, 45.0)
    fetal_post_band_hz: Tuple[float, float] = (15.0, 45.0)
    # Post-hoc fetal-band subtraction on maternal (helps when mECG still carries fQRS after split ICA)
    maternal_defetal: bool = True
    maternal_defetal_beta_clip: float = 2.5
    # ICA-side: down-rank ICs that correlate with fetal spatial proxy; tighten spatial blend if stacked maternal ICA looks fetal
    maternal_penalize_fetal_proxy: float = 1.75
    maternal_ica_fetal_corr_block_thr: float = 0.34
    # Fetal trace: penalize ICs that match maternal spatial proxy; optional band-limited maternal regression
    fetal_penalize_maternal_proxy: float = 1.35
    fetal_orthogonalize_maternal: bool = True
    fetal_orthogonalize_beta_max: float = 0.36
    # Raw 6-channel conditioning before dual-band / ICA (impedance scaling, wander, mains)
    preprocess_notch_50: bool = True
    preprocess_notch_100: bool = True
    preprocess_baseline_highpass_hz: float = 0.35
    preprocess_baseline_hp_order: int = 2
    preprocess_per_channel_scale: str = "robust"  # none | zscore | robust
    # 5 Hz FHR: suppress spurious RR / single-grid spikes (does not change fECG trace)
    fhr_inst_outlier_max_bpm: float = 18.0
    fhr_postmedian_halfwin: int = 1
    fhr_peak_height_factor: float = 1.05
    fhr_peak_prominence_factor: float = 0.11
    # After causal 5 Hz FHR: optional repair of large adjacent jumps, then median-smooth (0 disables each)
    fhr_output_despike_gap_bpm: float = 6.0
    fhr_output_jump_smooth_bpm: float = 10.0
    fhr_output_jump_smooth_win: int = 5
    # Wide rolling median reconcile (0 = off in hybrid mode; bad-segment refine handles plateaus).
    fhr_output_slow_reconcile_halfwin: int = 0
    fhr_output_slow_reconcile_margin_bpm: float = 15.0
    # Final symmetric neighbor median on the 5 Hz grid (0 = off).
    fhr_output_final_median_halfwin: int = 3
    # Multiroute ``z5_vref``: w0*ch0 + w5*ch5 - ch1 then PCA->6 like other augmented stacks
    vref_weight_ch0: float = 1.0
    vref_weight_ch5: float = 1.0
    # Per-slice ICA: scale maternal-band / fetal-band columns before z-score + PCA stack (ch1/ch6 = indices 0,5).
    use_ica_obs_spatial_guided_weights: bool = False
    ica_obs_maternal_channel_weights: Optional[Tuple[float, ...]] = None
    ica_obs_fetal_channel_weights: Optional[Tuple[float, ...]] = None
    # PCA stack: append V1±V6 bipoles per band → 16-D before PCA→6 (lateral common/differential structure).
    ica_obs_append_v1v6_bipoles: bool = False
    # Optional fetal-band ICA obs extras (off by default; baseline RUN-123345 uses ``standard`` mode).
    ica_obs_append_adaptive_hedge: bool = False
    ica_obs_append_middle_bipoles: bool = False
    # Fetal-band ICA on 6ch + single2-style V1±K·V6 hedge obs; single2 gates output (see --fetal-ica-hedge-obs).
    hybrid_fetal_hedge_ica_obs: bool = False
    # Unified band for hedge ICA obs + proxy + post filter when ``hybrid_fetal_hedge_ica_obs`` (single2-like).
    ica_fetal_obs_band_hz: Tuple[float, float] = (5.0, 40.0)
    hybrid_ica_gate_min_peaks: int = 12
    hybrid_ica_gate_peak_ratio: float = 0.55
    # If False: 5 Hz FHR stays NaN between peaks (trend visible; MAE vs gold may drop).
    fhr_fill_5hz_grid: bool = True
    ica_split_maternal_v1v6: bool = True
    ica_split_fetal_v1v6: bool = False
    hybrid_fetal_ica_gating: bool = False
    hybrid_single2_gate_margin: float = 0.0
    hybrid_fetal_score_fusion: bool = False
    hybrid_fetal_blend_score_margin: float = 0.12
    hybrid_maternal_score_fusion: bool = False
    hybrid_maternal_blend_score_margin: float = 0.12
    # FHR from fused fECG: False = single2-style interp+medfilt on fused trace (continuous 5 Hz).
    # True = causal window only where peaks exist (sparse; not recommended for MAE vs gold).
    fhr_from_fused_fecg: bool = False
    # Bad-window refine when mHR has long invalid runs (e.g. lost maternal peaks).
    refine_on_mhr_gap: bool = False
    mhr_gap_trigger_sec: float = 3.0
    bad_refine_fhr_window_sec: float = 15.0
    bad_refine_mhr_penalize_fetal_boost: float = 0.55
    bad_refine_neighbor_bridge_sec: float = 2.0
    # separation_mode: ``standard`` = RUN-123345 baseline (12→6 joint PCA); ``ica_split`` for experiments.
    separation_mode: str = "standard"
    physics_first_maternal_wmix_floor: float = 0.94
    # ICA: if fetal IC tracks maternal proxy more than fetal proxy, try alternate fetal IC
    ica_fetal_proxy_margin: float = 0.12
    # Bad-segment multiroute refine: penalize fetal that correlates with maternal in fetal band
    refine_penalize_fetal_maternal_corr: float = 1.15
    # Bad segment = 5 Hz FHR jump; re-BSS slice [t-pre, t+post] with multiroute, splice best
    use_fhr_jump_bad_refinement: bool = True
    fhr_jump_bpm: float = 5.0
    fhr_bad_pre_sec: float = 1.0
    fhr_bad_post_sec: float = 9.0
    # Max FHR-jump refine passes per segment (prevents infinite re-BSS loops).
    fhr_bad_refine_max_passes: int = 2
    bad_segment_blend_sec: float = 0.25
    bad_segment_min_sec: float = 0.8
    export_fhr_peak_series: bool = True


# ---------------------------------------------------------------------------
# Gold CSV
# ---------------------------------------------------------------------------


def load_gold_csv(gold_path: str, gold_fs: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time_s, FHR, UC from gold CSV."""
    df = pd.read_csv(gold_path)
    fhr = pd.to_numeric(df.iloc[:, 1], errors="coerce").ffill().bfill().values
    uc = pd.to_numeric(df.iloc[:, 2], errors="coerce").ffill().bfill().values
    if df.shape[1] > 0 and "RelTime" in str(df.columns[0]):
        t_ms = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        t_sec = t_ms / 1000.0
    else:
        t_sec = np.arange(len(fhr)) / gold_fs
    return t_sec, fhr, uc


def _fhr_inst_window_robust_mean(bpm: np.ndarray, max_dev_bpm: float) -> float:
    """Mean of instantaneous BPM in window, dropping points far from the median (spurious RR)."""
    bpm = np.asarray(bpm, dtype=np.float64).ravel()
    bpm = bpm[np.isfinite(bpm)]
    if bpm.size == 0:
        return float("nan")
    if bpm.size == 1:
        return float(bpm[0])
    if max_dev_bpm <= 0:
        return float(np.mean(bpm))
    med = float(np.median(bpm))
    sel = bpm[np.abs(bpm - med) <= float(max_dev_bpm)]
    if sel.size == 0:
        return med
    return float(np.mean(sel))


def _fhr_5hz_neighbor_median_smooth(
    fhr: np.ndarray,
    valid: np.ndarray,
    halfwin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Light symmetric median along time on valid samples only (reduces single-grid spikes)."""
    if halfwin <= 0:
        return fhr, valid
    n = len(fhr)
    out = fhr.copy()
    v2 = valid.copy()
    for k in range(n):
        if not valid[k] or not np.isfinite(fhr[k]):
            continue
        lo, hi = max(0, k - halfwin), min(n, k + halfwin + 1)
        seg = fhr[lo:hi]
        m = valid[lo:hi] & np.isfinite(seg)
        if np.sum(m) < 2:
            continue
        out[k] = float(np.median(seg[m]))
    return out, v2


def despike_fhr_adjacent_jumps(
    fhr: np.ndarray,
    valid: np.ndarray,
    *,
    gap_bpm: float,
    max_passes: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    On the 5 Hz grid: when two consecutive **valid** samples differ by more than ``gap_bpm``,
    replace the later index with a short-window median of valid neighbors; repeat until stable.
    """
    if gap_bpm <= 0:
        return fhr, valid
    y = np.asarray(fhr, dtype=np.float64).copy()
    v = np.asarray(valid, dtype=bool).copy()
    n = len(y)
    if n < 3:
        return y, v
    half = 2

    for _ in range(max_passes):
        worst_i = -1
        worst_d = 0.0
        for i in range(1, n):
            if not (v[i] and v[i - 1]) or not (np.isfinite(y[i]) and np.isfinite(y[i - 1])):
                continue
            d = abs(y[i] - y[i - 1])
            if d > gap_bpm and d > worst_d:
                worst_d = d
                worst_i = i
        if worst_i < 0:
            break
        lo = max(0, worst_i - half)
        hi = min(n, worst_i + half + 1)
        seg = y[lo:hi]
        m = v[lo:hi] & np.isfinite(seg)
        if np.sum(m) >= 1:
            y[worst_i] = float(np.median(seg[m]))
        else:
            y[worst_i] = y[worst_i - 1]
    return y, v


def fused_six_channel_energy_1d(env_ds: np.ndarray) -> np.ndarray:
    """Per-time: each channel scaled by median(|x|), then mean over channels (shape (T,))."""
    z = np.asarray(env_ds, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 1:
        return np.zeros(0, dtype=np.float64)
    env_norm = z.copy()
    for ch in range(min(6, env_norm.shape[1])):
        col = env_norm[:, ch]
        scale = float(np.nanmedian(np.abs(col))) + 1e-12
        env_norm[:, ch] = col / scale
    return np.mean(env_norm[:, :6], axis=1)


def fused_energy_oscillation_proxy(fused: np.ndarray, *, halfwin: int = 2) -> np.ndarray:
    """
    Contraction-oriented proxy: smooth rolling mean of ``|d fused / dt|`` on the energy grid.
    Emphasizes the '锯齿' / high-frequency wobble seen on fused energy during UC peaks.
    """
    x = np.asarray(fused, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    d[0] = abs(x[0] - x[1])
    for i in range(1, n):
        d[i] = abs(x[i] - x[i - 1])
    w = max(3, 2 * int(halfwin) + 1)
    if w % 2 == 0:
        w += 1
    return uniform_filter1d(d, size=w, mode="nearest")


def smooth_fhr_on_output_jumps(
    fhr: np.ndarray,
    valid: np.ndarray,
    *,
    jump_bpm: float,
    med_win: int,
    max_passes: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-process algorithm FHR on the output grid: while any two consecutive **valid**
    samples differ by more than ``jump_bpm``, apply a short symmetric median filter
    (SciPy ``median_filter``) to the series and repeat. Invalid samples are left unchanged.
    """
    if jump_bpm <= 0:
        return fhr, valid
    y = np.asarray(fhr, dtype=np.float64).copy()
    v = np.asarray(valid, dtype=bool).copy()
    n = len(y)
    if n < 2:
        return y, v
    win = int(med_win)
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1

    def max_adj_jump() -> float:
        mx = 0.0
        for i in range(1, n):
            if v[i] and v[i - 1] and np.isfinite(y[i]) and np.isfinite(y[i - 1]):
                mx = max(mx, abs(y[i] - y[i - 1]))
        return mx

    mask_f = v & np.isfinite(y)
    ref = float(np.nanmedian(y[mask_f])) if np.any(mask_f) else 120.0

    for _ in range(max_passes):
        if max_adj_jump() <= jump_bpm:
            break
        y_fill = np.where(np.isfinite(y), y, ref)
        y_sm = median_filter(y_fill, size=win, mode="nearest")
        y = np.where(v, y_sm, y)
    return y, v


def reconcile_fhr_slow_rolling_median(
    fhr: np.ndarray,
    valid: np.ndarray,
    *,
    halfwin: int,
    margin_bpm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Snap samples that disagree with a **wide** time-centered rolling median by at least
    ``margin_bpm``. Helps when despike / short median-smooth leave a flat wrong plateau for
    several seconds while surrounding context is plausible.
    """
    if halfwin <= 0 or margin_bpm <= 0:
        return fhr, valid
    y = np.asarray(fhr, dtype=np.float64).copy()
    v = np.asarray(valid, dtype=bool).copy()
    n = len(y)
    if n < 3:
        return y, v
    hw = int(halfwin)
    win = 2 * hw + 1
    min_periods = max(5, min(win, win // 2))
    y_masked = np.where(v & np.isfinite(y), y, np.nan)
    slow = (
        pd.Series(y_masked)
        .rolling(window=win, center=True, min_periods=min_periods)
        .median()
        .to_numpy(dtype=np.float64, copy=False)
    )
    ok = v & np.isfinite(y) & np.isfinite(slow)
    chg = ok & (np.abs(y - slow) >= float(margin_bpm))
    y[chg] = slow[chg]
    return y, v


def median_rr_bpm_from_trace(
    sig: np.ndarray,
    fs: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
) -> float:
    """Median instantaneous BPM from peaks in ``[hr_min_bpm, hr_max_bpm]`` (diagnostic)."""
    peaks = detect_ecg_peaks(
        np.asarray(sig, dtype=np.float64).ravel(),
        fs,
        float(hr_min_bpm),
        float(hr_max_bpm),
        height_factor=1.05,
        prominence_factor=0.11,
    )
    if len(peaks) < 3:
        return float("nan")
    rr = np.diff(peaks) / fs
    bpm = 60.0 / rr
    m = np.isfinite(bpm) & (bpm >= float(hr_min_bpm)) & (bpm <= float(hr_max_bpm))
    if not np.any(m):
        return float("nan")
    return float(np.median(bpm[m]))


def resolve_peak_bpm_windows(config: PipelineConfig) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
  Return (mhr_lo, mhr_hi), (fhr_lo, fhr_hi) with optional guard gap if bands overlap.

  Target clinical bands: maternal ~45–100(–120) bpm, fetal ~105–170 bpm.
  """
    m_lo, m_hi = float(config.mhr_peak_bpm_hz[0]), float(config.mhr_peak_bpm_hz[1])
    f_lo, f_hi = float(config.fhr_peak_bpm_hz[0]), float(config.fhr_peak_bpm_hz[1])
    gap = float(config.peak_bpm_guard_gap_bpm)
    if m_hi + gap > f_lo:
        m_hi = min(m_hi, f_lo - gap)
    if m_hi <= m_lo:
        m_hi = m_lo + max(5.0, gap)
    return (m_lo, m_hi), (f_lo, f_hi)


def fhr_peak_pick_kwargs_from_config(config: PipelineConfig) -> Dict[str, object]:
    """Kwargs for ``compute_fhr_5hz_single2_style*`` peak conditioning."""
    bp = config.fecg_fhr_bandpass_hz
    if bp is not None:
        bp_t = (float(bp[0]), float(bp[1]))
    else:
        bp_t = None
    _m, (f_lo, f_hi) = resolve_peak_bpm_windows(config)
    return {
        "detrend_median_sec": float(config.fecg_fhr_detrend_median_sec),
        "fhr_bandpass_hz": bp_t,
        "fhr_bandpass_order": int(config.fecg_fhr_bandpass_order),
        "hr_min_bpm": f_lo,
        "hr_max_bpm": f_hi,
    }


def rolling_median_detrend_1d(sig: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
    """Remove slow baseline via rolling median (single2-style local wander suppression)."""
    x = np.asarray(sig, dtype=np.float64).ravel()
    if window_sec <= 0 or len(x) < 8:
        return x - np.median(x)
    win = int(float(window_sec) * fs)
    if win % 2 == 0:
        win += 1
    win = max(3, min(win, len(x) // 2 * 2 - 1))
    if win < 3:
        return x - np.median(x)
    baseline = medfilt(x, win)
    return x - baseline


def bandpass_fecg_1d(
    sig: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float64).ravel()
    nyq = 0.5 * fs
    lo = max(float(low_hz), 0.5)
    hi = min(float(high_hz), nyq - 1.0)
    if hi <= lo:
        return x
    ord_n = max(2, min(int(order), 6))
    b, a = butter(ord_n, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def prepare_fecg_for_fhr_peaks(
    fetal_ecg: np.ndarray,
    fs: float,
    *,
    detrend_median_sec: float = 0.0,
    fhr_bandpass_hz: Optional[Tuple[float, float]] = None,
    fhr_bandpass_order: int = 4,
) -> np.ndarray:
    """
    Conditioning applied only for FHR peak detection (stored ``fetal_ecg`` unchanged).
    """
    sig = np.asarray(fetal_ecg, dtype=np.float64).ravel()
    if detrend_median_sec > 0:
        sig = rolling_median_detrend_1d(sig, fs, detrend_median_sec)
    if fhr_bandpass_hz is not None:
        lo, hi = float(fhr_bandpass_hz[0]), float(fhr_bandpass_hz[1])
        sig = bandpass_fecg_1d(sig, fs, lo, hi, order=fhr_bandpass_order)
    return sig


def compute_fhr_5hz_causal(
    fetal_ecg: np.ndarray,
    fs: float,
    output_hz: float = 5.0,
    smooth_sec: float = 2.5,
    hr_min_bpm: float = 95.0,
    hr_max_bpm: float = 210.0,
    *,
    inst_outlier_max_bpm: float = 18.0,
    postmedian_halfwin: int = 1,
    peak_height_factor: float = 1.05,
    peak_prominence_factor: float = 0.11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each output time t_k = k / output_hz, mean instantaneous fetal BPM
    from RR intervals whose closing-beat time falls in (t_k - smooth_sec, t_k].

    ``inst_outlier_max_bpm``: within each causal window, drop instantaneous BPM values farther than
    this from the window median before averaging (reduces fHR steps from spurious double peaks).
    ``postmedian_halfwin``: optional short median along the 5 Hz grid (0 disables).
    """
    sig = np.asarray(fetal_ecg, dtype=np.float64).ravel()
    duration = len(sig) / fs
    n_out = int(np.floor(duration * output_hz))
    if n_out < 1:
        return np.array([]), np.array([]), np.array([])
    times = np.arange(n_out) / output_hz
    fhr = np.full(n_out, np.nan, dtype=np.float64)
    valid = np.zeros(n_out, dtype=bool)

    peaks = detect_ecg_peaks(
        sig,
        fs,
        hr_min_bpm,
        hr_max_bpm,
        height_factor=peak_height_factor,
        prominence_factor=peak_prominence_factor,
    )
    t_inst, bpm_inst = instantaneous_hr_from_peaks(peaks, fs)
    if len(t_inst) < 1:
        return times, fhr, valid

    bpm_inst = np.clip(bpm_inst, hr_min_bpm, hr_max_bpm)
    for k in range(n_out):
        t_k = times[k]
        mask = (t_inst > t_k - smooth_sec) & (t_inst <= t_k)
        if np.any(mask):
            fhr[k] = _fhr_inst_window_robust_mean(bpm_inst[mask], inst_outlier_max_bpm)
            valid[k] = np.isfinite(fhr[k])
    if postmedian_halfwin > 0:
        fhr, valid = _fhr_5hz_neighbor_median_smooth(fhr, valid, postmedian_halfwin)
    return times, fhr, valid


def compute_fhr_5hz_single2_style(
    fetal_ecg: np.ndarray,
    fs: float,
    output_hz: float = 5.0,
    hr_min_bpm: float = 100.0,
    hr_max_bpm: float = 170.0,
    medfilt_kernel: int = 9,
    *,
    detrend_median_sec: float = 0.0,
    fhr_bandpass_hz: Optional[Tuple[float, float]] = None,
    fhr_bandpass_order: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peak-pick like ``single2.py`` (positive peaks + medfilt), resampled to ``output_hz``."""
    times, fhr, valid, _, _ = compute_fhr_5hz_single2_style_detailed(
        fetal_ecg,
        fs,
        output_hz=output_hz,
        hr_min_bpm=hr_min_bpm,
        hr_max_bpm=hr_max_bpm,
        medfilt_kernel=medfilt_kernel,
        detrend_median_sec=detrend_median_sec,
        fhr_bandpass_hz=fhr_bandpass_hz,
        fhr_bandpass_order=fhr_bandpass_order,
    )
    return times, fhr, valid


def compute_fhr_5hz_single2_style_detailed(
    fetal_ecg: np.ndarray,
    fs: float,
    output_hz: float = 5.0,
    hr_min_bpm: float = 100.0,
    hr_max_bpm: float = 170.0,
    medfilt_kernel: int = 9,
    fill_5hz_grid: bool = True,
    *,
    detrend_median_sec: float = 0.0,
    fhr_bandpass_hz: Optional[Tuple[float, float]] = None,
    fhr_bandpass_order: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like ``compute_fhr_5hz_single2_style`` but also returns peak-centered times and BPM
    **after** short medfilt on the peak series, **before** filling the 5 Hz grid.
    """
    sig = prepare_fecg_for_fhr_peaks(
        fetal_ecg,
        fs,
        detrend_median_sec=detrend_median_sec,
        fhr_bandpass_hz=fhr_bandpass_hz,
        fhr_bandpass_order=fhr_bandpass_order,
    )
    duration = len(sig) / fs
    n_out = int(np.floor(duration * output_hz))
    if n_out < 1:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    times = np.arange(n_out) / output_hz

    peaks = detect_ecg_peaks(
        sig,
        fs,
        float(hr_min_bpm),
        float(hr_max_bpm),
        height_factor=1.05,
        prominence_factor=0.11,
    )
    if len(peaks) < 5:
        return times, np.full(n_out, np.nan), np.zeros(n_out, dtype=bool), np.array([]), np.array([])

    intervals = np.diff(peaks) / fs
    raw_hr = 60.0 / intervals
    hr_times = (peaks[:-1].astype(np.float64) + peaks[1:].astype(np.float64)) / (2.0 * fs)

    ok = (
        (raw_hr >= float(hr_min_bpm))
        & (raw_hr <= float(hr_max_bpm))
        & np.isfinite(raw_hr)
    )
    if not np.any(ok):
        return times, np.full(n_out, np.nan), np.zeros(n_out, dtype=bool), np.array([]), np.array([])

    hr_times = hr_times[ok]
    f_hr = raw_hr[ok]
    valid_pts = np.isfinite(f_hr)
    if np.sum(valid_pts) >= 2:
        f_hr[~valid_pts] = np.interp(hr_times[~valid_pts], hr_times[valid_pts], f_hr[valid_pts])
    k = int(medfilt_kernel)
    if k % 2 == 0:
        k += 1
    f_hr_med = medfilt(f_hr, k)
    hr_times_out = hr_times.copy()
    hr_bpm_out = f_hr_med.copy()

    if fill_5hz_grid:
        fill = float(np.nanmean(f_hr_med)) if np.any(np.isfinite(f_hr_med)) else 130.0
        f_interp = interp1d(hr_times, f_hr_med, bounds_error=False, fill_value=fill)
        fhr = f_interp(times).astype(np.float64)
        valid = np.isfinite(fhr)
    else:
        fhr = np.full(n_out, np.nan, dtype=np.float64)
        valid = np.zeros(n_out, dtype=bool)
        for t, b in zip(hr_times, f_hr_med):
            if not np.isfinite(t) or not np.isfinite(b):
                continue
            idx = int(round(float(t) * output_hz))
            if 0 <= idx < n_out:
                fhr[idx] = float(b)
                valid[idx] = True
    return times, fhr, valid, hr_times_out, hr_bpm_out


def _compute_segment_fhr(
    fetal_ecg: np.ndarray,
    fs: float,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hz = float(config.fhr_output_hz)
    if config.use_hybrid_v16_fetal and not config.fhr_from_fused_fecg:
        return compute_fhr_5hz_single2_style(
            fetal_ecg,
            fs,
            output_hz=hz,
            medfilt_kernel=9,
            **fhr_peak_pick_kwargs_from_config(config),
        )
    _m, (f_lo, f_hi) = resolve_peak_bpm_windows(config)
    return compute_fhr_5hz_causal(
        fetal_ecg,
        fs,
        output_hz=hz,
        smooth_sec=config.fhr_smooth_sec,
        hr_min_bpm=f_lo,
        hr_max_bpm=f_hi,
        inst_outlier_max_bpm=config.fhr_inst_outlier_max_bpm,
        postmedian_halfwin=int(config.fhr_postmedian_halfwin),
        peak_height_factor=config.fhr_peak_height_factor,
        peak_prominence_factor=config.fhr_peak_prominence_factor,
    )


def compute_mhr_5hz_causal(
    maternal_ecg: np.ndarray,
    fs: float,
    output_hz: float = 5.0,
    smooth_sec: float = 2.5,
    hr_min_bpm: float = 45.0,
    hr_max_bpm: float = 125.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same causal windowing as ``compute_fhr_5hz_causal``, on separated maternal ECG.
    Peak picking uses slightly stricter thresholds than fetal (maternal QRS morphology).
    """
    sig = np.asarray(maternal_ecg, dtype=np.float64).ravel()
    duration = len(sig) / fs
    n_out = int(np.floor(duration * output_hz))
    if n_out < 1:
        return np.array([]), np.array([]), np.array([])
    times = np.arange(n_out) / output_hz
    mhr = np.full(n_out, np.nan, dtype=np.float64)
    valid = np.zeros(n_out, dtype=bool)

    peaks = detect_ecg_peaks(
        sig, fs, hr_min_bpm, hr_max_bpm, height_factor=1.12, prominence_factor=0.11
    )
    t_inst, bpm_inst = instantaneous_hr_from_peaks(peaks, fs)
    if len(t_inst) < 1:
        return times, mhr, valid

    bpm_inst = np.clip(bpm_inst, hr_min_bpm, hr_max_bpm)
    for k in range(n_out):
        t_k = times[k]
        mask = (t_inst > t_k - smooth_sec) & (t_inst <= t_k)
        if np.any(mask):
            mhr[k] = float(np.mean(bpm_inst[mask]))
            valid[k] = True
    return times, mhr, valid


def find_best_sync_offset_5hz(
    my_fhr: np.ndarray,
    gold_fhr: np.ndarray,
    max_scan_sec: int = 120,
    min_overlap_samples: Optional[int] = None,
    output_hz: float = 5.0,
) -> Tuple[int, float]:
    """Best integer sample shift at ``output_hz``; overlap >= ~30s of samples."""
    if min_overlap_samples is None:
        min_overlap_samples = int(30.0 * output_hz)
    best_mae = float("inf")
    best_off = 0
    max_shift = int(max_scan_sec * output_hz)
    for off in range(-max_shift, max_shift + 1):
        if off >= 0:
            g_seg = gold_fhr[off:]
            m_seg = my_fhr[: len(g_seg)]
        else:
            m_seg = my_fhr[-off:]
            g_seg = gold_fhr[: len(m_seg)]
        length = min(len(m_seg), len(g_seg))
        if length < min_overlap_samples:
            continue
        mae = float(np.nanmean(np.abs(m_seg[:length] - g_seg[:length])))
        if mae < best_mae:
            best_mae = mae
            best_off = off
    return best_off, best_mae


def align_series_by_offset_5hz(
    my_fhr: np.ndarray,
    gold_fhr: np.ndarray,
    offset_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if offset_samples >= 0:
        final_g = gold_fhr[offset_samples:]
        final_m = my_fhr[: len(gold_fhr) - offset_samples]
    else:
        final_m = my_fhr[-offset_samples:]
        final_g = gold_fhr[: len(my_fhr) + offset_samples]
    n = min(len(final_m), len(final_g))
    return final_m[:n], final_g[:n]


def build_bss_kwargs(config: PipelineConfig) -> dict:
    d: Dict[str, object] = {
        "window_sec": config.window_sec,
        "overlap_ratio": config.overlap,
        "use_template_cancel": config.template_cancel,
        "maternal_spatial_weight": config.blend_maternal,
        "blend_spatial_fetal": config.blend_fetal,
        "uterine_envelope_weight": config.uterine_envelope_weight,
        "maternal_ecg_band": config.maternal_ecg_band_hz,
        "fetal_band": config.fetal_band_hz,
        "fetal_post_band": config.fetal_post_band_hz,
        "maternal_penalize_fetal_proxy": config.maternal_penalize_fetal_proxy,
        "maternal_ica_fetal_corr_block_thr": config.maternal_ica_fetal_corr_block_thr,
        "fetal_penalize_maternal_proxy": config.fetal_penalize_maternal_proxy,
        "fetal_orthogonalize_maternal": config.fetal_orthogonalize_maternal,
        "fetal_orthogonalize_beta_max": config.fetal_orthogonalize_beta_max,
        "preprocess_notch_50": config.preprocess_notch_50,
        "preprocess_notch_100": config.preprocess_notch_100,
        "preprocess_baseline_highpass_hz": config.preprocess_baseline_highpass_hz,
        "preprocess_baseline_hp_order": config.preprocess_baseline_hp_order,
        "preprocess_per_channel_scale": config.preprocess_per_channel_scale,
        "ica_fetal_proxy_margin": config.ica_fetal_proxy_margin,
        "separation_mode": str(config.separation_mode).strip().lower(),
        "physics_first_maternal_wmix_floor": float(config.physics_first_maternal_wmix_floor),
        "vref_weight_ch0": float(config.vref_weight_ch0),
        "vref_weight_ch5": float(config.vref_weight_ch5),
        "use_pca_maternal_fetal_stack": str(config.separation_mode).strip().lower()
        not in ("ica_dual", "ica_split"),
        "ica_obs_append_v1v6_bipoles": bool(config.ica_obs_append_v1v6_bipoles),
        "ica_split_maternal_v1v6": bool(config.ica_split_maternal_v1v6),
        "ica_split_fetal_v1v6": bool(config.ica_split_fetal_v1v6),
        "ica_obs_append_adaptive_hedge": bool(config.ica_obs_append_adaptive_hedge),
        "ica_obs_append_middle_bipoles": bool(config.ica_obs_append_middle_bipoles),
        "hybrid_fetal_ica_gating": bool(config.hybrid_fetal_ica_gating),
        "hybrid_single2_gate_margin": float(config.hybrid_single2_gate_margin),
        "hybrid_fetal_score_fusion": bool(config.hybrid_fetal_score_fusion),
        "hybrid_fetal_blend_score_margin": float(config.hybrid_fetal_blend_score_margin),
        "hybrid_maternal_score_fusion": bool(config.hybrid_maternal_score_fusion),
        "hybrid_maternal_blend_score_margin": float(config.hybrid_maternal_blend_score_margin),
        "hybrid_fetal_hedge_ica_obs": bool(config.hybrid_fetal_hedge_ica_obs),
        "single2_polarity_v2": bool(config.single2_polarity_v2),
        "single2_band_hz": tuple(float(x) for x in config.single2_band_hz),
        "single2_bandpass_order": int(config.single2_bandpass_order),
        "refine_fhr_window_sec": float(config.bad_refine_fhr_window_sec),
        "refine_mhr_penalize_fetal_boost": float(config.bad_refine_mhr_penalize_fetal_boost),
        "refine_neighbor_bridge_sec": float(config.bad_refine_neighbor_bridge_sec),
    }
    if config.bss_inner_hop_sec is not None:
        d["hop_sec"] = float(config.bss_inner_hop_sec)
    if config.use_v1v6_single_ica:
        d["separation_mode"] = "v1v6_single_ica"
        d["use_pca_maternal_fetal_stack"] = True
        d["ica_obs_fetal_band"] = tuple(float(x) for x in config.single2_band_hz)
    if config.hybrid_fetal_hedge_ica_obs:
        obs_band = tuple(float(x) for x in config.ica_fetal_obs_band_hz)
        d["separation_mode"] = "ica_split"
        d["use_pca_maternal_fetal_stack"] = False
        d["ica_obs_append_adaptive_hedge"] = True
        d["hybrid_fetal_ica_gating"] = True
        d["ica_split_fetal_v1v6"] = False
        # ICA observation band (single2-like); scoring / single2 proxy stay at config fetal_band_hz.
        d["ica_obs_fetal_band"] = obs_band
        d["hybrid_ica_gate_min_peaks"] = int(config.hybrid_ica_gate_min_peaks)
        d["hybrid_ica_gate_peak_ratio"] = float(config.hybrid_ica_gate_peak_ratio)
    sep = str(d["separation_mode"]).strip().lower()
    preset_m = (1.35, 1.05, 0.55, 0.55, 1.05, 1.35)
    preset_f = (1.32, 1.0, 0.58, 0.58, 1.0, 1.32)
    if sep == "ica_split":
        d["ica_split_fetal_v1v6"] = bool(config.ica_split_fetal_v1v6)
        if config.ica_obs_maternal_channel_weights is None:
            d["ica_obs_maternal_channel_weights"] = preset_m
        if config.ica_obs_fetal_channel_weights is None:
            d["ica_obs_fetal_channel_weights"] = preset_f
    if config.use_ica_obs_spatial_guided_weights:
        mw = config.ica_obs_maternal_channel_weights or preset_m
        fw = config.ica_obs_fetal_channel_weights or preset_f
        d["ica_obs_maternal_channel_weights"] = tuple(float(x) for x in mw)
        d["ica_obs_fetal_channel_weights"] = tuple(float(x) for x in fw)
    else:
        if config.ica_obs_maternal_channel_weights is not None:
            d["ica_obs_maternal_channel_weights"] = tuple(float(x) for x in config.ica_obs_maternal_channel_weights)
        if config.ica_obs_fetal_channel_weights is not None:
            d["ica_obs_fetal_channel_weights"] = tuple(float(x) for x in config.ica_obs_fetal_channel_weights)
    return d


def build_refine_bss_kwargs(config: PipelineConfig) -> dict:
    """Same as ``build_bss_kwargs`` plus keys only used by bad-segment multiroute refine."""
    d = build_bss_kwargs(config)
    d["refine_penalize_fetal_maternal_corr"] = config.refine_penalize_fetal_maternal_corr
    if config.use_v1v6_single_ica:
        d["v1v6_single_ica_refine"] = True
        d["chunk_sec"] = float(config.chunk_sec)
        d["fetal_routes"] = tuple(config.fetal_chunk_routes)
    elif config.use_hybrid_v16_fetal:
        d["hybrid_v16_refine"] = True
        d["chunk_sec"] = float(config.chunk_sec)
        d["fetal_routes"] = tuple(config.fetal_chunk_routes)
    return d


# ---------------------------------------------------------------------------
# Six-channel energy envelope + spike removal
# ---------------------------------------------------------------------------

def _bandpass_1d(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.999)
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, x)


def hampel_filter_1d(x: np.ndarray, window: int, n_sigmas: float = 4.0) -> np.ndarray:
    """Replace outliers with local median (per sample)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3 or window < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    y = x.copy()
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) * 1.4826 + 1e-12
        if np.abs(x[i] - med) > n_sigmas * mad:
            y[i] = med
    return y


def compute_six_channel_energy_envelope(
    x6: np.ndarray,
    fs: float,
    band: Tuple[float, float] = (0.08, 10.0),
    fast_smooth_sec: float = 0.35,
    slow_smooth_sec: float = 2.0,
    hampel_window_sec: float = 2.0,
    hampel_n_sigmas: float = 4.0,
    bss_preprocess: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """
    Per-channel Hilbert energy envelope with spike suppression.

    Returns ndarray shape (n_samples, 6).
    ``bss_preprocess`` should be the ``preprocess_*`` subset of ``bss_kwargs`` so envelopes match BSS.
    """
    x0 = preprocess_common_for_pipeline(x6, fs, bss_preprocess)
    n = x0.shape[0]
    env = np.zeros((n, 6), dtype=np.float64)
    w_fast = max(3, int(fast_smooth_sec * fs))
    w_slow = max(w_fast + 1, int(slow_smooth_sec * fs))
    w_hampel = max(5, int(hampel_window_sec * fs))
    if w_hampel % 2 == 0:
        w_hampel += 1

    for ch in range(6):
        sig = _bandpass_1d(x0[:, ch], fs, band[0], band[1])
        e = np.abs(hilbert(sig))
        e = uniform_filter1d(e, size=w_fast, mode="nearest")
        e = hampel_filter_1d(e, w_hampel, n_sigmas=hampel_n_sigmas)
        # Winsorize extreme tails after Hampel
        p_lo, p_hi = np.percentile(e, [1.0, 99.0])
        e = np.clip(e, p_lo, p_hi)
        e = uniform_filter1d(e, size=min(w_slow, max(3, n // 3)), mode="nearest")
        env[:, ch] = e
    return env


def downsample_mean(x: np.ndarray, fs: float, target_fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Block-average downsample to target_fs; return t_sec, values."""
    if target_fs <= 0 or fs <= target_fs:
        t = np.arange(len(x)) / fs
        return t, x
    factor = int(round(fs / target_fs))
    factor = max(1, factor)
    n_blocks = len(x) // factor
    if n_blocks < 1:
        return np.arange(len(x)) / fs, x
    trimmed = x[: n_blocks * factor]
    blocks = trimmed.reshape(n_blocks, factor)
    y = np.mean(blocks, axis=1)
    t = (np.arange(n_blocks) + 0.5) * factor / fs
    return t, y


def energy_envelope_to_compare_grid(
    env6: np.ndarray,
    fs: float,
    target_fs: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample 6-channel envelope; return t_sec (n,) and env (n, 6)."""
    n = env6.shape[0]
    factor = max(1, int(round(fs / target_fs)))
    n_blocks = n // factor
    if n_blocks < 1:
        t = np.arange(n) / fs
        return t, env6
    trimmed = env6[: n_blocks * factor, :]
    blocks = trimmed.reshape(n_blocks, factor, 6)
    y = np.mean(blocks, axis=1)
    t = (np.arange(n_blocks) + 0.5) * factor / fs
    return t, y


# ---------------------------------------------------------------------------
# Four-panel report figure
# ---------------------------------------------------------------------------

def plot_segment_report(
    out: dict,
    env6: np.ndarray,
    fs: float,
    gold_t: np.ndarray,
    gold_fhr: np.ndarray,
    gold_uc: np.ndarray,
    offset_sec: float,
    aligned_my_fhr: np.ndarray,
    aligned_gold_fhr: np.ndarray,
    fhr_mae: float,
    tag: str,
    save_path: str,
    preview_sec: float = 10.0,
    energy_downsample_hz: float = 5.0,
    fhr_dt: float = 0.2,
    aligned_my_mhr: Optional[np.ndarray] = None,
    fused_osc_halfwin: int = 2,
) -> None:
    """Four panels: mECG, fECG, fused energy + oscillation proxy vs UC, aligned FHR @5Hz (+ optional mHR)."""
    m = out["maternal_ecg"]
    f = out["fetal_ecg"]
    n_prev = min(len(m), int(preview_sec * fs))

    t_hr = np.arange(len(aligned_my_fhr), dtype=np.float64) * fhr_dt

    env_t, env_ds = energy_envelope_to_compare_grid(env6, fs, energy_downsample_hz)
    if len(env_t) and len(gold_uc):
        uc_interp = interp1d(
            gold_t,
            gold_uc,
            bounds_error=False,
            fill_value=np.nanmean(gold_uc),
        )
        uc_on_env = uc_interp(env_t + offset_sec)
    else:
        uc_on_env = np.array([])

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    t_prev = np.arange(n_prev) / fs
    axes[0].plot(t_prev, m[:n_prev], color="C0", lw=0.7)
    axes[0].set_ylabel("mECG")
    axes[0].set_title(f"{tag} | Separated mECG ({preview_sec:.0f}s preview)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_prev, f[:n_prev], color="C1", lw=0.7)
    axes[1].set_ylabel("fECG")
    axes[1].set_title(f"Separated fECG ({preview_sec:.0f}s preview)")
    axes[1].grid(True, alpha=0.3)

    fused = fused_six_channel_energy_1d(env_ds)
    osc = fused_energy_oscillation_proxy(fused, halfwin=int(fused_osc_halfwin))
    axes[2].plot(env_t, fused, lw=0.75, color="C2", alpha=0.55, label="6ch fused (median-abs scaled)")
    axes[2].plot(env_t, osc, lw=1.0, color="C1", label="Fused |Δ| proxy (UC oscillation)")
    ax2b = axes[2].twinx()
    if len(env_t) and len(gold_uc):
        ax2b.plot(env_t, uc_on_env, "k--", lw=1.2, alpha=0.85, label="Gold UC")
    axes[2].set_ylabel("Fused / |Δ| proxy (a.u.)")
    ax2b.set_ylabel("Gold UC")
    axes[2].set_title(f"Fused energy + |Δ| proxy vs Gold UC | FHR sync offset={offset_sec:.2f}s")
    axes[2].grid(True, alpha=0.3)
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, ncol=2)

    if len(t_hr):
        axes[3].plot(t_hr, aligned_gold_fhr, "k--", label="Gold FHR", alpha=0.6, lw=1.0)
        axes[3].plot(t_hr, aligned_my_fhr, "r-", label="Algorithm FHR (5Hz)", lw=1.2)
        if (
            aligned_my_mhr is not None
            and len(aligned_my_mhr) == len(t_hr)
            and np.any(np.isfinite(aligned_my_mhr))
        ):
            axes[3].plot(
                t_hr,
                aligned_my_mhr,
                color="C0",
                ls="-",
                lw=1.1,
                alpha=0.9,
                label="Algorithm mHR (5Hz)",
            )
        mae_plot = float(np.nanmean(np.abs(aligned_my_fhr - aligned_gold_fhr)))
        axes[3].set_title(f"Aligned FHR @5Hz | offset={offset_sec:.2f}s | MAE={mae_plot:.2f} bpm")
    else:
        axes[3].set_title("Aligned FHR (insufficient 5Hz samples)")
    axes[3].set_ylabel("HR (bpm)")
    axes[3].set_xlabel("Aligned time (s)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"Segment report: {tag}", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-segment processing
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    subject: str
    segment: str
    tag: str
    offset_sec: float = 0.0
    fhr_mae_bpm: float = float("nan")
    fhr_coverage_pct: float = 0.0
    n_samples: int = 0
    duration_sec: float = 0.0
    success: bool = False
    error: str = ""


def process_one_segment(
    slm_path: str,
    gold_path: str,
    subject: str,
    segment: str,
    output_dir: str,
    config: PipelineConfig,
) -> SegmentResult:
    tag = f"{subject}_{segment}"
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    try:
        x6, fs = load_csv_6ch(slm_path, fs_hint=config.fs_hint)
        fs = float(fs)
        bss_kwargs = build_bss_kwargs(config)
        refine_bss_kwargs = build_refine_bss_kwargs(config)

        def _progress(ci: int, n_chunks: int, start: int, route: str, score: float) -> None:
            if not config.quiet and (ci == 1 or ci == n_chunks or ci % max(1, n_chunks // 10) == 0):
                print(
                    f"    chunk {ci}/{n_chunks} start={start / fs:.1f}s route={route} score={score:.3f}",
                    flush=True,
                )

        if config.use_v1v6_single_ica:
            v1_kw = dict(bss_kwargs)
            v1_kw["window_sec"] = float(config.chunk_sec)
            v1_kw["hop_sec"] = float(config.chunk_sec)
            v1_kw["single2_dual_polarity"] = bool(config.single2_dual_polarity)
            v1_kw["single2_polarity_v2"] = bool(config.single2_polarity_v2)
            v1_kw["chunk_quality_threshold"] = float(config.chunk_quality_threshold)
            v1_kw["fetal_routes"] = tuple(config.fetal_chunk_routes)
            out = v1v6_chunked_bss(
                x6,
                fs,
                chunk_sec=config.chunk_sec,
                bss_kwargs=v1_kw,
                progress_callback=None if config.quiet else _progress,
            )
        elif config.use_hybrid_v16_fetal:
            hybrid_kw = dict(bss_kwargs)
            hybrid_kw["window_sec"] = float(config.chunk_sec)
            hybrid_kw["hop_sec"] = float(config.bss_inner_hop_sec or config.chunk_sec)
            hybrid_kw["single2_dual_polarity"] = bool(config.single2_dual_polarity)
            hybrid_kw["single2_polarity_v2"] = bool(config.single2_polarity_v2)
            hybrid_kw["hybrid_fetal_hedge_ica_obs"] = bool(config.hybrid_fetal_hedge_ica_obs)
            hybrid_kw["hybrid_single2_gate_margin"] = float(config.hybrid_single2_gate_margin)
            out = hybrid_v16_chunked_bss(
                x6,
                fs,
                chunk_sec=config.chunk_sec,
                bss_kwargs=hybrid_kw,
                fetal_routes=tuple(config.fetal_chunk_routes),
                ica_override_margin=float(config.hybrid_ica_override_margin),
                single2_rescue_threshold=float(config.hybrid_single2_rescue_threshold),
                progress_callback=None if config.quiet else _progress,
            )
        elif config.use_chunked:
            out = chunked_multiroute_bss(
                x6,
                fs,
                chunk_sec=config.chunk_sec,
                chunk_hop_sec=config.chunk_hop_sec,
                multiroute=config.multiroute,
                quality_threshold=config.chunk_quality_threshold,
                bss_kwargs=bss_kwargs,
                progress_callback=None if config.quiet else _progress,
            )
        else:
            out = sliding_bss_three_outputs(x6, fs, verbose=not config.quiet, **bss_kwargs)

        if config.maternal_defetal:
            out["maternal_ecg"] = subtract_fetal_band_leakage_from_maternal(
                out["maternal_ecg"],
                out["fetal_ecg"],
                fs,
                fetal_low=config.fetal_post_band_hz[0],
                fetal_high=config.fetal_post_band_hz[1],
                beta_clip=config.maternal_defetal_beta_clip,
            )

        n = len(out["maternal_ecg"])
        _bss_preprocess = {k: v for k, v in bss_kwargs.items() if k.startswith("preprocess_")}
        env6 = compute_six_channel_energy_envelope(
            x6,
            fs,
            band=config.energy_band_hz,
            fast_smooth_sec=config.energy_fast_smooth_sec,
            slow_smooth_sec=config.energy_slow_smooth_sec,
            hampel_window_sec=config.energy_hampel_window_sec,
            hampel_n_sigmas=config.energy_hampel_n_sigmas,
            bss_preprocess=_bss_preprocess,
        )

        gold_t, gold_fhr, gold_uc = load_gold_csv(gold_path, config.gold_fs)
        hz = float(config.fhr_output_hz)
        hr_peak_t = np.array([])
        hr_peak_bpm = np.array([])
        use_single2_style_fhr = (
            (config.use_hybrid_v16_fetal or config.use_v1v6_single_ica)
            and not config.fhr_from_fused_fecg
        )
        if use_single2_style_fhr:
            my_t, my_fhr, valid_algo, hr_peak_t, hr_peak_bpm = compute_fhr_5hz_single2_style_detailed(
                out["fetal_ecg"],
                fs,
                output_hz=hz,
                fill_5hz_grid=bool(config.fhr_fill_5hz_grid),
                **fhr_peak_pick_kwargs_from_config(config),
            )
        else:
            my_t, my_fhr, valid_algo = _compute_segment_fhr(out["fetal_ecg"], fs, config)

        bad_intervals: List[Tuple[float, float]] = []
        n_bad_refined = 0
        fhr_refine_passes = 0
        max_fhr_passes = max(1, int(config.fhr_bad_refine_max_passes))
        if config.use_fhr_jump_bad_refinement and len(my_t) >= 2:
            for _pass in range(max_fhr_passes):
                bad_intervals = detect_fhr_jump_intervals(
                    my_t,
                    my_fhr,
                    jump_threshold_bpm=config.fhr_jump_bpm,
                    pre_sec=config.fhr_bad_pre_sec,
                    post_sec=config.fhr_bad_post_sec,
                )
                if not bad_intervals:
                    break
                n_this = apply_fhr_bad_segment_refinement(
                    out,
                    x6,
                    fs,
                    bad_intervals,
                    refine_bss_kwargs,
                    ramp_sec=config.bad_segment_blend_sec,
                    min_segment_sec=config.bad_segment_min_sec,
                )
                fhr_refine_passes += 1
                n_bad_refined += int(n_this)
                if n_this <= 0:
                    break
                if config.maternal_defetal:
                    out["maternal_ecg"] = subtract_fetal_band_leakage_from_maternal(
                        out["maternal_ecg"],
                        out["fetal_ecg"],
                        fs,
                        fetal_low=config.fetal_post_band_hz[0],
                        fetal_high=config.fetal_post_band_hz[1],
                        beta_clip=config.maternal_defetal_beta_clip,
                    )
                if use_single2_style_fhr:
                    my_t, my_fhr, valid_algo, hr_peak_t, hr_peak_bpm = (
                        compute_fhr_5hz_single2_style_detailed(
                            out["fetal_ecg"],
                            fs,
                            output_hz=hz,
                            fill_5hz_grid=bool(config.fhr_fill_5hz_grid),
                            **fhr_peak_pick_kwargs_from_config(config),
                        )
                    )
                else:
                    my_t, my_fhr, valid_algo = _compute_segment_fhr(out["fetal_ecg"], fs, config)
            bad_intervals = detect_fhr_jump_intervals(
                my_t,
                my_fhr,
                jump_threshold_bpm=config.fhr_jump_bpm,
                pre_sec=config.fhr_bad_pre_sec,
                post_sec=config.fhr_bad_post_sec,
            )

        my_fhr, valid_algo = despike_fhr_adjacent_jumps(
            my_fhr,
            valid_algo,
            gap_bpm=float(config.fhr_output_despike_gap_bpm),
        )
        my_fhr, valid_algo = smooth_fhr_on_output_jumps(
            my_fhr,
            valid_algo,
            jump_bpm=float(config.fhr_output_jump_smooth_bpm),
            med_win=int(config.fhr_output_jump_smooth_win),
        )
        my_fhr, valid_algo = reconcile_fhr_slow_rolling_median(
            my_fhr,
            valid_algo,
            halfwin=int(config.fhr_output_slow_reconcile_halfwin),
            margin_bpm=float(config.fhr_output_slow_reconcile_margin_bpm),
        )
        my_fhr, valid_algo = _fhr_5hz_neighbor_median_smooth(
            my_fhr,
            valid_algo,
            int(config.fhr_output_final_median_halfwin),
        )

        mhr_lo, mhr_hi = resolve_peak_bpm_windows(config)[0]
        _, my_mhr, valid_mhr = compute_mhr_5hz_causal(
            out["maternal_ecg"],
            fs,
            output_hz=hz,
            smooth_sec=config.fhr_smooth_sec,
            hr_min_bpm=mhr_lo,
            hr_max_bpm=mhr_hi,
        )
        fhr_lo, fhr_hi = resolve_peak_bpm_windows(config)[1]
        med_mhr_bpm = median_rr_bpm_from_trace(out["maternal_ecg"], fs, mhr_lo, mhr_hi)
        med_fhr_bpm = median_rr_bpm_from_trace(out["fetal_ecg"], fs, fhr_lo, fhr_hi)

        gold_interp = interp1d(
            gold_t,
            gold_fhr,
            bounds_error=False,
            fill_value=np.nanmedian(gold_fhr),
        )

        if len(my_t) < 2:
            aligned_my = aligned_gold = np.array([])
            aligned_my_mhr = None
            offset_samples = 0
            offset_sec = 0.0
            fhr_mae = float("nan")
            fhr_coverage = 0.0
        else:
            gold_on_grid = gold_interp(my_t).astype(np.float64)
            offset_samples, _ = find_best_sync_offset_5hz(
                my_fhr,
                gold_on_grid,
                max_scan_sec=config.max_sync_scan_sec,
                output_hz=hz,
            )
            offset_sec = float(offset_samples) / hz
            aligned_my, aligned_gold = align_series_by_offset_5hz(my_fhr, gold_on_grid, offset_samples)
            valid = np.isfinite(aligned_my) & np.isfinite(aligned_gold)
            fhr_mae = (
                float(np.nanmean(np.abs(aligned_my[valid] - aligned_gold[valid]))) if valid.any() else float("nan")
            )
            fhr_coverage = 100.0 * valid.sum() / max(len(aligned_my), 1)
            aligned_mhr_full, _ = align_series_by_offset_5hz(my_mhr, gold_on_grid, offset_samples)
            aligned_my_mhr = aligned_mhr_full[: len(aligned_my)].copy()

        prefix = os.path.join(output_dir, segment)
        pd.DataFrame(
            [
                {
                    "mhr_peak_bpm_hz": f"{mhr_lo}-{mhr_hi}",
                    "fhr_peak_bpm_hz": f"{fhr_lo}-{fhr_hi}",
                    "median_inst_bpm_maternal_trace": med_mhr_bpm,
                    "median_inst_bpm_fetal_trace": med_fhr_bpm,
                    "bpm_separation_ok": bool(
                        np.isfinite(med_mhr_bpm)
                        and np.isfinite(med_fhr_bpm)
                        and med_fhr_bpm >= fhr_lo
                        and med_mhr_bpm <= mhr_hi
                        and (med_fhr_bpm - med_mhr_bpm) >= float(config.peak_bpm_guard_gap_bpm) * 0.5
                    ),
                }
            ]
        ).to_csv(f"{prefix}_peak_bpm_diagnostics.csv", index=False)

        if config.use_fhr_jump_bad_refinement:
            pd.DataFrame(
                [{"t_start_s": a, "t_end_s": b} for a, b in bad_intervals],
                columns=["t_start_s", "t_end_s"],
            ).to_csv(f"{prefix}_fhr_bad_segments.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "n_intervals": len(bad_intervals),
                        "n_slices_refined": n_bad_refined,
                        "fhr_refine_passes": fhr_refine_passes,
                        "fhr_refine_max_passes": max_fhr_passes,
                    }
                ],
            ).to_csv(f"{prefix}_fhr_bad_refine_summary.csv", index=False)

        t_sec = np.arange(n) / fs
        pd.DataFrame(
            {
                "time_s": t_sec,
                "maternal_ecg": out["maternal_ecg"],
                "fetal_ecg": out["fetal_ecg"],
                "uterine_abdominal": out["uterine_abdominal"],
            }
        ).to_csv(f"{prefix}_separated.csv", index=False)

        if config.use_hybrid_v16_fetal and config.single2_dual_polarity:
            pol = diagnose_single2_v1v6_polarity(x6, fs, bss_kwargs)
            pol["tag"] = tag
            pol["dual_polarity"] = True
            pd.DataFrame([pol]).to_csv(f"{prefix}_single2_polarity.csv", index=False)

        chunk_meta = out.get("aux", {}).get("chunk_meta")
        if chunk_meta:
            rows_cm = []
            for cm in chunk_meta:
                rs = cm.get("route_scores") or {}
                row = {
                    "start_s": cm.get("start_s"),
                    "end_s": cm.get("end_s"),
                    "rescue_level": cm.get("rescue_level"),
                    "maternal_route": cm.get("maternal_route"),
                    "maternal_score": cm.get("maternal_score"),
                    "fetal_route": cm.get("fetal_route"),
                    "fetal_score": cm.get("fetal_score"),
                }
                for mk, mv in (cm.get("maternal_route_scores") or {}).items():
                    row[f"m_score_{mk}"] = mv
                for rk, rv in rs.items():
                    row[f"score_{rk}"] = rv
                rows_cm.append(row)
            pd.DataFrame(rows_cm).to_csv(f"{prefix}_fetal_chunk_routes.csv", index=False)

        env_t, env_ds = energy_envelope_to_compare_grid(env6, fs, config.energy_downsample_hz)
        env_df = pd.DataFrame({"time_s": env_t})
        for ch in range(6):
            env_df[f"ch{ch + 1}_energy"] = env_ds[:, ch]
        if len(env_t) and env_ds.shape[1] >= 6:
            fu = fused_six_channel_energy_1d(env_ds)
            env_df["fused_energy_median_abs_norm"] = fu
            env_df["fused_oscillation_proxy"] = fused_energy_oscillation_proxy(
                fu, halfwin=int(config.energy_fused_osc_halfwin)
            )
        env_df.to_csv(f"{prefix}_energy_envelope.csv", index=False)

        dt = 1.0 / hz
        if len(my_t):
            pd.DataFrame(
                {
                    "time_s": my_t,
                    "algorithm_fhr_bpm": my_fhr,
                    "gold_fhr_on_slm_grid_bpm": gold_interp(my_t),
                    "valid_algorithm": valid_algo,
                }
            ).to_csv(f"{prefix}_fhr_5hz.csv", index=False)
            if config.export_fhr_peak_series and hr_peak_t.size > 0:
                pd.DataFrame(
                    {
                        "time_s": hr_peak_t,
                        "instantaneous_fhr_bpm": hr_peak_bpm,
                    }
                ).to_csv(f"{prefix}_fhr_peaks_pre_grid.csv", index=False)
            if config.export_fhr_peak_series and len(my_t) >= 1:
                _, fhr_sp, valid_sp, _, _ = compute_fhr_5hz_single2_style_detailed(
                    out["fetal_ecg"],
                    fs,
                    output_hz=hz,
                    fill_5hz_grid=False,
                )
                pd.DataFrame(
                    {
                        "time_s": my_t,
                        "algorithm_fhr_bpm": fhr_sp,
                        "valid_algorithm": valid_sp,
                    }
                ).to_csv(f"{prefix}_fhr_5hz_sparse.csv", index=False)
            pd.DataFrame(
                {
                    "time_s": my_t,
                    "algorithm_mhr_bpm": my_mhr,
                    "valid_algorithm": valid_mhr,
                }
            ).to_csv(f"{prefix}_mhr_5hz.csv", index=False)

        if len(aligned_my):
            cmp_dict: Dict[str, object] = {
                "time_aligned_s": np.arange(len(aligned_my), dtype=np.float64) * dt,
                "algorithm_fhr_bpm": aligned_my,
                "gold_fhr_bpm": aligned_gold,
                "error_bpm": aligned_my - aligned_gold,
                "sync_offset_sec": offset_sec,
            }
            if aligned_my_mhr is not None and len(aligned_my_mhr) == len(aligned_my):
                cmp_dict["algorithm_mhr_bpm"] = aligned_my_mhr
            pd.DataFrame(cmp_dict).to_csv(f"{prefix}_fhr_comparison.csv", index=False)

        fhr_dt = dt
        plot_segment_report(
            out,
            env6,
            fs,
            gold_t,
            gold_fhr,
            gold_uc,
            offset_sec,
            aligned_my,
            aligned_gold,
            fhr_mae,
            tag,
            f"{prefix}_report.png",
            preview_sec=config.plot_preview_sec,
            energy_downsample_hz=config.energy_downsample_hz,
            fhr_dt=fhr_dt,
            aligned_my_mhr=aligned_my_mhr,
            fused_osc_halfwin=int(config.energy_fused_osc_halfwin),
        )

        if not config.quiet:
            print(
                f"  [{tag}] done in {time.time() - t0:.1f}s | offset={offset_sec:.2f}s MAE={fhr_mae:.2f}",
                flush=True,
            )

        return SegmentResult(
            subject=subject,
            segment=segment,
            tag=tag,
            offset_sec=offset_sec,
            fhr_mae_bpm=fhr_mae,
            fhr_coverage_pct=fhr_coverage,
            n_samples=n,
            duration_sec=n / fs,
            success=True,
        )
    except Exception as exc:
        if not config.quiet:
            print(f"  [{tag}] FAILED: {exc}", flush=True)
        return SegmentResult(
            subject=subject,
            segment=segment,
            tag=tag,
            offset_sec=0.0,
            fhr_mae_bpm=float("nan"),
            fhr_coverage_pct=0.0,
            n_samples=0,
            duration_sec=0.0,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def discover_segments(data_root: str, subject_filter: Optional[str] = None) -> List[Tuple[str, str, str, str]]:
    """Return list of (subject, segment, slm_path, gold_path).

    ``subject_filter`` may be a single subject folder name, or several names separated by commas
    (optional spaces), e.g. ``"代淼,夏天茂,王娟"``.
    """
    allowed: Optional[set] = None
    single: Optional[str] = None
    if subject_filter:
        parts = [p.strip() for p in str(subject_filter).split(",") if p.strip()]
        if len(parts) > 1:
            allowed = set(parts)
        elif len(parts) == 1:
            single = parts[0]

    pairs: List[Tuple[str, str, str, str]] = []
    for name in sorted(os.listdir(data_root)):
        sub_path = os.path.join(data_root, name)
        if not os.path.isdir(sub_path):
            continue
        if name.startswith(".") or name in ("Output", "outputs", "docs", "Fetal_Output_Data"):
            continue
        if allowed is not None:
            if name not in allowed:
                continue
        elif single is not None:
            if name != single:
                continue
        for fname in sorted(os.listdir(sub_path)):
            if not fname.endswith("_slm.csv"):
                continue
            seg = fname.replace("_slm.csv", "")
            gold_name = fname.replace("_slm.csv", "_g.csv")
            gold_path = os.path.join(sub_path, gold_name)
            if not os.path.exists(gold_path):
                continue
            pairs.append((name, seg, os.path.join(sub_path, fname), gold_path))
    return pairs


def save_code_snapshot(run_dir: str, project_root: str) -> None:
    """Copy pipeline source files into run_dir/code/."""
    code_dir = os.path.join(run_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    for fname in (
        "base_pipeline.py",
        "run_cohort.py",
        "chunked_multiroute_bss.py",
        "fhr_bad_refinement.py",
        "hybrid_v16_chunk_bss.py",
        "v1v6_chunk_bss.py",
        "abdominal_bss_separate.py",
        "single2.py",
        "single.py",
    ):
        src = os.path.join(project_root, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(code_dir, fname))


def run_batch(
    data_root: str,
    output_root: str = "Output",
    subject_filter: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> str:
    """
    Process all (or one) subject(s). Returns path to run directory.
    """
    config = config or PipelineConfig()
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_id = datetime.now().strftime("RUN-%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "pipeline_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    save_code_snapshot(run_dir, project_root)

    segments = discover_segments(data_root, subject_filter)
    if not segments:
        raise FileNotFoundError(f"No *_slm.csv + *_g.csv pairs under {data_root}")

    print(f"Run {run_id}: {len(segments)} segment(s)", flush=True)
    results: List[SegmentResult] = []

    for subject, seg, slm_path, gold_path in segments:
        sub_out = os.path.join(run_dir, subject)
        print(f"Processing {subject}/{seg} ...", flush=True)
        res = process_one_segment(slm_path, gold_path, subject, seg, sub_out, config)
        results.append(res)

    # Cohort summary
    rows = []
    for r in results:
        rows.append({
            "subject": r.subject,
            "segment": r.segment,
            "tag": r.tag,
            "success": r.success,
            "offset_sec": r.offset_sec,
            "fhr_mae_bpm": r.fhr_mae_bpm,
            "fhr_coverage_pct": r.fhr_coverage_pct,
            "duration_sec": r.duration_sec,
            "error": r.error,
        })
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(run_dir, "cohort_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    subject_mae_path = write_subject_mae_interval_summary(summary_df, run_dir)

    ok = summary_df["success"].sum()
    mae_vals = summary_df.loc[summary_df["success"], "fhr_mae_bpm"].dropna()
    mean_mae = float(mae_vals.mean()) if len(mae_vals) else float("nan")
    print(f"\nDone: {ok}/{len(results)} segments | mean FHR MAE={mean_mae:.2f} bpm", flush=True)
    print(f"Per-subject MAE intervals: {subject_mae_path}", flush=True)
    print(f"Results: {run_dir}", flush=True)
    return run_dir


def write_subject_mae_interval_summary(summary_df: pd.DataFrame, run_dir: str) -> str:
    """
    Aggregate segment-level FHR MAE into per-subject min–max intervals.

    Writes ``subject_mae_intervals.csv`` and prints a compact table.
    """
    ok = summary_df.loc[summary_df["success"] & summary_df["fhr_mae_bpm"].notna()].copy()
    rows: List[dict] = []
    if ok.empty:
        out_path = os.path.join(run_dir, "subject_mae_intervals.csv")
        pd.DataFrame(columns=["subject", "n_segments", "mae_min_bpm", "mae_max_bpm", "mae_mean_bpm", "mae_median_bpm", "mae_range_bpm"]).to_csv(
            out_path, index=False
        )
        return out_path

    for subject, grp in ok.groupby("subject", sort=True):
        mae = grp["fhr_mae_bpm"].astype(float)
        lo = float(mae.min())
        hi = float(mae.max())
        rows.append(
            {
                "subject": str(subject),
                "n_segments": int(len(grp)),
                "mae_min_bpm": lo,
                "mae_max_bpm": hi,
                "mae_mean_bpm": float(mae.mean()),
                "mae_median_bpm": float(mae.median()),
                "mae_std_bpm": float(mae.std(ddof=0)) if len(mae) > 1 else 0.0,
                "mae_range_bpm": f"{lo:.2f}–{hi:.2f}",
            }
        )
    subj_df = pd.DataFrame(rows).sort_values("subject")
    out_path = os.path.join(run_dir, "subject_mae_intervals.csv")
    subj_df.to_csv(out_path, index=False)

    print("\nFHR MAE (bpm) by subject:", flush=True)
    for _, r in subj_df.iterrows():
        print(
            f"  {r['subject']}: {r['mae_range_bpm']} "
            f"(n={int(r['n_segments'])}, mean={r['mae_mean_bpm']:.2f}, median={r['mae_median_bpm']:.2f})",
            flush=True,
        )
    return out_path
