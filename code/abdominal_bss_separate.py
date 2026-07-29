#!/usr/bin/env python3
"""
Six-channel abdominal signal separation: uterine/abdominal activity, maternal ECG, fetal ECG.

Unsupervised pipeline: dual-band preprocessing, sliding-window FastICA or SOBI,
component auto-labeling, optional maternal template cancellation on residuals.
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, find_peaks, hilbert, welch
from scipy.linalg import eigh
from scipy.ndimage import median_filter, uniform_filter1d

try:
    from sklearn.decomposition import FastICA, PCA
    from sklearn.exceptions import ConvergenceWarning
except ImportError as e:  # pragma: no cover
    raise ImportError("scikit-learn is required: pip install scikit-learn") from e


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _parse_sample_rate_from_lines(lines: list[str]) -> Optional[float]:
    for line in lines[:30]:
        if "采样率" in line or "Hz" in line:
            m = re.search(r"(\d+)\s*Hz", line)
            if m:
                return float(m.group(1))
    return None


def load_csv_6ch(
    filepath: str,
    start_sample: int = 0,
    max_samples: Optional[int] = None,
    fs_hint: float = 500.0,
) -> Tuple[np.ndarray, float]:
    """
    Load 6 bipolar channels from data_record-style CSV (comment lines with //, Chinese header).

    Returns
    -------
    data : ndarray, shape (n_samples, 6)
    fs : float, sampling rate (from header if present, else fs_hint)
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    comment_prefixes = ("//", "#")
    header_line_idx = None
    meta_lines: list[str] = []
    for i, line in enumerate(raw_lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith(comment_prefixes):
            meta_lines.append(s)
            continue
        header_line_idx = i
        break

    if header_line_idx is None:
        raise ValueError(f"No data header found in {filepath}")

    fs = _parse_sample_rate_from_lines(meta_lines) or fs_hint

    header = raw_lines[header_line_idx].strip().split(",")
    col_names = [c.strip() for c in header]

    ch_cols: list[int] = []
    patterns = [
        "通道1",
        "通道2",
        "通道3",
        "通道4",
        "通道5",
        "通道6",
    ]
    for p in patterns:
        for j, name in enumerate(col_names):
            if p in name:
                ch_cols.append(j)
                break
        else:
            ch_cols = []
            break

    if len(ch_cols) != 6:
        # Fallback: last 6 columns of CSV
        df_head = pd.read_csv(
            filepath,
            skiprows=header_line_idx,
            nrows=1,
            header=0,
            encoding="utf-8",
        )
        ncols = df_head.shape[1]
        if ncols < 6:
            raise ValueError(f"Expected at least 6 columns, got {ncols}")
        ch_cols = list(range(ncols - 6, ncols))

    df = pd.read_csv(
        filepath,
        skiprows=header_line_idx,
        header=0,
        encoding="utf-8",
    )
    if df.shape[1] <= max(ch_cols):
        raise ValueError(f"Column index out of range: ch_cols={ch_cols}, ncols={df.shape[1]}")

    data = df.iloc[:, ch_cols].to_numpy(dtype=np.float64)
    if np.any(np.isnan(data)):
        data = np.nan_to_num(data, nan=0.0)

    n = data.shape[0]
    start = max(0, int(start_sample))
    end = n if max_samples is None else min(n, start + int(max_samples))
    data = data[start:end, :].copy()

    return data, float(fs)


# ---------------------------------------------------------------------------
# Preprocessing (dual-band)
# ---------------------------------------------------------------------------

def _bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        raise ValueError(f"Invalid band [{low}, {high}] Hz for fs={fs}")
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, x, axis=0)


def subtract_fetal_band_leakage_from_maternal(
    maternal_ecg: np.ndarray,
    fetal_ecg: np.ndarray,
    fs: float,
    fetal_low: float = 15.0,
    fetal_high: float = 45.0,
    beta_clip: float = 2.5,
    order: int = 3,
) -> np.ndarray:
    """
    Optional post-hoc cleanup: subtract coherent fetal-band energy from maternal.

    Prefer fixing separation in ICA (``maternal_penalize_fetal_proxy`` / spatial blend in
    ``sliding_bss_three_outputs``). Use this only for A/B tests — it can remove legitimate
    broadband structure if the linear model is wrong.
    """
    m = np.asarray(maternal_ecg, dtype=np.float64).ravel()
    f = np.asarray(fetal_ecg, dtype=np.float64).ravel()
    n = min(len(m), len(f))
    if n < 32:
        return m.copy()
    m, f = m[:n], f[:n]
    hi = min(float(fetal_high), 0.45 * fs)
    lo = float(max(1.5, fetal_low))
    f_b = _bandpass(f.reshape(-1, 1), fs, lo, hi, order=order).ravel()
    m_b = _bandpass(m.reshape(-1, 1), fs, lo, hi, order=order).ravel()
    f_b = f_b - np.mean(f_b)
    den = float(np.dot(f_b, f_b)) + 1e-12
    num = float(np.dot(m_b, f_b))
    beta = max(0.0, min(float(num / den), float(beta_clip)))
    return m - beta * f_b


def regress_out_maternal_band_from_fetal(
    fetal_ecg: np.ndarray,
    maternal_ecg: np.ndarray,
    fs: float,
    band: Tuple[float, float],
    beta_max: float = 0.38,
    order: int = 3,
) -> np.ndarray:
    """
    Remove a capped linear projection of band-pass maternal from fetal (same band).

    When fetal picks up coherent maternal QRS energy in the fetal analysis band, fHR peak
    times track the mother. This is a mild, band-limited cleanup (|beta| <= ``beta_max``),
    not a full ICA replacement.
    """
    f = np.asarray(fetal_ecg, dtype=np.float64).ravel()
    m = np.asarray(maternal_ecg, dtype=np.float64).ravel()
    n = min(len(f), len(m))
    if n < 64:
        return f.copy()
    f, m = f[:n], m[:n]
    lo = float(band[0])
    hi = min(float(band[1]), 0.45 * fs)
    if hi <= lo + 0.5:
        return f.copy()
    m_b = _bandpass(m.reshape(-1, 1), fs, lo, hi, order=order).ravel()
    m_c = m_b - np.mean(m_b)
    var_m = float(np.dot(m_c, m_c)) + 1e-12
    f_c = f - np.mean(f)
    beta = float(np.dot(f_c, m_c) / var_m)
    bmx = float(abs(beta_max))
    beta = float(np.clip(beta, -bmx, bmx))
    out = f - beta * m_c
    return _zscore(out)


def _notch_50(x: np.ndarray, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [49.0 / nyq, 51.0 / nyq], btype="bandstop")
    return filtfilt(b, a, x, axis=0)


def _notch_100(x: np.ndarray, fs: float, order: int = 3) -> np.ndarray:
    """Second harmonic of mains (50 Hz systems); no-op if Nyquist too low."""
    nyq = fs / 2.0
    if nyq <= 105.0:
        return x
    lo = max(98.0 / nyq, 1e-5)
    hi = min(102.0 / nyq, 0.999)
    if lo >= hi:
        return x
    b, a = butter(order, [lo, hi], btype="bandstop")
    return filtfilt(b, a, x, axis=0)


def _highpass_baseline(x: np.ndarray, fs: float, low_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase high-pass per column to reduce slow wander / low-frequency oscillation."""
    if low_hz <= 0 or x.shape[0] < 32:
        return x
    nyq = fs / 2.0
    lo = max(low_hz / nyq, 1e-5)
    lo = min(lo, 0.99)
    b, a = butter(order, lo, btype="highpass")
    return filtfilt(b, a, x, axis=0)


def preprocess_common(
    x6: np.ndarray,
    fs: float,
    *,
    notch_50: bool = True,
    notch_100_hz: bool = True,
    baseline_highpass_hz: float = 0.35,
    baseline_hp_order: int = 2,
    per_channel_scale: Literal["none", "zscore", "robust"] = "robust",
) -> np.ndarray:
    """
    Six-channel conditioning before dual-band / ICA.

    - Removes per-channel DC (median).
    - Optional zero-phase high-pass (``baseline_highpass_hz``) to attenuate slow baseline / wander;
      set to 0 to disable.
    - Optional 50 Hz and 100 Hz notches (impedance + environment).
    - Optional per-channel amplitude equalisation (``zscore`` or ``robust`` MAD scale) to reduce
      electrode-impedance gain mismatch across channels; ``none`` preserves legacy behaviour.
    """
    y = x6.copy().astype(np.float64)
    y -= np.median(y, axis=0, keepdims=True)
    if baseline_highpass_hz > 0.0:
        min_len = max(64, int(8.0 * fs / max(baseline_highpass_hz, 0.05)))
        if y.shape[0] >= min_len:
            y = _highpass_baseline(y, fs, baseline_highpass_hz, order=baseline_hp_order)
            y -= np.median(y, axis=0, keepdims=True)
    if notch_50:
        y = _notch_50(y, fs)
    if notch_100_hz:
        y = _notch_100(y, fs)
    if per_channel_scale == "none":
        return y
    for ch in range(y.shape[1]):
        col = y[:, ch]
        if per_channel_scale == "zscore":
            mu = float(np.mean(col))
            sig = float(np.std(col)) + 1e-12
            y[:, ch] = (col - mu) / sig
        else:
            med = float(np.median(col))
            mad = float(np.median(np.abs(col - med))) * 1.4826 + 1e-9
            y[:, ch] = (col - med) / mad
    return y


def preprocess_common_for_pipeline(
    x6: np.ndarray,
    fs: float,
    bss_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """
    Apply ``preprocess_common`` using optional keys from ``bss_kwargs`` (from ``build_bss_kwargs``).
    Unknown / missing keys use the same defaults as ``preprocess_common``.
    """
    kw = bss_kwargs or {}
    pcs = str(kw.get("preprocess_per_channel_scale", "robust"))
    if pcs not in ("none", "zscore", "robust"):
        pcs = "robust"
    return preprocess_common(
        x6,
        fs,
        notch_50=bool(kw.get("preprocess_notch_50", True)),
        notch_100_hz=bool(kw.get("preprocess_notch_100", True)),
        baseline_highpass_hz=float(kw.get("preprocess_baseline_highpass_hz", 0.35)),
        baseline_hp_order=int(kw.get("preprocess_baseline_hp_order", 2)),
        per_channel_scale=pcs,  # type: ignore[arg-type]
    )


def spatial_maternal_proxy(x_ecg: np.ndarray) -> np.ndarray:
    """V-shape bipolar mean (similar to 1.py) for robust maternal R-peak detection."""
    c = x_ecg.T
    left = c[0] - c[2]
    right = c[5] - c[3]
    return 0.5 * (left + right)


def robust_maternal_reference(x_m: np.ndarray) -> np.ndarray:
    """
    Combine V-bipoles with upper-abdomen sum (ch1+ch2), then take coordinate-wise median.
    When ch1/ch2 carry the clearest maternal R waves, this tracks them without a single bad
    bipolar dominating (ICA overlay often fails here).
    """
    L = x_m[:, 0] - x_m[:, 2]
    R = x_m[:, 5] - x_m[:, 3]
    top = 0.5 * (x_m[:, 0] + x_m[:, 1])
    mean_lr = 0.5 * (L + R)
    M = np.column_stack([mean_lr, L, R, top])
    return np.median(M, axis=1)


def uterine_contraction_envelope(x0_pre: np.ndarray, fs: float) -> np.ndarray:
    """
    Physiology-first uterine / abdominal activity curve: C3/C4-heavy band (0.08–10 Hz),
    Hilbert envelope, then two-stage smoothing (~0.35 s and ~5 s) to match contraction-scale
    modulation visible in raw lower electrodes.
    """
    y = _bandpass(x0_pre, fs, 0.08, 10.0, order=3)
    bot = 0.62 * y[:, 2] + 0.38 * y[:, 3]
    env = np.abs(hilbert(bot))
    w_fast = max(3, int(0.35 * fs))
    e1 = uniform_filter1d(env, size=w_fast, mode="nearest")
    w_slow = min(max(w_fast + 1, int(5.0 * fs)), max(3, len(e1) // 3))
    e2 = uniform_filter1d(e1, size=w_slow, mode="nearest")
    return detrend(e2, type="linear")


def spatial_fetal_proxy(x_fetal_band: np.ndarray) -> np.ndarray:
    """Same spatial combiner as 1.py fetal path, applied on fetal-band–filtered channels."""
    c = x_fetal_band.T
    left = c[0] - c[2]
    right = c[5] - c[3]
    return 0.5 * (left + right)


def branch_ecg(x6: np.ndarray, fs: float, low: float = 5.0, high: float = 40.0) -> np.ndarray:
    """Mid-frequency branch for ECG family (mECG / fECG candidates)."""
    return _bandpass(x6, fs, low, high)


def branch_uterine(
    x6: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 20.0,
    envelope: bool = True,
    rms_ms: float = 200.0,
) -> np.ndarray:
    """
    Wider low/mid band for abdominal / uterine activity; optional smooth envelope per channel.
    """
    y = _bandpass(x6, fs, low, high)
    if not envelope:
        return y
    env = np.abs(hilbert(y, axis=0))
    win = max(3, int(rms_ms / 1000.0 * fs))
    if win % 2 == 0:
        win += 1
    return uniform_filter1d(env, size=win, axis=0, mode="nearest")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - np.mean(x)
    s = np.std(x)
    return x / (s + 1e-12)


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 8:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.abs(np.dot(a, b) / d))


def _welch_psd_bands_once(
    sig: np.ndarray,
    fs: float,
    fetal_lo: float,
    fetal_hi: float,
    nperseg: int,
) -> Tuple[float, float, float]:
    """
    One Welch estimate per IC for scoring (was two separate Welch calls).

    Returns
    -------
    e_ecg : band 8–40 Hz energy
    e_low : band 0.5–8 Hz energy
    e_fetal : band [fetal_lo, fetal_hi] energy
    """
    n = int(len(sig))
    if n < 32:
        return 0.0, 0.0, 0.0
    nseg = int(min(n, max(64, nperseg)))
    f, pxx = welch(sig, fs=fs, nperseg=nseg, window="hann", nfft=max(nseg * 2, 256))
    pxx = np.maximum(pxx, 1e-20)
    hi_f = min(float(fetal_hi), 0.48 * fs)
    lo_f = float(fetal_lo)
    m_ecg = (f >= 8.0) & (f <= 40.0)
    low = (f >= 0.5) & (f < 8.0)
    fetal = (f >= lo_f) & (f <= hi_f)
    return float(np.sum(pxx[m_ecg])), float(np.sum(pxx[low])), float(np.sum(pxx[fetal]))


def _zscore_window_1d(v: np.ndarray) -> np.ndarray:
    """Per-slice zero-mean unit-variance (1-D)."""
    x = np.asarray(v, dtype=np.float64).ravel()
    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-6)
    return x


def pca_stack_maternal_fetal(
    seg_m: np.ndarray,
    seg_f: np.ndarray,
    n_keep: int = 6,
    random_state: int = 0,
    append_v1v6_bipoles: bool = False,
) -> np.ndarray:
    """
    Stack maternal-band and fetal-band observations -> PCA -> ``n_keep`` components for ICA.

    Default: ``hstack(seg_m, seg_f)`` → shape (win, 12), then PCA.

    When ``append_v1v6_bipoles`` is True, append **per band** two columns derived from
    lateral leads (columns 0 and 5): ``V1+V6`` and ``V1-V6``, each z-scored within the
    window, giving shape (win, 16) before PCA.
    """
    if append_v1v6_bipoles:
        c0m, c5m = seg_m[:, 0], seg_m[:, 5]
        c0f, c5f = seg_f[:, 0], seg_f[:, 5]
        bp_m = np.column_stack(
            (_zscore_window_1d(c0m + c5m), _zscore_window_1d(c0m - c5m))
        )
        bp_f = np.column_stack(
            (_zscore_window_1d(c0f + c5f), _zscore_window_1d(c0f - c5f))
        )
        x_stacked = np.hstack([seg_m, bp_m, seg_f, bp_f])
    else:
        x_stacked = np.hstack([seg_m, seg_f])
    x_stacked = x_stacked - np.mean(x_stacked, axis=0, keepdims=True)
    pca = PCA(n_components=n_keep, whiten=True, random_state=random_state)
    return pca.fit_transform(x_stacked)


def prepare_band_ica_observation(
    seg: np.ndarray,
    channel_weights: Optional[np.ndarray] = None,
    *,
    append_v1v6_bipoles: bool = False,
    append_adaptive_v1v6_hedge: bool = False,
    append_middle_bipoles: bool = False,
    fs: float = 500.0,
    hedge_v1_wide: Optional[np.ndarray] = None,
    hedge_v6_wide: Optional[np.ndarray] = None,
    hedge_bandpass_hz: Tuple[float, float] = (5.0, 40.0),
    seg_wide_6ch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-slice band observation: optional weights, z-score per column.

    Optional extras (before z-score on extras, slice z-scored):
    - ``append_v1v6_bipoles``: simple V1+V6 / V1-V6 on band channels.
    - ``append_adaptive_v1v6_hedge``: K from **wideband** V1/V6 (``hedge_v1_wide`` / ``hedge_v6_wide``),
      then hedge bandpassed at ``hedge_bandpass_hz`` (must match fetal ICA band / single2).
    - ``append_middle_bipoles``: lateral ladder diffs on **wideband** slice columns.
    """
    seg_raw = np.asarray(seg, dtype=np.float64).copy()
    out = seg_raw.copy()
    if channel_weights is not None:
        out *= np.asarray(channel_weights, dtype=np.float64).reshape(1, -1)
    c0_band = seg_raw[:, 0].copy()
    c5_band = seg_raw[:, 5].copy()
    out -= np.mean(out, axis=0)
    out /= np.std(out, axis=0) + 1e-6
    extras: List[np.ndarray] = []
    if append_v1v6_bipoles:
        extras.append(_zscore_window_1d(c0_band + c5_band))
        extras.append(_zscore_window_1d(c0_band - c5_band))
    if append_adaptive_v1v6_hedge:
        from chunked_multiroute_bss import _v1_v6_bandpass_hedge, _v1_v6_hedge_baseline

        if hedge_v1_wide is not None and hedge_v6_wide is not None:
            v1_k = np.asarray(hedge_v1_wide, dtype=np.float64).ravel()
            v6_k = np.asarray(hedge_v6_wide, dtype=np.float64).ravel()
        else:
            v1_k, v6_k = c0_band, c5_band
        v1_c, v6_c, k = _v1_v6_hedge_baseline(v1_k, v6_k, float(fs))
        bp = tuple(float(x) for x in hedge_bandpass_hz)
        extras.append(_zscore_window_1d(_v1_v6_bandpass_hedge(v1_c, v6_c, k, 1.0, float(fs), band_hz=bp)))
        extras.append(_zscore_window_1d(_v1_v6_bandpass_hedge(v1_c, v6_c, k, -1.0, float(fs), band_hz=bp)))
    if append_middle_bipoles and out.shape[1] >= 6:
        wb = np.asarray(seg_wide_6ch if seg_wide_6ch is not None else seg, dtype=np.float64)
        extras.extend(
            [
                _zscore_window_1d(wb[:, 0] - wb[:, 1]),
                _zscore_window_1d(wb[:, 0] - wb[:, 2]),
                _zscore_window_1d(wb[:, 5] - wb[:, 4]),
                _zscore_window_1d(wb[:, 5] - wb[:, 3]),
            ]
        )
    if extras:
        out = np.hstack([out, np.column_stack(extras)])
    return out


def _bandpass_1d(sig: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float64).ravel()
    return _bandpass(x.reshape(-1, 1), fs, low, high, order=order).ravel()


def build_v1v6_ica_observation(
    x0_seg: np.ndarray,
    fs: float,
    *,
    maternal_band: Tuple[float, float],
    fetal_band: Tuple[float, float],
    hedge_band: Tuple[float, float],
    include_fetal_hedge_plus: bool = True,
    include_fetal_hedge_minus: bool = True,
    include_maternal_hedge_minus: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    V1/V6 single-ICA observation stack (no mid-abdomen channels).

    Columns (default): V1_m, V6_m, V1_f, V6_f, hedge_f+, hedge_f-, hedge_m-.
    ``hedge_m-`` = maternal-band filtered ``V1 - K·V6``; fetal hedges use ``hedge_band``.
    """
    from chunked_multiroute_bss import _v1_v6_bandpass_hedge, _v1_v6_hedge_baseline

    x0_seg = np.asarray(x0_seg, dtype=np.float64)
    mb = (float(maternal_band[0]), min(float(maternal_band[1]), 0.48 * fs))
    fb = (float(fetal_band[0]), min(float(fetal_band[1]), 0.48 * fs))
    hb = (float(hedge_band[0]), min(float(hedge_band[1]), 0.48 * fs))

    v1_m = _bandpass_1d(x0_seg[:, 0], fs, mb[0], mb[1])
    v6_m = _bandpass_1d(x0_seg[:, 5], fs, mb[0], mb[1])
    v1_f = _bandpass_1d(x0_seg[:, 0], fs, fb[0], fb[1])
    v6_f = _bandpass_1d(x0_seg[:, 5], fs, fb[0], fb[1])

    cols: List[np.ndarray] = [
        _zscore_window_1d(v1_m),
        _zscore_window_1d(v6_m),
        _zscore_window_1d(v1_f),
        _zscore_window_1d(v6_f),
    ]
    col_names = ["v1_m", "v6_m", "v1_f", "v6_f"]

    v1_c, v6_c, k = _v1_v6_hedge_baseline(x0_seg[:, 0], x0_seg[:, 5], float(fs))
    hedge_plus = hedge_minus = None
    if include_fetal_hedge_plus:
        hedge_plus = _zscore_window_1d(
            _v1_v6_bandpass_hedge(v1_c, v6_c, k, 1.0, float(fs), band_hz=hb)
        )
        cols.append(hedge_plus)
        col_names.append("hedge_f_plus")
    if include_fetal_hedge_minus:
        hedge_minus = _zscore_window_1d(
            _v1_v6_bandpass_hedge(v1_c, v6_c, k, -1.0, float(fs), band_hz=hb)
        )
        cols.append(hedge_minus)
        col_names.append("hedge_f_minus")
    if include_maternal_hedge_minus:
        h_raw = v1_c - float(k) * v6_c
        h_m = _bandpass_1d(h_raw, fs, mb[0], mb[1])
        cols.append(_zscore_window_1d(h_m))
        col_names.append("hedge_m_minus")

    prox_m = _zscore_window_1d(0.5 * (v1_m + v6_m))
    if hedge_plus is not None and hedge_minus is not None:
        c_p = abs(float(np.corrcoef(hedge_plus, hedge_minus)[0, 1]))
        prox_f = hedge_plus if c_p < 0.85 else hedge_plus
    elif hedge_plus is not None:
        prox_f = hedge_plus
    elif hedge_minus is not None:
        prox_f = hedge_minus
    else:
        prox_f = _zscore_window_1d(0.5 * (v1_f + v6_f))

    meta = {
        "k": float(k),
        "col_names": col_names,
        "prox_m": prox_m,
        "prox_f": prox_f,
        "hedge_plus": hedge_plus,
        "hedge_minus": hedge_minus,
    }
    return np.column_stack(cols), meta


def v1v6_fetal_ic_candidates(
    S: np.ndarray,
    sf: np.ndarray,
    im: int,
    iu: int,
    *,
    max_candidates: int = 4,
) -> List[int]:
    """Ranked fetal IC indices (excluding maternal and uterine picks)."""
    n = S.shape[1]
    order = list(np.argsort(-sf))
    out: List[int] = []
    for ji in order:
        j = int(ji)
        if j in (im, iu):
            continue
        if j not in out:
            out.append(j)
        if len(out) >= max_candidates:
            break
    return out


def pca_single_band_stack(
    seg: np.ndarray,
    n_keep: int = 6,
    random_state: int = 0,
) -> np.ndarray:
    """PCA on one band's observation matrix (6 or 8 columns) -> ``n_keep`` components for ICA."""
    x = np.asarray(seg, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    n_in = int(x.shape[1])
    n_samp = int(x.shape[0])
    n_comp = int(min(n_keep, n_in, max(1, n_samp - 1)))
    if n_in <= n_comp:
        return x
    pca = PCA(n_components=n_comp, whiten=True, random_state=random_state)
    return pca.fit_transform(x)


def _zscore_matrix_cols(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z -= np.mean(z, axis=0)
    z /= np.std(z, axis=0) + 1e-6
    return z


def _fastica_on_observation(
    obs: np.ndarray,
    *,
    bss: str,
    rng: np.random.RandomState,
    fastica_max_iter: int,
    fastica_tol: float,
) -> np.ndarray:
    """Run FastICA/SOBI on prepared observation; returns sources (n_samples, n_components)."""
    try:
        if bss == "sobi":
            S, _ = sobi_separate(obs, n_lags=15, lag_step=1)
        else:
            S, _ = fastica_separate(
                obs,
                random_state=int(rng.randint(0, 1_000_000)),
                max_iter=fastica_max_iter,
                tol=fastica_tol,
            )
        return S
    except Exception:
        return obs[:, : min(6, obs.shape[1])].copy()


# ---------------------------------------------------------------------------
# SOBI (joint diagonalization of time-lagged covariances)
# ---------------------------------------------------------------------------
# rjd(): approximate joint diagonalization (JADE-style Jacobi angles).
# Adapted from pyRiemann (BSD-3-Clause): pyriemann/utils/ajd.py
# Cardoso & Souloumiac, SIAM J. Matrix Anal. Appl., 1996.


def rjd(X: np.ndarray, eps: float = 1e-8, n_iter_max: int = 100) -> np.ndarray:
    """
    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Real symmetric matrices to diagonalize jointly.

    Returns
    -------
    V : ndarray, shape (n, n)
        Orthogonal diagonalizer (use sources = z @ V with whitened rows z).
    """
    n_matrices, _, _ = X.shape
    A = np.concatenate(X, axis=0).T
    n, n_matrices_x_n = A.shape
    V = np.eye(n)

    for _ in range(n_iter_max):
        crit = False
        for p in range(n):
            for q in range(p + 1, n):
                Ip = np.arange(p, n_matrices_x_n, n)
                Iq = np.arange(q, n_matrices_x_n, n)
                g = np.array([A[p, Ip] - A[q, Iq], A[p, Iq] + A[q, Ip]])
                gg = g @ g.T
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton**2 + toff**2))
                c = np.cos(theta)
                s = np.sin(theta)
                crit = crit or (np.abs(s) > eps)
                if np.abs(s) > eps:
                    tmp = A[:, Ip].copy()
                    A[:, Ip] = c * A[:, Ip] + s * A[:, Iq]
                    A[:, Iq] = c * A[:, Iq] - s * tmp
                    tmp = A[p, :].copy()
                    A[p, :] = c * A[p, :] + s * A[q, :]
                    A[q, :] = c * A[q, :] - s * tmp
                    tmp = V[:, p].copy()
                    V[:, p] = c * V[:, p] + s * V[:, q]
                    V[:, q] = c * V[:, q] - s * tmp
        if not crit:
            break
    return V


def sobi_separate(
    x: np.ndarray,
    n_lags: int = 20,
    lag_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Second-order blind identification on whitened segments.

    Parameters
    ----------
    x : (n_samples, n_channels), already band-limited / scaled

    Returns
    -------
    S : (n_samples, n_channels) estimated sources (ICA outputs)
    W : (n_channels, n_channels) unmixing so S = X @ W.T (sklearn-like convention)
    """
    n, p = x.shape
    x = x - np.mean(x, axis=0, keepdims=True)
    cov0 = (x.T @ x) / max(n, 1)
    evals, evecs = eigh(cov0)
    emax = np.max(evals) + 1e-12
    evals = np.clip(evals, emax * 1e-6, None)
    whiten = np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    z = x @ whiten.T

    lags = [lag_step * (i + 1) for i in range(n_lags)]
    mats = []
    for tau in lags:
        if tau >= n // 4:
            continue
        z0 = z[:-tau, :]
        z1 = z[tau:, :]
        r = (z0.T @ z1) / z0.shape[0]
        mats.append(0.5 * (r + r.T))
    if not mats:
        mats = [np.eye(p)]
    stack = np.stack(mats, axis=0)
    V = rjd(stack.copy(), eps=1e-8, n_iter_max=100)
    # Whitened rows z; joint diagonalizer V -> source estimate columns in S = z @ V
    S = z @ V
    W = V.T @ whiten
    return S, W


# ---------------------------------------------------------------------------
# FastICA wrapper
# ---------------------------------------------------------------------------

def fastica_separate(
    x: np.ndarray,
    random_state: int = 0,
    max_iter: int = 400,
    tol: float = 2e-3,
    suppress_convergence_warning: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: (n_samples, n_channels)
    Returns S (n_samples, n_channels), W (n_channels, n_channels) with X @ W.T ~ S
    """
    ica = FastICA(
        n_components=x.shape[1],
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    if suppress_convergence_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            S = ica.fit_transform(x)
    else:
        S = ica.fit_transform(x)
    W = ica.components_
    return S, W


def fastica_fit_subsample_transform_full(
    x_full: np.ndarray,
    random_state: int,
    max_fit_rows: int,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Fit FastICA on at most ``max_fit_rows`` samples (evenly spaced), then ``transform``
    the full matrix. Avoids calling ``fit_transform`` on very long traces (which can
    appear to hang for tens of minutes).
    """
    n, p = x_full.shape
    if n <= max_fit_rows:
        S, _ = fastica_separate(x_full, random_state=random_state, max_iter=max_iter, tol=tol)
        return S
    idx = np.linspace(0, n - 1, num=max_fit_rows, dtype=np.int64)
    x_fit = x_full[idx]
    ica = FastICA(
        n_components=p,
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ica.fit(x_fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return ica.transform(x_full)


# ---------------------------------------------------------------------------
# Component scoring (unsupervised)
# ---------------------------------------------------------------------------


def _acf_hr_score(
    sig: np.ndarray,
    fs: float,
    bpm_min: float,
    bpm_max: float,
    max_samples: int = 3000,
) -> float:
    """Peakiness of normalized autocorrelation in plausible HR band (decimated if very long)."""
    s = np.asarray(sig, dtype=np.float64).ravel()
    s = s - np.mean(s)
    std = float(np.std(s)) + 1e-12
    s = s / std
    n = len(s)
    fs_eff = float(fs)
    if n > max(500, max_samples):
        step = int(np.ceil(n / float(max_samples)))
        s = s[::step]
        fs_eff = fs / float(step)
    max_lag = int(fs_eff * 60.0 / bpm_min)
    min_lag = int(fs_eff * 60.0 / bpm_max)
    max_lag = max(max_lag, min_lag + 2)
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2 :]
    ac = ac / (ac[0] + 1e-12)
    if min_lag >= len(ac):
        return 0.0
    region = ac[min_lag : min(max_lag, len(ac))]
    return float(np.max(region) - np.median(region))


def _kurtosis_excess(sig: np.ndarray) -> float:
    z = (sig - np.mean(sig)) / (np.std(sig) + 1e-12)
    return float(np.mean(z**4) - 3.0)


def score_components(
    S: np.ndarray,
    fs: float,
    maternal_bpm: Tuple[float, float] = (45, 125),
    fetal_bpm: Tuple[float, float] = (110, 185),
    fetal_spec_band: Tuple[float, float] = (18.0, 42.0),
    prox_m: Optional[np.ndarray] = None,
    prox_f: Optional[np.ndarray] = None,
    maternal_penalize_fetal_proxy: float = 0.0,
    fetal_penalize_maternal_proxy: float = 0.0,
    *,
    scoring_welch_nperseg: int = 512,
    scoring_acf_max_samples: int = 3000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns scores (n_components,) for maternal-like, fetal-like, uterine-like, and index order.
    Higher is better match. Optional spatial references boost selection when aligned.

    ``maternal_penalize_fetal_proxy`` (default 0): when >0, down-ranks ICA components that correlate
    with the fetal spatial proxy ``prox_f``, reducing fetal QRS picked as the maternal IC.
    ``fetal_penalize_maternal_proxy`` (default 0): penalizes fetal-like ICs that correlate with the
    maternal spatial proxy, reducing maternal QRS dominating the fetal trace (fHR bias).
    """
    n = S.shape[1]
    sm = np.zeros(n)
    sf = np.zeros(n)
    su = np.zeros(n)
    for i in range(n):
        c = S[:, i]
        e_ecg, e_low, e_fetal_hz = _welch_psd_bands_once(
            c, fs, fetal_spec_band[0], fetal_spec_band[1], scoring_welch_nperseg
        )
        km = _acf_hr_score(c, fs, maternal_bpm[0], maternal_bpm[1], scoring_acf_max_samples)
        kf = _acf_hr_score(c, fs, fetal_bpm[0], fetal_bpm[1], scoring_acf_max_samples)
        ku = _kurtosis_excess(c)
        cm = _abs_corr(c, prox_m) if prox_m is not None else 0.0
        cf = _abs_corr(c, prox_f) if prox_f is not None else 0.0
        # Maternal: HR band + QRS spectrum + kurtosis + match V-shape reference − fetal-proxy leakage
        sm[i] = (
            2.0 * km
            + 1.0 * np.log1p(e_ecg)
            + 0.45 * max(ku, 0.0)
            + 2.8 * cm
            - float(maternal_penalize_fetal_proxy) * cf
        )
        # Fetal: faster HR + energy in fetal-frequency band + ref correlation − maternal dominance
        sf[i] = (
            2.1 * kf
            + 1.1 * np.log1p(e_fetal_hz)
            + 0.35 * np.log1p(e_ecg)
            + 0.25 * max(ku, 0.0)
            - 0.55 * km
            + 3.0 * cf
            - float(fetal_penalize_maternal_proxy) * cm
        )
        # Uterine/abdominal: more low-band relative to ECG, less extreme kurtosis
        ratio = e_low / (e_ecg + 1e-12)
        su[i] = (
            np.log1p(ratio)
            + 0.5 * _acf_hr_score(c, fs, 5, 45, scoring_acf_max_samples)
            - 0.25 * max(ku, 0.0)
        )

    return sm, sf, su, np.arange(n)


def pick_three_sources(
    S: np.ndarray,
    sm: np.ndarray,
    sf: np.ndarray,
    su: np.ndarray,
    *,
    prox_m: Optional[np.ndarray] = None,
    prox_f: Optional[np.ndarray] = None,
    ica_fetal_proxy_margin: float = 0.12,
) -> Tuple[int, int, int]:
    """Choose distinct IC indices for maternal, fetal, uterine.

    When ``prox_m`` / ``prox_f`` are given and ``ica_fetal_proxy_margin`` > 0, if the top fetal IC
    correlates much more with the maternal spatial proxy than the fetal proxy (typical maternal-QRS
    leakage), re-select among the next-best fetal candidates one that better matches fetal geometry.
    """
    n = S.shape[1]
    om = np.argsort(-sm)
    of = np.argsort(-sf)
    ou = np.argsort(-su)
    im = int(om[0])
    iff = int(of[0])
    iu = int(ou[0])
    if iff == im:
        iff = int(of[1]) if n > 1 else im
    if iu == im or iu == iff:
        for j in ou:
            if int(j) not in (im, iff):
                iu = int(j)
                break
    if len({im, iff, iu}) < 3:
        combo = sm + sf + su
        picked: list[int] = []
        for j in np.argsort(-combo):
            ji = int(j)
            if ji not in picked:
                picked.append(ji)
            if len(picked) == 3:
                break
        while len(picked) < 3:
            picked.append(picked[-1])
        im, iff, iu = picked[0], picked[1], picked[2]

    if (
        prox_m is not None
        and prox_f is not None
        and S.shape[0] >= 16
        and float(ica_fetal_proxy_margin) > 0.0
    ):
        margin = float(ica_fetal_proxy_margin)
        cm0 = _abs_corr(S[:, iff], prox_m)
        cf0 = _abs_corr(S[:, iff], prox_f)
        if cm0 > cf0 + margin:
            candidates: list[Tuple[float, int]] = []
            for idx in of[: min(6, n)]:
                ji = int(idx)
                if ji in (im, iu):
                    continue
                cm = _abs_corr(S[:, ji], prox_m)
                cf = _abs_corr(S[:, ji], prox_f)
                if cf + 0.5 * margin >= cm:
                    candidates.append((float(sf[ji]), ji))
            if candidates:
                iff = max(candidates, key=lambda t: t[0])[1]

    return im, iff, iu


# ---------------------------------------------------------------------------
# Maternal template cancellation (optional)
# ---------------------------------------------------------------------------

def detect_r_peaks_maternal_proxy(sig: np.ndarray, fs: float) -> np.ndarray:
    s = sig - np.median(sig)
    height = np.std(s) * 1.5
    dist = int(0.35 * fs)
    peaks, _ = find_peaks(np.abs(s), distance=dist, height=height)
    return peaks


def template_subtract_residual(
    x6: np.ndarray,
    fs: float,
    maternal_1d: np.ndarray,
    half_width_ms: float = 60.0,
) -> np.ndarray:
    """
    Build mean QRS template from |maternal_1d| peaks and subtract aligned copies from each channel.
    """
    n, c = x6.shape
    peaks = detect_r_peaks_maternal_proxy(maternal_1d, fs)
    if len(peaks) < 5:
        return x6.copy()
    hw = int(half_width_ms / 1000.0 * fs)
    snippets = []
    for p in peaks:
        if p < hw or p + hw >= n:
            continue
        snippets.append(x6[p - hw : p + hw, :])
    if not snippets:
        return x6.copy()
    tmpl = np.median(np.stack(snippets, axis=0), axis=0)
    residual = x6.copy().astype(np.float64)
    for p in peaks:
        if p < hw or p + hw >= n:
            continue
        residual[p - hw : p + hw, :] -= tmpl
    return residual


# ---------------------------------------------------------------------------
# Sliding-window pipeline + overlap-add
# ---------------------------------------------------------------------------

def _match_sign(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    if np.dot(prev, cur) < 0:
        return -cur
    return cur


def separate_from_csv(
    csv_path: str,
    *,
    start_sample: int = 0,
    max_samples: Optional[int] = None,
    fs_hint: float = 500.0,
    **kwargs,
) -> dict:
    """Load six channels then run `sliding_bss_three_outputs`."""
    x6, fs = load_csv_6ch(csv_path, start_sample=start_sample, max_samples=max_samples, fs_hint=fs_hint)
    return sliding_bss_three_outputs(x6, fs, **kwargs)


def sliding_bss_three_outputs(
    x6: np.ndarray,
    fs: float,
    window_sec: float = 3.0,
    hop_sec: Optional[float] = None,
    overlap_ratio: float = 0.72,
    bss: Literal["fastica", "sobi"] = "fastica",
    use_template_cancel: bool = False,
    random_state: int = 0,
    maternal_ecg_band: Tuple[float, float] = (1.0, 45.0),
    fetal_band: Tuple[float, float] = (17.0, 42.0),
    uterine_band: Tuple[float, float] = (0.5, 20.0),
    fetal_post_band: Tuple[float, float] = (15.0, 45.0),
    blend_spatial_fetal: float = 0.38,
    use_pca_maternal_fetal_stack: bool = True,
    slice_quality_weighting: bool = True,
    ecg_band: Optional[Tuple[float, float]] = None,
    maternal_spatial_weight: float = 0.88,
    uterine_envelope_weight: float = 0.93,
    blend_spatial_maternal: Optional[float] = None,
    verbose: bool = True,
    progress_every: int = 20,
    fastica_max_iter: int = 400,
    fastica_tol: float = 2e-3,
    residual_ica_fit_max_rows: int = 120_000,
    maternal_penalize_fetal_proxy: float = 1.75,
    maternal_ica_fetal_corr_block_thr: float = 0.34,
    fetal_penalize_maternal_proxy: float = 1.35,
    fetal_orthogonalize_maternal: bool = True,
    fetal_orthogonalize_beta_max: float = 0.36,
    ica_fetal_proxy_margin: float = 0.12,
    scoring_welch_nperseg: int = 512,
    scoring_acf_max_samples: int = 3000,
    preprocess_notch_50: bool = True,
    preprocess_notch_100: bool = True,
    preprocess_baseline_highpass_hz: float = 0.35,
    preprocess_baseline_hp_order: int = 2,
    preprocess_per_channel_scale: Literal["none", "zscore", "robust"] = "robust",
    separation_mode: str = "standard",
    physics_first_maternal_wmix_floor: float = 0.94,
    vref_weight_ch0: float = 1.0,
    vref_weight_ch5: float = 1.0,
    ica_obs_maternal_channel_weights: Optional[Sequence[float]] = None,
    ica_obs_fetal_channel_weights: Optional[Sequence[float]] = None,
    ica_obs_append_v1v6_bipoles: bool = False,
    ica_split_maternal_v1v6: bool = True,
    ica_split_fetal_v1v6: bool = False,
    ica_obs_append_adaptive_hedge: bool = False,
    ica_obs_append_middle_bipoles: bool = False,
    ica_obs_fetal_band: Optional[Tuple[float, float]] = None,
) -> dict:
    """
    Full pipeline on (n_samples, 6) raw-ish data.

    - Maternal path uses a **wider** band (default 1–45 Hz) for clear QRS; fetal path uses
      **fetal-focused** band (default 17–42 Hz). ICA input is PCA(12→6) or PCA(16→6) on
      ``[maternal-band | optional V1±V6 bipoles | fetal-band | optional V1±V6 bipoles]`` per slice so the
      optimizer sees frequency-separated observations (skipped when ``separation_mode`` is ``ica_dual``).
    - Per-window scores add correlation with **spatial fetal / maternal proxies** on the same slice.
      Maternal IC scores subtract ``maternal_penalize_fetal_proxy * corr(IC, fetal_proxy)`` so fetal-like
      components are less likely to be chosen as maternal; fetal IC scores subtract
      ``fetal_penalize_maternal_proxy * corr(IC, maternal_proxy)`` to reduce the opposite swap (maternal fQRS
      dominating fECG and biasing fHR).
    - After overlap-add, if the stacked maternal ICA correlates with the fetal spatial proxy in the fetal
      band above ``maternal_ica_fetal_corr_block_thr``, the spatial blend weight ``wmix`` is raised toward
      the robust maternal reference (physics-first).
    - Optional **slice quality** weights down slices with weak maternal or uterine modulation (reduces
      garbage ICA when one modality is absent).
    - **Outputs**: maternal and uterine traces are **dominated by physics-based references** (robust
      spatial + band-pass; C3/C4 contraction envelope). ICA refines fetal more than maternal.
    - ``ecg_band`` kept for backward compatibility: if set, overrides ``maternal_ecg_band``.
    - ``blend_spatial_maternal`` (deprecated): if set, overrides ``maternal_spatial_weight``.
    - ``scoring_welch_nperseg`` / ``scoring_acf_max_samples``: cap per-IC Welch length and ACF length for faster scoring.
    - ``fetal_orthogonalize_maternal``: optional band-limited regression of maternal out of final ``fetal_ecg``
      (capped by ``fetal_orthogonalize_beta_max``) to suppress linear maternal leakage before fHR peak pick.
    - ``preprocess_*``: raw 6ch conditioning (high-pass wander, 50/100 Hz notches, per-channel robust/z-score
      scale for impedance mismatch) before dual-band filtering; set ``preprocess_per_channel_scale="none"``
      and ``preprocess_baseline_highpass_hz=0`` to approximate legacy behaviour.
    - ``separation_mode``: ``ica_dual`` runs ICA on the per-slice maternal-band stack only (disables the
      default 12→6 PCA maternal|fetal stack). ``ica_split`` runs **separate** ICA on maternal-band and
      fetal-band observations (maternal IC from ``seg_m`` only, fetal IC from ``seg_f`` only); maternal
      path typically uses V1±V6 bipoles (``ica_split_maternal_v1v6``) and lateral channel weights.
      ``physics_first`` raises the minimum maternal physics+spatial blend toward
      ``physics_first_maternal_wmix_floor``. ``vref_weight_ch0`` / ``vref_weight_ch5`` are consumed by
      chunked multiroute ``z5_vref`` (unused inside this function).
    - ``ica_obs_maternal_channel_weights`` / ``ica_obs_fetal_channel_weights``: optional length-6 positive
      gains applied **per slice** to ``seg_m`` / ``seg_f`` **before** per-channel z-score and PCA stack,
      so mid-abdomen (e.g. ch3–ch4) can be down-weighted and lateral fetal-band leads (ch1/ch6) up-weighted
      without changing the raw overlap-add reconstruction path.
    - ``ica_obs_append_v1v6_bipoles``: if True, PCA input adds ``V1+V6`` and ``V1-V6`` (cols 0 and 5)
      **within each band**, slice-z-scored → 16-D stack then PCA to six ICA inputs.
    """
    if blend_spatial_maternal is not None:
        maternal_spatial_weight = float(blend_spatial_maternal)

    sep = str(separation_mode).strip().lower()
    split_bands = sep == "ica_split"
    v1v6_single = sep == "v1v6_single_ica"
    if sep in ("ica_dual", "ica_split"):
        effective_use_pca = False
    elif v1v6_single:
        effective_use_pca = True
    else:
        effective_use_pca = bool(use_pca_maternal_fetal_stack)

    x0 = preprocess_common(
        x6,
        fs,
        notch_50=preprocess_notch_50,
        notch_100_hz=preprocess_notch_100,
        baseline_highpass_hz=preprocess_baseline_highpass_hz,
        baseline_hp_order=preprocess_baseline_hp_order,
        per_channel_scale=preprocess_per_channel_scale,
    )
    mb = ecg_band if ecg_band is not None else maternal_ecg_band
    x_m = branch_ecg(x0, fs, low=mb[0], high=mb[1])
    x_f = _bandpass(x0, fs, fetal_band[0], fetal_band[1])
    x_ut = branch_uterine(x0, fs, low=uterine_band[0], high=uterine_band[1], envelope=True)
    x_ecg = x_m

    if use_template_cancel:
        proxy0 = robust_maternal_reference(x_m)
        # Only subtract on maternal-band channels; fetal-band pass-through avoids suppressing fQRS.
        x_m_bss = template_subtract_residual(x_m, fs, proxy0)
        x_f_bss = x_f
    else:
        x_m_bss = x_m
        x_f_bss = x_f

    n = x0.shape[0]
    win = int(window_sec * fs)
    if hop_sec is None:
        hop = max(1, int(win * (1.0 - overlap_ratio)))
    else:
        hop = int(hop_sec * fs)
    if win < 128 or n < win:
        win = min(n, max(256, min(n, int(2.5 * fs))))
        hop = max(win // 4, 1)

    maternal_acc = np.zeros(n)
    fetal_acc = np.zeros(n)
    uterine_acc = np.zeros(n)
    w_m = np.zeros(n)
    w_f = np.zeros(n)
    w_u = np.zeros(n)

    prev_m = prev_f = prev_u = None
    rng = np.random.RandomState(random_state)

    n_win = max(0, (n - win) // hop + 1) if n >= win else 0

    _ica_wm: Optional[np.ndarray] = None
    _ica_wf: Optional[np.ndarray] = None
    if ica_obs_maternal_channel_weights is not None:
        wm = np.asarray(ica_obs_maternal_channel_weights, dtype=np.float64).ravel()
        if wm.size == 6 and np.all(np.isfinite(wm)) and np.all(wm > 0):
            _ica_wm = wm
        elif verbose:
            warnings.warn(
                "ica_obs_maternal_channel_weights ignored: need 6 finite positive values",
                stacklevel=2,
            )
    if ica_obs_fetal_channel_weights is not None:
        wf = np.asarray(ica_obs_fetal_channel_weights, dtype=np.float64).ravel()
        if wf.size == 6 and np.all(np.isfinite(wf)) and np.all(wf > 0):
            _ica_wf = wf
        elif verbose:
            warnings.warn(
                "ica_obs_fetal_channel_weights ignored: need 6 finite positive values",
                stacklevel=2,
            )

    t_loop = time.time()
    if verbose and n_win > 0:
        print(
            f"BSS: {n_win} time slices (window {win / fs:.2f}s, hop {hop / fs:.3f}s). "
            f"This can take several minutes on long recordings.",
            flush=True,
        )

    for wi, start in enumerate(range(0, n - win + 1, hop)):
        end = start + win
        seg_m = x_m_bss[start:end, :].copy()
        if split_bands and ica_obs_fetal_band is not None:
            obs_fb = (
                float(ica_obs_fetal_band[0]),
                min(float(ica_obs_fetal_band[1]), 0.48 * fs),
            )
            seg_f = _bandpass(x0[start:end, :], fs, obs_fb[0], obs_fb[1])
        else:
            seg_f = x_f_bss[start:end, :].copy()
        if _ica_wm is not None:
            seg_m *= _ica_wm.reshape(1, -1)
        if _ica_wf is not None:
            seg_f *= _ica_wf.reshape(1, -1)
        seg_m -= np.mean(seg_m, axis=0)
        seg_m /= np.std(seg_m, axis=0) + 1e-6
        seg_f -= np.mean(seg_f, axis=0)
        seg_f /= np.std(seg_f, axis=0) + 1e-6

        prox_m = spatial_maternal_proxy(seg_m)
        prox_f = spatial_fetal_proxy(seg_f)
        fetal_spec = (fetal_band[0], min(fetal_band[1], 0.48 * fs))

        if v1v6_single:
            hedge_bp = ica_obs_fetal_band if ica_obs_fetal_band is not None else (5.0, 40.0)
            obs_v1, obs_meta = build_v1v6_ica_observation(
                x0[start:end, :],
                fs,
                maternal_band=mb,
                fetal_band=fetal_band,
                hedge_band=hedge_bp,
            )
            prox_m = obs_meta["prox_m"]
            prox_f = obs_meta["prox_f"]
            try:
                z = pca_single_band_stack(
                    obs_v1, n_keep=6, random_state=int(rng.randint(0, 10_000))
                )
                seg_ica = _zscore_matrix_cols(z)
            except Exception:
                seg_ica = _zscore_matrix_cols(obs_v1[:, : min(6, obs_v1.shape[1])])
            S = _fastica_on_observation(
                seg_ica,
                bss=bss,
                rng=rng,
                fastica_max_iter=fastica_max_iter,
                fastica_tol=fastica_tol,
            )
            sm, sf, su, _ = score_components(
                S,
                fs,
                prox_m=prox_m,
                prox_f=prox_f,
                fetal_spec_band=fetal_spec,
                maternal_penalize_fetal_proxy=maternal_penalize_fetal_proxy,
                fetal_penalize_maternal_proxy=fetal_penalize_maternal_proxy,
                scoring_welch_nperseg=scoring_welch_nperseg,
                scoring_acf_max_samples=scoring_acf_max_samples,
            )
            im, iff, iu = pick_three_sources(
                S,
                sm,
                sf,
                su,
                prox_m=prox_m,
                prox_f=prox_f,
                ica_fetal_proxy_margin=ica_fetal_proxy_margin,
            )
            m = S[:, im].copy()
            f = S[:, iff].copy()
            u = S[:, iu].copy()
        elif split_bands:
            # seg_m/seg_f already channel-weighted and z-scored above.
            obs_m = prepare_band_ica_observation(
                seg_m,
                None,
                append_v1v6_bipoles=bool(ica_split_maternal_v1v6),
            )
            obs_f = prepare_band_ica_observation(
                seg_f,
                None,
                append_v1v6_bipoles=bool(ica_split_fetal_v1v6),
                append_adaptive_v1v6_hedge=bool(ica_obs_append_adaptive_hedge),
                append_middle_bipoles=bool(ica_obs_append_middle_bipoles),
                fs=float(fs),
                hedge_v1_wide=x0[start:end, 0],
                hedge_v6_wide=x0[start:end, 5],
                hedge_bandpass_hz=(
                    float(ica_obs_fetal_band[0]) if ica_obs_fetal_band is not None else float(fetal_band[0]),
                    min(
                        float(ica_obs_fetal_band[1]) if ica_obs_fetal_band is not None else float(fetal_band[1]),
                        0.48 * fs,
                    ),
                ),
                seg_wide_6ch=x0[start:end, :],
            )
            try:
                zm = pca_single_band_stack(
                    obs_m, n_keep=6, random_state=int(rng.randint(0, 10_000))
                )
                zm = _zscore_matrix_cols(zm)
            except Exception:
                zm = obs_m[:, : min(6, obs_m.shape[1])]
            try:
                zf = pca_single_band_stack(
                    obs_f, n_keep=6, random_state=int(rng.randint(0, 10_000))
                )
                zf = _zscore_matrix_cols(zf)
            except Exception:
                zf = obs_f[:, : min(6, obs_f.shape[1])]
            Sm = _fastica_on_observation(
                zm, bss=bss, rng=rng, fastica_max_iter=fastica_max_iter, fastica_tol=fastica_tol
            )
            Sf = _fastica_on_observation(
                zf, bss=bss, rng=rng, fastica_max_iter=fastica_max_iter, fastica_tol=fastica_tol
            )
            sm, _, su_m, _ = score_components(
                Sm,
                fs,
                prox_m=prox_m,
                prox_f=prox_f,
                fetal_spec_band=fetal_spec,
                maternal_penalize_fetal_proxy=maternal_penalize_fetal_proxy,
                fetal_penalize_maternal_proxy=0.0,
                scoring_welch_nperseg=scoring_welch_nperseg,
                scoring_acf_max_samples=scoring_acf_max_samples,
            )
            _, sf, _, _ = score_components(
                Sf,
                fs,
                prox_m=prox_m,
                prox_f=prox_f,
                fetal_spec_band=fetal_spec,
                maternal_penalize_fetal_proxy=0.0,
                fetal_penalize_maternal_proxy=fetal_penalize_maternal_proxy,
                scoring_welch_nperseg=scoring_welch_nperseg,
                scoring_acf_max_samples=scoring_acf_max_samples,
            )
            im = int(np.argmax(sm))
            iff = int(np.argmax(sf))
            iu = int(np.argmax(su_m))
            m = Sm[:, im].copy()
            f = Sf[:, iff].copy()
            u = Sm[:, iu].copy()
        else:
            if effective_use_pca:
                try:
                    z = pca_stack_maternal_fetal(
                        seg_m,
                        seg_f,
                        n_keep=6,
                        random_state=int(rng.randint(0, 10_000)),
                        append_v1v6_bipoles=bool(ica_obs_append_v1v6_bipoles),
                    )
                    seg_ica = _zscore_matrix_cols(z)
                except Exception:
                    seg_ica = seg_m
            else:
                seg_ica = seg_m

            S = _fastica_on_observation(
                seg_ica,
                bss=bss,
                rng=rng,
                fastica_max_iter=fastica_max_iter,
                fastica_tol=fastica_tol,
            )

            sm, sf, su, _ = score_components(
                S,
                fs,
                prox_m=prox_m,
                prox_f=prox_f,
                fetal_spec_band=fetal_spec,
                maternal_penalize_fetal_proxy=maternal_penalize_fetal_proxy,
                fetal_penalize_maternal_proxy=fetal_penalize_maternal_proxy,
                scoring_welch_nperseg=scoring_welch_nperseg,
                scoring_acf_max_samples=scoring_acf_max_samples,
            )
            im, iff, iu = pick_three_sources(
                S,
                sm,
                sf,
                su,
                prox_m=prox_m,
                prox_f=prox_f,
                ica_fetal_proxy_margin=ica_fetal_proxy_margin,
            )
            m = S[:, im].copy()
            f = S[:, iff].copy()
            u = S[:, iu].copy()

        if verbose and n_win > 0:
            if (wi + 1) % progress_every == 0 or (wi + 1) == n_win:
                elapsed = time.time() - t_loop
                done = wi + 1
                rate = done / max(elapsed, 1e-6)
                eta = (n_win - done) / max(rate, 1e-6)
                print(
                    f"  slices {done}/{n_win} ({100.0 * done / n_win:.1f}%) "
                    f"elapsed {elapsed:.1f}s  eta ~{eta:.0f}s",
                    flush=True,
                )

        if prev_m is not None:
            m = _match_sign(prev_m, m)
            f = _match_sign(prev_f, f)
            u = _match_sign(prev_u, u)
        prev_m, prev_f, prev_u = m.copy(), f.copy(), u.copy()

        w0 = np.hanning(end - start)
        if slice_quality_weighting:
            qm = np.std(prox_m) / (np.mean(np.abs(seg_m)) + 1e-9)
            qm = float(np.clip(qm / 2.5, 0.2, 1.0))
            ut1 = np.mean(x_ut[start:end, :], axis=1)
            qu = float(np.clip(np.std(ut1) / (np.mean(np.abs(ut1)) + 1e-9) / 4.0, 0.15, 1.0))
            qf = np.std(prox_f) / (np.mean(np.abs(seg_f)) + 1e-9)
            qf = float(np.clip(qf / 2.5, 0.2, 1.0))
            wm = w0 * qm
            wf = w0 * qf
            wu = w0 * qu
        else:
            wm = wf = wu = w0

        maternal_acc[start:end] += m * wm
        fetal_acc[start:end] += f * wf
        uterine_acc[start:end] += u * wu
        w_m[start:end] += wm
        w_f[start:end] += wf
        w_u[start:end] += wu

    w_m[w_m < 1e-9] = 1.0
    w_f[w_f < 1e-9] = 1.0
    w_u[w_u < 1e-9] = 1.0
    maternal_ica = maternal_acc / w_m
    fetal_ica = fetal_acc / w_f
    uterine_ica = uterine_acc / w_u

    m_sp = spatial_maternal_proxy(x_m)
    f_sp = spatial_fetal_proxy(x_f)

    # --- Maternal: physics-first (robust spatial + band-pass); ICA only small correction when consistent
    m_ref = robust_maternal_reference(x_m)
    lo_m = float(max(1.5, mb[0]))
    hi_m = float(min(45.0, mb[1]))
    m_bp = _bandpass(m_ref.reshape(-1, 1), fs, lo_m, hi_m, order=3).ravel()
    m_base = _zscore(m_bp)
    ica_m = _zscore(maternal_ica)
    c_m = np.corrcoef(m_base, ica_m)[0, 1]
    if np.isfinite(c_m) and c_m < 0:
        ica_m = -ica_m
    f_hi = min(float(fetal_band[1]), 0.45 * fs)
    ica_m_fb = _bandpass(ica_m.reshape(-1, 1), fs, float(fetal_band[0]), f_hi, order=3).ravel()
    f_sp_fb = _bandpass(f_sp.reshape(-1, 1), fs, float(fetal_band[0]), f_hi, order=3).ravel()
    try:
        c_mf = float(np.abs(np.corrcoef(ica_m_fb, f_sp_fb)[0, 1]))
    except Exception:
        c_mf = 0.0
    if not np.isfinite(c_mf):
        c_mf = 0.0
    wmix = float(np.clip(maternal_spatial_weight, 0.0, 1.0))
    if not np.isfinite(c_m) or abs(c_m) < 0.28:
        wmix = max(wmix, 0.9)
    if c_mf > float(maternal_ica_fetal_corr_block_thr):
        wmix = max(wmix, 0.94)
    wmix_blend = float(wmix)
    if sep == "physics_first":
        wmix_blend = max(wmix_blend, float(np.clip(physics_first_maternal_wmix_floor, 0.0, 1.0)))
    if split_bands and np.isfinite(c_m) and abs(c_m) >= 0.30:
        # Maternal-band ICA is dedicated to mECG: allow more ICA in the final blend when aligned.
        wmix_blend = min(wmix_blend, 0.76)
    maternal_ecg = _zscore(wmix_blend * m_base + (1.0 - wmix_blend) * ica_m)

    # --- Fetal: keep spatial-heavy blend; tighten when ICA is poorly correlated
    f_bp = _bandpass(
        f_sp.reshape(-1, 1),
        fs,
        fetal_band[0],
        min(fetal_band[1], 0.45 * fs),
        order=3,
    ).ravel()
    f_base = _zscore(f_bp)
    ica_f = _zscore(fetal_ica)
    c_f = np.corrcoef(f_base, ica_f)[0, 1]
    if np.isfinite(c_f) and c_f < 0:
        ica_f = -ica_f
    wf = float(np.clip(blend_spatial_fetal, 0.0, 1.0))
    if not np.isfinite(c_f) or abs(c_f) < 0.2:
        wf = max(wf, 0.55)
    fetal_ecg = _zscore(wf * f_base + (1.0 - wf) * ica_f)
    fetal_ecg = _bandpass(
        fetal_ecg.reshape(-1, 1), fs, fetal_post_band[0], fetal_post_band[1], order=3
    ).ravel()

    # --- Uterine: C3/C4 low-frequency envelope (contraction-scale); ICA uterine is weak here
    u_phys = uterine_contraction_envelope(x0, fs)
    u_ica = _zscore(uterine_ica)
    c_u = np.corrcoef(_zscore(u_phys), u_ica)[0, 1]
    if np.isfinite(c_u) and c_u < 0:
        u_ica = -u_ica
    wu_env = float(np.clip(uterine_envelope_weight, 0.0, 1.0))
    if not np.isfinite(c_u) or abs(c_u) < 0.18:
        wu_env = max(wu_env, 0.92)
    uterine_combined = _zscore(wu_env * _zscore(u_phys) + (1.0 - wu_env) * u_ica)

    if use_template_cancel:
        if verbose:
            print(
                "Second pass: fetal refinement on maternal residual "
                f"(ICA fit <= {residual_ica_fit_max_rows} rows, then transform full length)...",
                flush=True,
            )
        residual = template_subtract_residual(x_m, fs, maternal_ecg)
        res = residual / (np.std(residual, axis=0) + 1e-6)
        try:
            # Full-length FastICA.fit is prohibitive on long records; always subsample-fit here.
            S2 = fastica_fit_subsample_transform_full(
                res,
                random_state=random_state + 1,
                max_fit_rows=residual_ica_fit_max_rows,
                max_iter=fastica_max_iter,
                tol=fastica_tol,
            )
            prox_r = spatial_fetal_proxy(
                _bandpass(res, fs, fetal_band[0], min(fetal_band[1], 0.48 * fs))
            )
            sm2, sf2, _, _ = score_components(
                S2,
                fs,
                prox_f=prox_r,
                fetal_spec_band=(fetal_band[0], fetal_band[1]),
                maternal_penalize_fetal_proxy=maternal_penalize_fetal_proxy,
                scoring_welch_nperseg=scoring_welch_nperseg,
                scoring_acf_max_samples=scoring_acf_max_samples,
            )
            iff2 = int(np.argmax(sf2 - 0.45 * sm2))
            fetal_refined = S2[:, iff2]
            fetal_ecg = 0.55 * fetal_ecg + 0.45 * _zscore(fetal_refined)
            fetal_ecg = _bandpass(
                fetal_ecg.reshape(-1, 1), fs, fetal_post_band[0], fetal_post_band[1]
            ).ravel()
        except Exception:
            pass
        if verbose:
            print("Second pass done.", flush=True)

    if fetal_orthogonalize_maternal:
        lo_fp = float(fetal_post_band[0])
        hi_fp = min(float(fetal_post_band[1]), 0.45 * fs)
        fetal_ecg = regress_out_maternal_band_from_fetal(
            fetal_ecg,
            maternal_ecg,
            fs,
            (lo_fp, hi_fp),
            beta_max=fetal_orthogonalize_beta_max,
        )
        fetal_ecg = _bandpass(
            fetal_ecg.reshape(-1, 1), fs, fetal_post_band[0], fetal_post_band[1], order=3
        ).ravel()

    return {
        "maternal_ecg": maternal_ecg,
        "fetal_ecg": fetal_ecg,
        "uterine_abdominal": uterine_combined,
        "aux": {
            "x_ecg_branch": x_ecg,
            "x_fetal_band_branch": x_f,
            "x_uterine_branch": x_ut,
            "maternal_robust_ref": m_ref,
            "uterine_envelope_physical": u_phys,
            "fs": fs,
        },
    }


# ---------------------------------------------------------------------------
# QC plots
# ---------------------------------------------------------------------------

def plot_qc(
    out: dict,
    fs: float,
    max_seconds: Optional[float] = None,
    save_path: Optional[str] = None,
    max_plot_points: int = 800_000,
) -> None:
    """
    Plot QC over the full record by default. Long signals are decimated along time
    for rendering (``max_plot_points`` cap per series).
    """
    import matplotlib.pyplot as plt

    m = out["maternal_ecg"]
    f = out["fetal_ecg"]
    u = out["uterine_abdominal"]
    n = len(m)
    if max_seconds is not None:
        k = min(n, max(1, int(max_seconds * fs)))
    else:
        k = n

    n_plot = min(k, max(1, int(max_plot_points)))
    idx = np.unique(np.linspace(0, k - 1, num=n_plot, dtype=np.int64))
    t = idx.astype(np.float64) / fs

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(t, m[idx], color="C0", lw=0.35)
    axes[0].set_ylabel("mECG")
    axes[1].plot(t, f[idx], color="C1", lw=0.35)
    axes[1].set_ylabel("fECG")
    axes[2].plot(t, u[idx], color="C2", lw=0.35)
    axes[2].set_ylabel("Uterine / abd.")
    xb = out["aux"]["x_ecg_branch"][idx, :3]
    axes[3].plot(t, xb[:, 0], lw=0.25, alpha=0.75, label="ch1")
    axes[3].plot(t, xb[:, 1], lw=0.25, alpha=0.75, label="ch2")
    axes[3].plot(t, xb[:, 2], lw=0.25, alpha=0.75, label="ch3")
    axes[3].set_ylabel("Raw (maternal band)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", fontsize=7, ncol=3)
    title = f"BSS QC (0–{k / fs:.1f}s of {n / fs:.1f}s"
    if len(idx) < k:
        title += f", decimated to {len(idx)} pts"
    title += ")"
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_outputs_csv(
    out: dict,
    fs: float,
    path_prefix: str,
) -> None:
    n = len(out["maternal_ecg"])
    t = np.arange(n) / fs
    df = pd.DataFrame(
        {
            "time_s": t,
            "maternal_ecg": out["maternal_ecg"],
            "fetal_ecg": out["fetal_ecg"],
            "uterine_abdominal": out["uterine_abdominal"],
        }
    )
    df.to_csv(f"{path_prefix}_three_channels.csv", index=False)


# ---------------------------------------------------------------------------
# Dynamic heart rate (from separated mECG / fECG)
# ---------------------------------------------------------------------------


def detect_ecg_peaks(
    sig: np.ndarray,
    fs: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
    height_factor: float = 1.15,
    prominence_factor: float = 0.12,
) -> np.ndarray:
    """
    R-peak–like detection on a 1-D separated trace using ``|signal|``.
    ``distance`` is set from max HR; optional post-filter on implausible RR.
    """
    s = np.asarray(sig, dtype=np.float64).ravel()
    s = s - np.median(s)
    std = float(np.std(s)) + 1e-12
    min_dist = max(3, int(60.0 / hr_max_bpm * fs))
    peaks, _ = find_peaks(
        np.abs(s),
        distance=min_dist,
        height=std * height_factor,
        prominence=std * prominence_factor,
    )
    if len(peaks) < 2:
        return peaks
    max_rr = 60.0 / hr_min_bpm
    keep = [0]
    for i in range(1, len(peaks)):
        dt = (peaks[i] - peaks[keep[-1]]) / fs
        if dt >= 60.0 / hr_max_bpm * 0.85 and dt <= max_rr * 1.15:
            keep.append(i)
    filtered = peaks[np.array(keep, dtype=np.int64)]
    if len(filtered) < max(8, int(0.22 * len(peaks))):
        return peaks
    return filtered


def instantaneous_hr_from_peaks(peaks: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each completed RR interval, return (time of the later beat in seconds, HR in bpm).

    Returns
    -------
    t_sec : (n_beats - 1,) time aligned to the beat closing the interval (second R peak).
    bpm : instantaneous rate 60/RR.
    """
    if len(peaks) < 2:
        return np.array([]), np.array([])
    rr_sec = np.diff(peaks.astype(np.float64)) / fs
    rr_sec = np.clip(rr_sec, 0.12, 2.5)
    bpm = 60.0 / rr_sec
    t_sec = peaks[1:].astype(np.float64) / fs
    return t_sec, bpm


def compute_hr_from_bss_output(
    out: dict,
    fs: float,
    maternal_hr_range: Tuple[float, float] = (45.0, 185.0),
    fetal_hr_range: Tuple[float, float] = (95.0, 210.0),
    smooth_median: int = 3,
) -> dict:
    """
    Peak-pick on ``maternal_ecg`` / ``fetal_ecg`` and instantaneous HR time series.
    """
    sig_m = out["maternal_ecg"]
    sig_f = out["fetal_ecg"]
    pm = detect_ecg_peaks(
        sig_m,
        fs,
        maternal_hr_range[0],
        maternal_hr_range[1],
        height_factor=1.12,
        prominence_factor=0.11,
    )
    pf = detect_ecg_peaks(
        sig_f,
        fs,
        fetal_hr_range[0],
        fetal_hr_range[1],
        height_factor=0.95,
        prominence_factor=0.09,
    )
    tm, bm = instantaneous_hr_from_peaks(pm, fs)
    tf, bf = instantaneous_hr_from_peaks(pf, fs)
    bm = np.clip(bm, maternal_hr_range[0], maternal_hr_range[1])
    bf = np.clip(bf, fetal_hr_range[0], fetal_hr_range[1])
    if smooth_median >= 3 and len(bm) >= 3:
        k = min(smooth_median, len(bm))
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            bm = median_filter(bm, size=k)
    if smooth_median >= 3 and len(bf) >= 3:
        k = min(smooth_median, len(bf))
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            bf = median_filter(bf, size=k)
    return {
        "maternal_peaks": pm,
        "fetal_peaks": pf,
        "maternal_t_bpm": tm,
        "maternal_bpm": bm,
        "fetal_t_bpm": tf,
        "fetal_bpm": bf,
        "fs": fs,
    }


def save_hr_csv(hr: dict, path_prefix: str) -> None:
    """Write maternal / fetal instantaneous HR and peak lists."""
    pm, pf = hr["maternal_peaks"], hr["fetal_peaks"]
    fs = float(hr["fs"])
    df_m = pd.DataFrame(
        {
            "peak_time_s": pm.astype(np.float64) / fs,
            "peak_sample": pm,
        }
    )
    df_m.to_csv(f"{path_prefix}_maternal_peaks.csv", index=False)
    df_f = pd.DataFrame(
        {
            "peak_time_s": pf.astype(np.float64) / fs,
            "peak_sample": pf,
        }
    )
    df_f.to_csv(f"{path_prefix}_fetal_peaks.csv", index=False)
    if len(hr["maternal_bpm"]) > 0:
        df_mi = pd.DataFrame(
            {
                "time_s": hr["maternal_t_bpm"],
                "instantaneous_bpm": hr["maternal_bpm"],
                "rr_interval_s": 60.0 / hr["maternal_bpm"],
            }
        )
    else:
        df_mi = pd.DataFrame(columns=["time_s", "instantaneous_bpm", "rr_interval_s"])
    df_mi.to_csv(f"{path_prefix}_maternal_hr_instantaneous.csv", index=False)
    if len(hr["fetal_bpm"]) > 0:
        df_fi = pd.DataFrame(
            {
                "time_s": hr["fetal_t_bpm"],
                "instantaneous_bpm": hr["fetal_bpm"],
                "rr_interval_s": 60.0 / hr["fetal_bpm"],
            }
        )
    else:
        df_fi = pd.DataFrame(columns=["time_s", "instantaneous_bpm", "rr_interval_s"])
    df_fi.to_csv(f"{path_prefix}_fetal_hr_instantaneous.csv", index=False)


def plot_hr_dynamics(
    hr: dict,
    save_path: str,
    max_plot_points: int = 120_000,
) -> None:
    """New figure: maternal and fetal instantaneous HR vs time (does not touch BSS QC figure)."""
    import matplotlib.pyplot as plt

    tm, bm = hr["maternal_t_bpm"], hr["maternal_bpm"]
    tf, bf = hr["fetal_t_bpm"], hr["fetal_bpm"]

    def _subsample(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(t)
        if n <= max_plot_points or n == 0:
            return t, y
        idx = np.unique(np.linspace(0, n - 1, num=max_plot_points, dtype=np.int64))
        return t[idx], y[idx]

    tm_p, bm_p = _subsample(tm, bm)
    tf_p, bf_p = _subsample(tf, bf)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    if len(tm_p) == 0:
        axes[0].text(0.5, 0.5, "No maternal RR intervals", ha="center", va="center", transform=axes[0].transAxes)
    else:
        axes[0].plot(tm_p, bm_p, color="C0", lw=0.7, marker=".", markersize=2, alpha=0.85)
    axes[0].set_ylabel("Maternal HR (bpm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"Maternal instantaneous HR (n_peaks={len(hr['maternal_peaks'])}, "
        f"n_intervals={len(bm)})"
    )
    if len(tf_p) == 0:
        axes[1].text(0.5, 0.5, "No fetal RR intervals", ha="center", va="center", transform=axes[1].transAxes)
    else:
        axes[1].plot(tf_p, bf_p, color="C1", lw=0.7, marker=".", markersize=2, alpha=0.85)
    axes[1].set_ylabel("Fetal HR (bpm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f"Fetal instantaneous HR (n_peaks={len(hr['fetal_peaks'])}, n_intervals={len(bf)})")
    fig.suptitle("Dynamic heart rate (from separated BSS outputs)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Six-channel abdominal BSS separation")
    p.add_argument("csv", help="Path to data_record CSV")
    p.add_argument("--out", default="bss_out", help="Output prefix for CSV and PNG")
    p.add_argument("--start", type=int, default=0, help="Start sample index")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples to load")
    p.add_argument("--fs", type=float, default=500.0, help="Fallback sampling rate if not in file")
    p.add_argument("--window", type=float, default=3.0, help="Slice length for BSS (s)")
    p.add_argument("--hop", type=float, default=None, help="Hop in seconds (overrides --overlap if set)")
    p.add_argument(
        "--overlap",
        type=float,
        default=0.72,
        help="Overlap ratio when --hop not set; hop = window*(1-overlap)",
    )
    p.add_argument("--bss", choices=("fastica", "sobi"), default="fastica")
    p.add_argument("--template-cancel", action="store_true", help="Maternal template subtraction + residual ICA")
    p.add_argument(
        "--plot-seconds",
        type=float,
        default=None,
        help="If set, only plot first N seconds; default plots full length (decimated)",
    )
    p.add_argument("--plot-max-points", type=int, default=800_000, help="Max points per trace in QC figure")
    p.add_argument("--ecg-low", type=float, default=1.0, help="Maternal / broad ECG branch low Hz")
    p.add_argument("--ecg-high", type=float, default=45.0, help="Maternal / broad ECG branch high Hz")
    p.add_argument("--fetal-low", type=float, default=17.0, help="Fetal-focused band for ICA stack (Hz)")
    p.add_argument("--fetal-high", type=float, default=42.0, help="Fetal-focused band for ICA stack (Hz)")
    p.add_argument("--fetal-post-low", type=float, default=15.0, help="Output bandpass on fECG (Hz)")
    p.add_argument("--fetal-post-high", type=float, default=45.0, help="Output bandpass on fECG (Hz)")
    p.add_argument("--abd-low", type=float, default=0.5)
    p.add_argument("--abd-high", type=float, default=20.0)
    p.add_argument("--no-pca-stack", action="store_true", help="Disable 12ch (maternal|fetal) PCA before ICA")
    p.add_argument("--no-slice-quality", action="store_true", help="Disable per-slice quality weights")
    p.add_argument(
        "--blend-maternal",
        type=float,
        default=0.88,
        help="Weight on robust spatial+bandpass maternal (1=pure reference path; ICA is residual)",
    )
    p.add_argument(
        "--uterine-envelope-weight",
        type=float,
        default=0.93,
        help="Weight on C3/C4 low-frequency contraction envelope vs ICA uterine component",
    )
    p.add_argument("--blend-fetal", type=float, default=0.42, help="Weight of spatial fetal comb (0-1)")
    p.add_argument("--quiet", action="store_true", help="No progress lines (only final messages)")
    p.add_argument("--progress-every", type=int, default=20, help="Print progress every N slices")
    p.add_argument("--fastica-max-iter", type=int, default=400, help="FastICA max iterations per slice")
    p.add_argument("--fastica-tol", type=float, default=2e-3, help="FastICA convergence tolerance")
    p.add_argument(
        "--residual-ica-fit-rows",
        type=int,
        default=120_000,
        help="Max rows for fitting residual ICA (--template-cancel); avoids hang on long files",
    )
    args = p.parse_args()

    x6, fs = load_csv_6ch(args.csv, start_sample=args.start, max_samples=args.max_samples, fs_hint=args.fs)
    print(f"Loaded shape={x6.shape}, fs={fs}", flush=True)

    t0 = time.time()
    out = sliding_bss_three_outputs(
        x6,
        fs,
        window_sec=args.window,
        hop_sec=args.hop,
        overlap_ratio=args.overlap,
        bss=args.bss,
        use_template_cancel=args.template_cancel,
        maternal_ecg_band=(args.ecg_low, args.ecg_high),
        fetal_band=(args.fetal_low, args.fetal_high),
        uterine_band=(args.abd_low, args.abd_high),
        fetal_post_band=(args.fetal_post_low, args.fetal_post_high),
        use_pca_maternal_fetal_stack=not args.no_pca_stack,
        slice_quality_weighting=not args.no_slice_quality,
        maternal_spatial_weight=args.blend_maternal,
        uterine_envelope_weight=args.uterine_envelope_weight,
        blend_spatial_fetal=args.blend_fetal,
        verbose=not args.quiet,
        progress_every=max(1, args.progress_every),
        fastica_max_iter=max(50, args.fastica_max_iter),
        fastica_tol=max(1e-6, args.fastica_tol),
        residual_ica_fit_max_rows=max(5000, args.residual_ica_fit_rows),
    )
    print(f"BSS core done in {time.time() - t0:.1f}s", flush=True)
    print("Writing CSV...", flush=True)
    save_outputs_csv(out, fs, args.out)
    print("Rendering QC figure (may take a bit on long records)...", flush=True)
    plot_qc(
        out,
        fs,
        max_seconds=args.plot_seconds,
        save_path=f"{args.out}_qc.png",
        max_plot_points=args.plot_max_points,
    )
    print(f"Wrote {args.out}_three_channels.csv and {args.out}_qc.png", flush=True)

    print("Computing dynamic HR (peaks + RR) from separated traces...", flush=True)
    hr = compute_hr_from_bss_output(out, fs)
    save_hr_csv(hr, args.out)
    hr_plot_pts = min(int(args.plot_max_points), 150_000)
    plot_hr_dynamics(hr, save_path=f"{args.out}_hr_dynamics.png", max_plot_points=hr_plot_pts)
    print(
        f"Wrote {args.out}_hr_dynamics.png (new); "
        f"{args.out}_maternal_peaks.csv, {args.out}_fetal_peaks.csv; "
        f"{args.out}_maternal_hr_instantaneous.csv, {args.out}_fetal_hr_instantaneous.csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
