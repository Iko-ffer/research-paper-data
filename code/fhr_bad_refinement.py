#!/usr/bin/env python3
"""
5 Hz FHR jump–triggered bad windows: multiroute re-BSS on raw slice, splice back.

User rule: if two consecutive 5 Hz samples differ by more than ``jump_threshold_bpm``,
take the *later* sample time as trigger and mark [t - pre_sec, t + post_sec] for refinement.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple, Union

import numpy as np

from chunked_multiroute_bss import refine_segment_pick_best_route

try:
    from hybrid_v16_chunk_bss import (
        FETAL_COMPARE_ROUTES,
        refine_hybrid_fetal_segment,
    )
except ImportError:
    FETAL_COMPARE_ROUTES = ()
    refine_hybrid_fetal_segment = None  # type: ignore

try:
    from v1v6_chunk_bss import refine_v1v6_segment
except ImportError:
    refine_v1v6_segment = None  # type: ignore

try:
    from hybrid_v16_chunk_bss import neighbor_blend_maternal_edges
except ImportError:
    neighbor_blend_maternal_edges = None  # type: ignore

BadInterval = Dict[str, object]
RefineMode = Literal["fhr_jump", "mhr_gap", "both"]


def detect_fhr_jump_intervals(
    t_sec: np.ndarray,
    fhr: np.ndarray,
    jump_threshold_bpm: float = 5.0,
    pre_sec: float = 1.0,
    post_sec: float = 9.0,
) -> List[Tuple[float, float]]:
    """
    Build merged [t0, t1] intervals from 5 Hz FHR first-differences.

    Trigger time is ``t_sec[i]`` when ``|fhr[i]-fhr[i-1]| > threshold`` (second sample of the pair).
    """
    t_sec = np.asarray(t_sec, dtype=np.float64)
    fhr = np.asarray(fhr, dtype=np.float64)
    raw: List[Tuple[float, float]] = []
    for i in range(1, len(fhr)):
        if not (np.isfinite(fhr[i - 1]) and np.isfinite(fhr[i])):
            continue
        if abs(float(fhr[i] - fhr[i - 1])) > jump_threshold_bpm:
            t_trig = float(t_sec[i])
            raw.append((t_trig - pre_sec, t_trig + post_sec))
    if not raw:
        return []
    raw.sort(key=lambda x: x[0])
    merged = [raw[0]]
    for a, b in raw[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def merge_bad_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Union of [t0,t1] intervals with overlap merged."""
    if not intervals:
        return []
    raw = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged = [raw[0]]
    for a, b in raw[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def detect_mhr_gap_intervals(
    t_sec: np.ndarray,
    mhr: np.ndarray,
    valid: np.ndarray,
    *,
    min_gap_sec: float = 3.0,
    pre_sec: float = 1.0,
    post_sec: float = 9.0,
) -> List[Tuple[float, float]]:
    """
    Intervals where algorithm mHR is invalid for at least ``min_gap_sec`` (lost maternal peaks / causal window).
    """
    t_sec = np.asarray(t_sec, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).ravel()
    n = min(len(t_sec), len(valid))
    if n < 2:
        return []
    dt = float(t_sec[1] - t_sec[0]) if n > 1 else 0.2
    raw: List[Tuple[float, float]] = []
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        gap_len = (j - i) * dt
        if gap_len >= min_gap_sec:
            t0 = float(t_sec[i])
            t1 = float(t_sec[j - 1]) if j > i else t0
            raw.append((t0 - pre_sec, t1 + post_sec))
        i = j
    return merge_bad_intervals(raw)


def _splice_with_ramp(
    out: Dict[str, np.ndarray],
    s0: int,
    s1: int,
    m_new: np.ndarray,
    f_new: np.ndarray,
    u_new: np.ndarray,
    ramp_len: int,
    *,
    fetal_only: bool = False,
) -> None:
    """Replace out[s0:s1] for m/f/u (or fetal only) with linear crossfade ramps."""
    L = s1 - s0
    if len(f_new) != L:
        return
    ramp_len = max(0, min(ramp_len, L // 8))
    if fetal_only:
        keys = ("fetal_ecg",)
        news = (f_new,)
    else:
        keys = ("maternal_ecg", "fetal_ecg", "uterine_abdominal")
        news = (m_new, f_new, u_new)
    olds = tuple(out[k][s0:s1].copy() for k in keys)

    for key, newv, oldv in zip(keys, news, olds):
        if len(newv) != L:
            continue
        if ramp_len <= 0:
            out[key][s0:s1] = newv
            continue
        aL = np.linspace(0.0, 1.0, ramp_len, endpoint=False)
        out[key][s0 : s0 + ramp_len] = (1.0 - aL) * oldv[:ramp_len] + aL * newv[:ramp_len]
        out[key][s0 + ramp_len : s1 - ramp_len] = newv[ramp_len : L - ramp_len]
        aR = np.linspace(0.0, 1.0, ramp_len, endpoint=False)
        out[key][s1 - ramp_len : s1] = (1.0 - aR) * newv[L - ramp_len : L] + aR * oldv[L - ramp_len : L]


def build_tagged_bad_intervals(
    bad_fhr: List[Tuple[float, float]],
    bad_mhr: List[Tuple[float, float]],
) -> List[BadInterval]:
    """Merge FHR-jump and mHR-gap intervals; tag each with trigger reason(s)."""
    tagged: List[BadInterval] = []
    for a, b in bad_fhr:
        tagged.append({"t0": float(a), "t1": float(b), "fhr_jump": True, "mhr_gap": False})
    for a, b in bad_mhr:
        tagged.append({"t0": float(a), "t1": float(b), "fhr_jump": False, "mhr_gap": True})
    if not tagged:
        return []
    tagged.sort(key=lambda x: float(x["t0"]))
    merged: List[BadInterval] = [dict(tagged[0])]
    for item in tagged[1:]:
        la = float(merged[-1]["t0"])
        lb = float(merged[-1]["t1"])
        a, b = float(item["t0"]), float(item["t1"])
        if a <= lb:
            merged[-1]["t1"] = max(lb, b)
            merged[-1]["fhr_jump"] = bool(merged[-1]["fhr_jump"]) or bool(item["fhr_jump"])
            merged[-1]["mhr_gap"] = bool(merged[-1]["mhr_gap"]) or bool(item["mhr_gap"])
        else:
            merged.append(dict(item))
    return merged


def _interval_refine_mode(item: BadInterval) -> str:
    fj = bool(item.get("fhr_jump"))
    mg = bool(item.get("mhr_gap"))
    if fj and mg:
        return "both"
    if mg:
        return "mhr_gap"
    return "fhr_jump"


def apply_adaptive_bad_segment_refinement(
    out: Dict[str, np.ndarray],
    x6: np.ndarray,
    fs: float,
    tagged_intervals: List[BadInterval],
    bss_kwargs: dict,
    ramp_sec: float = 0.25,
    min_segment_sec: float = 0.8,
) -> int:
    """
    Mode-aware bad-window refine: FHR-jump vs mHR-gap use different BSS/refine strategies.
    """
    n = int(len(out["fetal_ecg"]))
    t_end = n / fs
    n_done = 0
    ramp_len = max(0, int(ramp_sec * fs))
    hybrid_refine = bool(bss_kwargs.get("hybrid_v16_refine"))
    v1v6_refine = bool(bss_kwargs.get("v1v6_single_ica_refine"))

    for item in tagged_intervals:
        t0c = max(0.0, float(item["t0"]))
        t1c = min(t_end, float(item["t1"]))
        if t1c - t0c < min_segment_sec:
            continue
        s0 = max(0, int(np.floor(t0c * fs)))
        s1 = min(n, int(np.ceil(t1c * fs)))
        if s1 - s0 < int(min_segment_sec * fs):
            continue
        seg = x6[s0:s1].copy()
        mode = _interval_refine_mode(item)
        try:
            chunk_sec = float(bss_kwargs.get("chunk_sec", 10.0))
            if bss_kwargs.get("v1v6_single_ica_refine") and refine_v1v6_segment is not None:
                m, f, u, _, _ = refine_v1v6_segment(seg, fs, chunk_sec, bss_kwargs)
            elif hybrid_refine and refine_hybrid_fetal_segment is not None:
                routes = tuple(bss_kwargs.get("fetal_routes", FETAL_COMPARE_ROUTES))
                m, f, u, _, _ = refine_hybrid_fetal_segment(
                    seg, fs, chunk_sec, bss_kwargs, routes=routes
                )
            else:
                m, f, u, _, _ = refine_segment_pick_best_route(seg, fs, bss_kwargs)
        except Exception:
            continue
        if bool(item.get("mhr_gap")) and neighbor_blend_maternal_edges is not None:
            bridge = float(bss_kwargs.get("refine_neighbor_bridge_sec", 2.0))
            m = neighbor_blend_maternal_edges(m, out["maternal_ecg"], s0, s1, fs, bridge)
        fetal_only = (hybrid_refine or v1v6_refine) and not bool(item.get("mhr_gap"))
        _splice_with_ramp(out, s0, s1, m, f, u, ramp_len, fetal_only=fetal_only)
        n_done += 1
    return n_done


def apply_fhr_bad_segment_refinement(
    out: Dict[str, np.ndarray],
    x6: np.ndarray,
    fs: float,
    bad_intervals: Union[List[Tuple[float, float]], List[BadInterval]],
    bss_kwargs: dict,
    ramp_sec: float = 0.25,
    min_segment_sec: float = 0.8,
    *,
    refine_full_traces: bool = False,
) -> int:
    """
    For each bad interval, re-run multiroute BSS on ``x6`` and splice best m/f/u into ``out``.

    Accepts plain ``(t0,t1)`` tuples or tagged dicts from ``build_tagged_bad_intervals``.
    """
    if bad_intervals and isinstance(bad_intervals[0], dict):
        return apply_adaptive_bad_segment_refinement(
            out, x6, fs, bad_intervals, bss_kwargs, ramp_sec=ramp_sec, min_segment_sec=min_segment_sec
        )
    tagged = [
        {
            "t0": float(a),
            "t1": float(b),
            "fhr_jump": True,
            "mhr_gap": bool(refine_full_traces),
        }
        for a, b in bad_intervals  # type: ignore[misc]
    ]
    return apply_adaptive_bad_segment_refinement(
        out, x6, fs, tagged, bss_kwargs, ramp_sec=ramp_sec, min_segment_sec=min_segment_sec
    )
