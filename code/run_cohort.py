#!/usr/bin/env python3
"""CLI entry for batch or single-subject pipeline runs."""

from __future__ import annotations

import argparse
import os

from base_pipeline import PipelineConfig, run_batch
from hybrid_v16_chunk_bss import FETAL_COMPARE_ROUTES


def main() -> None:
    p = argparse.ArgumentParser(description="Six-channel abdominal signal batch pipeline")
    p.add_argument(
        "--data-root",
        default=".",
        help="Root folder containing subject subdirectories",
    )
    p.add_argument(
        "--output-root",
        default="Output",
        help="Output root; each run creates Output/RUN-YYYYMMDD-HHMMSS/",
    )
    p.add_argument(
        "--subject",
        default=None,
        help="Process only this subject (e.g. 代淼), or comma-separated names (e.g. 代淼,夏天茂,王娟)",
    )
    p.add_argument("--window", type=float, default=10.0, help="ICA window (s); hybrid mode = chunk length")
    p.add_argument("--overlap", type=float, default=0.75, help="Inner overlap ratio (only if --bss-hop-sec < 0)")
    p.add_argument(
        "--bss-hop-sec",
        type=float,
        default=10.0,
        help="ICA hop (s). Hybrid default 10 = one ICA per 10s chunk",
    )
    p.add_argument("--no-template-cancel", action="store_true", help="Disable maternal template cancel")
    p.add_argument(
        "--v1v6-single-ica",
        action="store_true",
        help="V1/V6+hedge single ICA per 10s chunk (replaces hybrid; primary fECG from fetal IC)",
    )
    p.add_argument(
        "--no-hybrid-v16",
        action="store_true",
        help="Disable hybrid single2+V1/V6 ICA (10s non-overlapping chunks)",
    )
    p.add_argument(
        "--no-single2-dual-polarity",
        action="store_true",
        help="Disable V1/V6 hedge sign search (use legacy fixed v1+k*v6 only)",
    )
    p.add_argument(
        "--no-single2-polarity-v2",
        action="store_true",
        help="Use legacy polarity pick (fetal_route_quality only, no maternal suppression gates)",
    )
    p.add_argument(
        "--legacy-chunked",
        action="store_true",
        help="Old 10s/8s overlap-add 6ch multiroute (requires --no-hybrid-v16)",
    )
    p.add_argument("--chunk-sec", type=float, default=10.0, help="Chunk length (s)")
    p.add_argument(
        "--chunk-hop-sec",
        type=float,
        default=10.0,
        help="Chunk hop (s); hybrid uses non-overlapping hop=chunk",
    )
    p.add_argument(
        "--no-multiroute",
        action="store_true",
        help="Legacy chunked mode: native ICA only per outer window",
    )
    p.add_argument("--fhr-hz", type=float, default=5.0, help="Output FHR sample rate (match gold)")
    p.add_argument("--fhr-smooth-sec", type=float, default=2.5, help="Causal window for mean instantaneous FHR")
    p.add_argument("--quiet", action="store_true", help="Less console output")
    p.add_argument("--no-fhr-bad-refine", action="store_true", help="Disable 5Hz FHR jump bad-segment refine")
    p.add_argument(
        "--fhr-bad-refine-max-passes",
        type=int,
        default=2,
        help="Max FHR-jump re-BSS passes per segment (prevents infinite loops; default 2)",
    )
    p.add_argument(
        "--maternal-defetal",
        action="store_true",
        help="Post-hoc fetal-band subtraction on maternal (default on for ica_split)",
    )
    p.add_argument(
        "--no-maternal-defetal",
        action="store_true",
        help="Disable maternal defetal even when using ica_split",
    )
    p.add_argument(
        "--legacy-preprocess",
        action="store_true",
        help="Raw 6ch: only median + 50Hz notch (no HP, no 100Hz notch, no per-channel scale)",
    )
    p.add_argument(
        "--separation-mode",
        default="standard",
        choices=("standard", "physics_first", "ica_dual", "ica_split"),
        help="ICA mode: standard = RUN-123345 baseline (joint 12->6 PCA per slice)",
    )
    p.add_argument(
        "--fhr-output-jump-smooth-bpm",
        type=float,
        default=10.0,
        help="Median-smooth while adjacent valid FHR differ by more than this (0=off)",
    )
    p.add_argument(
        "--fhr-final-median-halfwin",
        type=int,
        default=3,
        help="Final symmetric neighbor median on 5 Hz grid (0=off)",
    )
    p.add_argument("--vref-weight-ch0", type=float, default=1.0)
    p.add_argument("--vref-weight-ch5", type=float, default=1.0)
    p.add_argument(
        "--ica-obs-spatial-weights",
        action="store_true",
        help="ICA observation: preset per-channel gains (demote ch3–ch4, boost fetal-band ch1/ch6) before PCA stack",
    )
    p.add_argument(
        "--ica-v1v6-bipoles",
        action="store_true",
        help="ICA PCA stack: add V1+V6 and V1-V6 per maternal/fetal band (16-D → PCA→6)",
    )
    p.add_argument(
        "--fhr-causal-on-fused",
        action="store_true",
        help="5 Hz FHR from causal peaks on fused fECG (default: single2-style interp+medfilt on fused trace)",
    )
    p.add_argument(
        "--fetal-ica-hedge-obs",
        action="store_true",
        help="Fetal ica_split on band 6ch + adaptive V1±K·V6 hedge ICA obs; single2 hard-gates (no blend)",
    )
    p.add_argument(
        "--fetal-ica-middle-bipoles",
        action="store_true",
        help="With --fetal-ica-hedge-obs: append middle lateral bipoles to fetal ICA observation",
    )
    p.add_argument(
        "--fhr-sparse-grid",
        action="store_true",
        help="5 Hz FHR: NaN between peaks (no full-grid interpolation); also writes *_fhr_5hz_sparse.csv",
    )
    p.add_argument(
        "--anti-uc-wander",
        action="store_true",
        help="Fetal path: single2 8–40 Hz order-4 + FHR rolling-median detrend 0.8 s (see METHOD doc)",
    )
    p.add_argument(
        "--single2-band-low",
        type=float,
        default=None,
        help="Single2 hedge bandpass low Hz (default 5; anti-uc uses 8)",
    )
    p.add_argument(
        "--single2-band-high",
        type=float,
        default=40.0,
        help="Single2 hedge bandpass high Hz",
    )
    p.add_argument(
        "--single2-bandpass-order",
        type=int,
        default=None,
        help="Butterworth order for single2 hedge bandpass (default 2; anti-uc uses 4)",
    )
    p.add_argument(
        "--fecg-fhr-detrend-sec",
        type=float,
        default=None,
        help="Rolling median detrend window on fECG before FHR peaks (0=off)",
    )
    p.add_argument(
        "--fecg-fhr-bandpass-low",
        type=float,
        default=None,
        help="Extra FHR-only bandpass low Hz on fECG before peaks (optional)",
    )
    p.add_argument(
        "--fecg-fhr-bandpass-order",
        type=int,
        default=4,
        help="Order for optional FHR-only bandpass",
    )
    p.add_argument(
        "--mhr-peak-max-bpm",
        type=float,
        default=100.0,
        help="Maternal peak-pick RR upper limit (default 100; use 120 with wider gap)",
    )
    p.add_argument(
        "--fhr-peak-min-bpm",
        type=float,
        default=110.0,
        help="Fetal peak-pick RR lower limit (default 110; user band often 105)",
    )
    p.add_argument(
        "--fhr-peak-max-bpm",
        type=float,
        default=170.0,
        help="Fetal peak-pick RR upper limit",
    )
    p.add_argument(
        "--peak-bpm-guard-gap",
        type=float,
        default=8.0,
        help="Min BPM gap between maternal max and fetal min windows",
    )
    args = p.parse_args()

    if args.legacy_chunked and not args.no_hybrid_v16:
        p.error("--legacy-chunked requires --no-hybrid-v16")
    if args.v1v6_single_ica and args.legacy_chunked:
        p.error("--v1v6-single-ica cannot be used with --legacy-chunked")

    use_v1v6 = bool(args.v1v6_single_ica)
    use_hybrid = (not args.no_hybrid_v16) and (not use_v1v6)
    use_chunked = bool(args.legacy_chunked)

    data_root = os.path.abspath(args.data_root)
    legacy_pre = bool(args.legacy_preprocess)

    if args.anti_uc_wander:
        s2_lo, s2_ord, detrend_sec, fhr_bp = 8.0, 4, 0.8, None
    else:
        s2_lo = 5.0 if args.single2_band_low is None else float(args.single2_band_low)
        s2_ord = 2 if args.single2_bandpass_order is None else int(args.single2_bandpass_order)
        detrend_sec = 0.0 if args.fecg_fhr_detrend_sec is None else float(args.fecg_fhr_detrend_sec)
        fhr_bp = None
        if args.fecg_fhr_bandpass_low is not None:
            fhr_bp = (float(args.fecg_fhr_bandpass_low), float(args.single2_band_high))

    config = PipelineConfig(
        window_sec=args.window,
        overlap=args.overlap,
        template_cancel=not args.no_template_cancel,
        use_v1v6_single_ica=use_v1v6,
        use_hybrid_v16_fetal=use_hybrid,
        fetal_chunk_routes=FETAL_COMPARE_ROUTES,
        separation_mode="v1v6_single_ica" if use_v1v6 else args.separation_mode,
        use_chunked=use_chunked,
        chunk_sec=args.chunk_sec,
        chunk_hop_sec=args.chunk_hop_sec if use_chunked else args.chunk_sec,
        bss_inner_hop_sec=None if args.bss_hop_sec < 0 else float(args.bss_hop_sec),
        multiroute=not args.no_multiroute,
        fhr_output_hz=args.fhr_hz,
        fhr_smooth_sec=args.fhr_smooth_sec,
        use_fhr_jump_bad_refinement=not args.no_fhr_bad_refine,
        fhr_bad_refine_max_passes=max(1, int(args.fhr_bad_refine_max_passes)),
        maternal_defetal=bool(args.maternal_defetal) and not args.no_maternal_defetal,
        quiet=args.quiet,
        preprocess_notch_100=False if legacy_pre else True,
        preprocess_baseline_highpass_hz=0.0 if legacy_pre else 0.35,
        preprocess_per_channel_scale="none" if legacy_pre else "robust",
        fhr_output_jump_smooth_bpm=float(args.fhr_output_jump_smooth_bpm),
        fhr_output_slow_reconcile_halfwin=0 if use_hybrid else 20,
        fhr_output_final_median_halfwin=int(args.fhr_final_median_halfwin),
        single2_dual_polarity=not args.no_single2_dual_polarity,
        single2_polarity_v2=not args.no_single2_polarity_v2,
        vref_weight_ch0=float(args.vref_weight_ch0),
        vref_weight_ch5=float(args.vref_weight_ch5),
        use_ica_obs_spatial_guided_weights=bool(args.ica_obs_spatial_weights),
        ica_obs_append_v1v6_bipoles=bool(args.ica_v1v6_bipoles),
        fhr_from_fused_fecg=bool(args.fhr_causal_on_fused),
        hybrid_fetal_hedge_ica_obs=bool(args.fetal_ica_hedge_obs),
        ica_obs_append_middle_bipoles=bool(args.fetal_ica_middle_bipoles),
        fhr_fill_5hz_grid=not bool(args.fhr_sparse_grid),
        single2_band_hz=(float(s2_lo), float(args.single2_band_high)),
        single2_bandpass_order=int(s2_ord),
        fecg_fhr_detrend_median_sec=float(detrend_sec),
        fecg_fhr_bandpass_hz=fhr_bp,
        fecg_fhr_bandpass_order=int(args.fecg_fhr_bandpass_order),
        mhr_peak_bpm_hz=(45.0, float(args.mhr_peak_max_bpm)),
        fhr_peak_bpm_hz=(float(args.fhr_peak_min_bpm), float(args.fhr_peak_max_bpm)),
        peak_bpm_guard_gap_bpm=float(args.peak_bpm_guard_gap),
    )
    run_batch(
        data_root=data_root,
        output_root=args.output_root,
        subject_filter=args.subject,
        config=config,
    )


if __name__ == "__main__":
    main()
