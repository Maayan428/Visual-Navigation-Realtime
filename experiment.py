"""
Formal evaluation of the real-time visual navigation system.

Compares GPS estimates produced by the navigator against ground-truth GPS
from the query flight's SRT file, and reports standard localization metrics.

Usage
-----
    python experiment.py --db db/ \
                         --query-srt data/DJI_0019.SRT \
                         --query-video data/DJI_0019.MP4 \
                         --out-dir out/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geo_utils import haversine
from navigator import RealTimeNavigator


def run_experiment(db_path: str, query_srt: str, query_video: str,
                   out_dir: str = 'out/', fast: bool = True,
                   min_inliers: int = 4) -> dict:
    """Full evaluation pipeline comparing estimated GPS to ground truth.

    Parameters
    ----------
    db_path : str
        Path to the GeoDatabase directory (built by preprocess.py).
    query_srt : str
        SRT file for the query flight (GPS is used as ground truth only).
    query_video : str
        MP4 video of the query flight.
    out_dir : str
        Directory for output plots and CSV.
    fast : bool
        If True (default), use ORB + BFMatcher; if False use SuperPoint + LightGlue.

    Returns
    -------
    dict
        Metrics:
          mean_error_m, median_error_m, p90_error_m, max_error_m,
          location_rate, mean_processing_time_ms, mean_num_inliers
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load navigator (GPS is withheld from the navigator itself)
    nav = RealTimeNavigator(db_path=db_path, fast=fast, min_inliers=min_inliers)

    # 2. Process the query video (SRT is passed only for ground-truth evaluation)
    print(f"\nProcessing query video: {query_video}")
    results_df = nav.locate_video_stream(
        video_path=query_video,
        srt_path=query_srt,
        out_dir=out_dir,
    )

    # 3. Compute metrics — guard against missing columns when nothing was localized
    for col in ('est_lat', 'est_lon', 'processing_time_ms', 'num_inliers',
                'error_m', 'geo_rejected_count'):
        if col not in results_df.columns:
            results_df[col] = float('nan') if col != 'geo_rejected_count' else 0
    located = results_df.dropna(subset=['est_lat', 'est_lon'])
    total_frames = len(results_df)
    located_frames = len(located)
    location_rate = located_frames / total_frames if total_frames > 0 else 0.0
    frames_rejected_by_geo_filter = int(results_df['geo_rejected_count'].sum())

    # Recompute errors for frames that have both estimate and ground truth
    valid = located.dropna(subset=['gt_lat', 'gt_lon'])
    if len(valid) > 0:
        errors = valid.apply(
            lambda r: haversine(r['gt_lat'], r['gt_lon'], r['est_lat'], r['est_lon']),
            axis=1,
        ).values
    else:
        errors = np.array([])

    mean_error = float(np.mean(errors)) if len(errors) > 0 else float('nan')
    median_error = float(np.median(errors)) if len(errors) > 0 else float('nan')
    p90_error = float(np.percentile(errors, 90)) if len(errors) > 0 else float('nan')
    max_error = float(np.max(errors)) if len(errors) > 0 else float('nan')

    proc_times = located['processing_time_ms'].dropna()
    mean_time = float(proc_times.mean()) if len(proc_times) > 0 else float('nan')

    inlier_col = located.get('num_inliers', pd.Series(dtype=float)) if 'num_inliers' in located.columns else pd.Series(dtype=float)
    mean_inliers = float(inlier_col.mean()) if len(inlier_col) > 0 else float('nan')

    metrics = {
        'mean_error_m':                  mean_error,
        'median_error_m':                median_error,
        'p90_error_m':                   p90_error,
        'max_error_m':                   max_error,
        'location_rate':                 location_rate,
        'located_frames':                located_frames,
        'total_frames':                  total_frames,
        'frames_rejected_by_geo_filter': frames_rejected_by_geo_filter,
        'mean_processing_time_ms':       mean_time,
        'mean_num_inliers':              mean_inliers,
    }

    # 4. Generate plots
    _plot_error_histogram(errors, out_dir)
    _plot_error_over_time(results_df, out_dir)

    # 5. Print summary table
    print("\n" + "=" * 55)
    print("  LOCALIZATION EXPERIMENT RESULTS")
    print("=" * 55)
    print(f"  Frames processed     : {total_frames}")
    print(f"  Frames localized     : {located_frames}  "
          f"({location_rate * 100:.1f}%)")
    print(f"  Rejected by geo filter: {frames_rejected_by_geo_filter}")
    if len(errors) > 0:
        print(f"  Mean error           : {mean_error:.1f} m")
        print(f"  Median error         : {median_error:.1f} m")
        print(f"  90th percentile error: {p90_error:.1f} m")
        print(f"  Max error            : {max_error:.1f} m")
    print(f"  Mean processing time : {mean_time:.1f} ms/frame")
    print(f"  Mean inliers         : {mean_inliers:.1f}")
    print("=" * 55)

    return metrics


# -----------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------

def _plot_error_histogram(errors: np.ndarray, out_dir: str) -> None:
    if len(errors) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=40, color='steelblue', edgecolor='white')
    ax.axvline(float(np.median(errors)), color='red', linestyle='--',
               label=f'Median {np.median(errors):.1f} m')
    ax.axvline(float(np.percentile(errors, 90)), color='orange', linestyle='--',
               label=f'P90 {np.percentile(errors, 90):.1f} m')
    ax.set_xlabel('Localization error (m)')
    ax.set_ylabel('Number of frames')
    ax.set_title('Error distribution')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'error_histogram.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Error histogram saved to {path}")


def _plot_error_over_time(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['frame_idx'], df.get('error_m', [None] * len(df)),
            color='steelblue', linewidth=0.8, label='Error (m)')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Localization error (m)')
    ax.set_title('Localization error over time')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'error_over_time.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Error-over-time plot saved to {path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate visual navigation accuracy against ground-truth GPS.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--db',           required=True, help='GeoDatabase directory')
    parser.add_argument('--query-srt',    required=True, help='Query flight SRT file')
    parser.add_argument('--query-video',  required=True, help='Query flight MP4 file')
    parser.add_argument('--out-dir',      default='out/', help='Output directory')
    parser.add_argument('--fast', action=argparse.BooleanOptionalAction, default=True,
                        help='Use ORB (fast, CPU) instead of SuperPoint (default: True)')
    args = parser.parse_args()

    run_experiment(args.db, args.query_srt, args.query_video, args.out_dir,
                   fast=args.fast)


if __name__ == '__main__':
    main()
