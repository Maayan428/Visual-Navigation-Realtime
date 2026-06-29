#!/usr/bin/env python3
"""
Real-Time Visual Navigation for Drones (GNSS-Denied)
=====================================================

Three operating modes:

  preprocess  – Build the geo-referenced visual database from a reference flight.
  navigate    – Run real-time localization on a query video.
  experiment  – Formal evaluation against ground-truth GPS.

Examples
--------
  python main.py preprocess --srt data/DJI_0017.SRT \\
                            --video "data/DJI 0006.mp4" \\
                            --out-dir db/

  python main.py navigate   --db db/ \\
                            --video "data/DJI 0007.mp4" \\
                            --srt data/DJI_0007.SRT \\
                            --out-dir out/

  python main.py experiment --db db/ \\
                            --query-srt data/DJI_0019.SRT \\
                            --query-video data/DJI_0019.MP4 \\
                            --out-dir out/
"""

import argparse
import os
import sys


def _resolve(path: str) -> str:
    """Return absolute path; resolve relative paths from the current directory."""
    return os.path.abspath(path)


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        sys.exit(f"File not found ({label}): {path}")


def _require_dir(path: str, label: str) -> None:
    if not os.path.isdir(path):
        sys.exit(f"Directory not found ({label}): {path}")


# -----------------------------------------------------------------------
# Subcommand handlers
# -----------------------------------------------------------------------

def cmd_preprocess(args) -> None:
    from preprocess import build_database
    source  = args.source
    out_dir = _resolve(args.out_dir)
    gis_dir = _resolve(args.gis_dir)

    srt = video = None
    if source in ('video', 'both'):
        if not args.srt or not args.video:
            sys.exit('--srt and --video are required when --source is video or both')
        srt   = _resolve(args.srt)
        video = _resolve(args.video)
        _require_file(srt,   '--srt')
        _require_file(video, '--video')

    if source in ('gis', 'both'):
        _require_dir(gis_dir, '--gis-dir')

    build_database(srt, video, out_dir,
                   sample_every=args.sample_every,
                   device=args.device,
                   fast=args.fast,
                   source=source,
                   gis_dir=gis_dir)


def cmd_navigate(args) -> None:
    from navigator import RealTimeNavigator
    db = _resolve(args.db)
    video = _resolve(args.video)
    out_dir = _resolve(args.out_dir)
    _require_dir(db, '--db')
    _require_file(video, '--video')

    srt = _resolve(args.srt) if args.srt else None
    if srt:
        _require_file(srt, '--srt')

    nav = RealTimeNavigator(db_path=db, device=args.device, fast=args.fast,
                            min_inliers=args.min_inliers)
    nav.locate_video_stream(video_path=video, srt_path=srt, out_dir=out_dir)


def cmd_experiment(args) -> None:
    from experiment import run_experiment
    db = _resolve(args.db)
    query_srt = _resolve(args.query_srt)
    query_video = _resolve(args.query_video)
    out_dir = _resolve(args.out_dir)
    _require_dir(db, '--db')
    _require_file(query_srt, '--query-srt')
    _require_file(query_video, '--query-video')
    run_experiment(db, query_srt, query_video, out_dir, fast=args.fast,
                   min_inliers=args.min_inliers)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Real-Time Visual Navigation for Drones (GNSS-Denied)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--device', default='cpu',
                        help='PyTorch device (cpu or cuda)')

    sub = parser.add_subparsers(dest='command', required=True)

    _fast_arg = dict(action='store_true', default=True,
                     help='Use ORB + BFMatcher (fast, CPU); default on')
    _no_fast_arg = dict(action='store_false', dest='fast',
                        help='Use SuperPoint + LightGlue (accurate, GPU recommended)')

    # ---- preprocess ----
    p_pre = sub.add_parser('preprocess', help='Build geo-referenced visual database')
    p_pre.add_argument('--srt',          default=None, help='Reference flight SRT file')
    p_pre.add_argument('--video',        default=None, help='Reference flight MP4 file')
    p_pre.add_argument('--out-dir',      default='db/', help='Output directory for database')
    p_pre.add_argument('--sample-every', type=int, default=15,
                       help='Process every Nth video frame (default 15 ≈ 2 fps)')
    p_pre.add_argument('--source',       default='video',
                       choices=['video', 'gis', 'both'],
                       help='Data source: video frames, GIS satellite tiles, or both')
    p_pre.add_argument('--gis-dir',      default='db_gis/',
                       help='Satellite patch directory (from map_fetcher.py)')
    p_pre.add_argument('--fast',    **_fast_arg)
    p_pre.add_argument('--no-fast', **_no_fast_arg)

    # ---- navigate ----
    p_nav = sub.add_parser('navigate', help='Localize a query video in real time')
    p_nav.add_argument('--db',           required=True, help='GeoDatabase directory')
    p_nav.add_argument('--video',        required=True, help='Query flight MP4 file')
    p_nav.add_argument('--srt',          default=None,
                       help='[optional] Query SRT for ground-truth comparison')
    p_nav.add_argument('--out-dir',      default='out/', help='Output directory')
    p_nav.add_argument('--min-inliers',  type=int, default=4,
                       help='Min RANSAC inliers to accept a match (default 4)')
    p_nav.add_argument('--fast',    **_fast_arg)
    p_nav.add_argument('--no-fast', **_no_fast_arg)

    # ---- experiment ----
    p_exp = sub.add_parser('experiment', help='Evaluate localization accuracy')
    p_exp.add_argument('--db',           required=True, help='GeoDatabase directory')
    p_exp.add_argument('--query-srt',    required=True, help='Query flight SRT (ground truth)')
    p_exp.add_argument('--query-video',  required=True, help='Query flight MP4 file')
    p_exp.add_argument('--out-dir',      default='out/', help='Output directory')
    p_exp.add_argument('--min-inliers',  type=int, default=4,
                       help='Min RANSAC inliers to accept a match (default 4)')
    p_exp.add_argument('--fast',    **_fast_arg)
    p_exp.add_argument('--no-fast', **_no_fast_arg)

    args = parser.parse_args()

    if args.command == 'preprocess':
        cmd_preprocess(args)
    elif args.command == 'navigate':
        cmd_navigate(args)
    elif args.command == 'experiment':
        cmd_experiment(args)


if __name__ == '__main__':
    main()
