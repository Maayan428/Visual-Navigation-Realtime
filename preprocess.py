"""
Offline database builder for the real-time visual navigation system.

Reads a DJI reference flight (SRT + MP4), samples frames at a configurable
rate, extracts SuperPoint features, and saves a FAISS-indexed GeoDatabase.

Usage
-----
    python preprocess.py --srt data/DJI_0017.SRT \
                         --video "data/DJI 0017.MP4" \
                         --out-dir db/ \
                         --sample-every 15
"""

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
from tqdm import tqdm

from feature_extractor import SuperPointExtractor, ORBExtractor, extract_global_descriptor
from geo_utils import compute_ground_footprint
from retrieval import GeoDatabase


# -----------------------------------------------------------------------
# Dense SRT parser (all frames, not just 1 fps)
# -----------------------------------------------------------------------

def _parse_srt_raw(path: str) -> dict:
    """Parse a DJI SRT file and return a mapping {frame_cnt: record}.

    Unlike srt_parser.parse_srt(), this function keeps ALL frames
    (not just every 30th), enabling dense video sampling.  Zero-GPS
    frames are still filtered out.

    Parameters
    ----------
    path : str
        Path to the DJI SRT telemetry file.

    Returns
    -------
    dict
        {frame_cnt (int): record (dict)} where record contains at least
        'lat', 'lon', 'rel_alt', 'abs_alt', 'timestamp'.
    """
    with open(path, encoding='utf-8-sig') as f:
        text = f.read()

    blocks = re.split(r'\n\s*\n', text.strip())
    records = {}

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 5:
            continue

        fc_line = re.sub(r'<[^>]+>', '', lines[2]).strip()
        m = re.search(r'(?:FrameCnt|SrtCnt)\s*:\s*(\d+)', fc_line)
        if not m:
            continue
        frame_cnt = int(m.group(1))

        timestamp = lines[3].strip()
        tel_line = re.sub(r'<[^>]+>', '', lines[4]).strip()

        # Reuse the same field extraction logic as srt_parser._parse_telemetry
        record = {'timestamp': timestamp}
        for fm in re.finditer(r'\[(\w+)\s*:\s*([^\]\s,]+)\]', tel_line):
            key, val = fm.group(1), fm.group(2)
            if key == 'latitude':
                record['lat'] = float(val)
            elif key == 'longitude':
                record['lon'] = float(val)
            elif key == 'focal_len':
                record['focal_len'] = float(val)

        ma = re.search(r'\[rel_alt:\s*([\d.]+)\s+abs_alt:\s*([\d.]+)\]', tel_line)
        if ma:
            record['rel_alt'] = float(ma.group(1))
            record['abs_alt'] = float(ma.group(2))

        if 'lat' not in record or 'lon' not in record or 'rel_alt' not in record:
            continue
        if record['lat'] == 0.0 and record['lon'] == 0.0:
            continue

        records[frame_cnt] = record

    return records


def _interpolate_gps(frame_cnt: int, srt_map: dict) -> dict | None:
    """Return GPS record for frame_cnt, interpolating between neighbours if needed."""
    if frame_cnt in srt_map:
        return srt_map[frame_cnt]

    keys = sorted(srt_map.keys())
    if not keys:
        return None

    # Binary search for surrounding keys
    lo, hi = None, None
    for k in keys:
        if k < frame_cnt:
            lo = k
        elif k > frame_cnt and hi is None:
            hi = k
            break

    if lo is None:
        return srt_map[keys[0]]
    if hi is None:
        return srt_map[keys[-1]]

    # Linear interpolation
    t = (frame_cnt - lo) / (hi - lo)
    r0, r1 = srt_map[lo], srt_map[hi]
    rec = dict(r0)
    rec['lat'] = r0['lat'] + t * (r1['lat'] - r0['lat'])
    rec['lon'] = r0['lon'] + t * (r1['lon'] - r0['lon'])
    rec['rel_alt'] = r0['rel_alt'] + t * (r1['rel_alt'] - r0['rel_alt'])
    return rec


# -----------------------------------------------------------------------
# GIS patch loader
# -----------------------------------------------------------------------

def _load_gis_patches(gis_dir: str, extractor, db: 'GeoDatabase') -> tuple:
    """Extract features from pre-downloaded satellite patches and add to db.

    Returns
    -------
    tuple (added: int, skipped: int)
    """
    meta_path   = os.path.join(gis_dir, 'metadata.json')
    patches_dir = os.path.join(gis_dir, 'patches')

    with open(meta_path) as f:
        metadata = json.load(f)

    added = skipped = 0
    pbar = tqdm(metadata, desc='Loading GIS patches')

    for meta in pbar:
        npy_path = os.path.join(patches_dir, meta['filename'])
        if not os.path.exists(npy_path):
            skipped += 1
            continue

        gray = np.load(npy_path)                    # (H, W) uint8
        bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray

        feats = extractor.extract(bgr)
        kp    = feats['keypoints']
        desc  = feats['descriptors']

        if len(kp) < 50:
            skipped += 1
            continue

        global_desc = extract_global_descriptor(bgr, extractor)
        gsd         = meta['gsd_m_per_px']
        patch_m     = meta['width_m']

        frame_meta = {
            'frame_id':     f"gis_{meta['row']}_{meta['col']}",
            'lat':          meta['lat'],
            'lon':          meta['lon'],
            'alt':          0.0,       # sentinel: satellite, not drone
            'timestamp':    '',
            'camera_lat':   meta['lat'],
            'camera_lon':   meta['lon'],
            'gsd_m_per_px': gsd,
            'gsd':          gsd,
            'width_m':      patch_m,
            'height_m':     patch_m,
            'source_video': 'gis',
            'image_w':      640,
            'image_h':      640,
        }

        db.add_frame(frame_meta, global_desc, kp, desc)
        added += 1

    pbar.close()
    return added, skipped


# -----------------------------------------------------------------------
# Main build function
# -----------------------------------------------------------------------

def build_database(srt_path: str | None, video_path: str | None, out_dir: str,
                   sample_every: int = 15, device: str = 'cpu',
                   fast: bool = True, source: str = 'video',
                   gis_dir: str = 'db_gis/') -> 'GeoDatabase':
    """Build a geo-referenced visual database from a reference flight and/or GIS tiles.

    Parameters
    ----------
    srt_path : str or None
        DJI SRT telemetry file.  Required when source is 'video' or 'both'.
    video_path : str or None
        Companion MP4 video file.  Required when source is 'video' or 'both'.
    out_dir : str
        Directory where the database will be saved.
    sample_every : int
        Process every Nth video frame (default 15 ≈ 2 fps for 30 fps video).
    device : str
        PyTorch device.  Ignored in fast mode.
    fast : bool
        If True (default), use ORB features.
        If False, use SuperPoint + LightGlue.
    source : str
        One of 'video', 'gis', or 'both'.
    gis_dir : str
        Directory containing satellite patches from map_fetcher.py.
        Required when source is 'gis' or 'both'.

    Returns
    -------
    GeoDatabase
    """
    print(f"Mode: {'FAST (ORB)' if fast else 'ACCURATE (SuperPoint)'}  "
          f"source={source}")

    extractor: ORBExtractor | SuperPointExtractor = (
        ORBExtractor() if fast else SuperPointExtractor(device=device)
    )
    db = GeoDatabase()

    # ------------------------------------------------------------------
    # Video frames
    # ------------------------------------------------------------------
    if source in ('video', 'both'):
        print(f"\nParsing SRT: {srt_path}")
        srt_map = _parse_srt_raw(srt_path)
        print(f"  {len(srt_map)} SRT frames with valid GPS")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            sys.exit(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Video: {total_frames} frames @ {fps:.1f} fps  ({img_w}×{img_h})")

        source_name = os.path.basename(video_path)
        skipped_gps = skipped_kp = vid_added = 0

        pbar = tqdm(total=total_frames // sample_every + 1, desc='Building DB (video)')
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue

            frame_cnt = frame_idx
            gps = _interpolate_gps(frame_cnt, srt_map)
            if gps is None or gps['lat'] == 0.0:
                skipped_gps += 1
                pbar.update(1)
                continue

            lat, lon, alt = gps['lat'], gps['lon'], gps.get('rel_alt', 50.0)
            fp = compute_ground_footprint(lat, lon, alt)

            feats = extractor.extract(frame_bgr)
            kp    = feats['keypoints']
            desc  = feats['descriptors']

            if len(kp) < 50:
                skipped_kp += 1
                pbar.update(1)
                continue

            global_desc = extract_global_descriptor(frame_bgr, extractor)

            frame_meta = {
                'frame_id':     frame_cnt,
                'lat':          lat,
                'lon':          lon,
                'alt':          alt,
                'timestamp':    gps.get('timestamp', ''),
                'camera_lat':   fp['center_lat'],
                'camera_lon':   fp['center_lon'],
                'gsd_m_per_px': fp['gsd_m_per_px'],
                'gsd':          fp['gsd'],
                'width_m':      fp['width_m'],
                'height_m':     fp['height_m'],
                'source_video': source_name,
                'image_w':      img_w,
                'image_h':      img_h,
            }

            db.add_frame(frame_meta, global_desc, kp, desc)
            vid_added += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        print(f"  Video frames added: {vid_added}  "
              f"(skipped GPS={skipped_gps}, kp<50={skipped_kp})")

    # ------------------------------------------------------------------
    # GIS satellite patches
    # ------------------------------------------------------------------
    if source in ('gis', 'both'):
        print(f"\nLoading GIS patches from {gis_dir} ...")
        gis_added, gis_skipped = _load_gis_patches(gis_dir, extractor, db)
        print(f"  GIS patches added: {gis_added}  (skipped {gis_skipped})")

    if len(db.records) == 0:
        sys.exit("No frames were added to the database. Check input paths.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    lats = [r['lat'] for r in db.records]
    lons = [r['lon'] for r in db.records]
    drone_alts = [r['alt'] for r in db.records if r.get('alt', 0) > 0]
    print(f"\nTotal frames in DB : {len(db.records)}")
    print(f"  GPS coverage: lat [{min(lats):.6f}, {max(lats):.6f}]  "
          f"lon [{min(lons):.6f}, {max(lons):.6f}]")
    if drone_alts:
        print(f"  Mean drone altitude: {sum(drone_alts)/len(drone_alts):.1f} m")

    print("\nBuilding FAISS index...")
    db.build_index()

    feature_type = 'orb' if fast else 'superpoint'
    print(f"\nSaving database to {out_dir} ...")
    db.save(out_dir, feature_type=feature_type)

    return db


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build geo-referenced visual database from DJI flight data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--srt',          default=None, help='DJI SRT telemetry file')
    parser.add_argument('--video',        default=None, help='Companion MP4 video file')
    parser.add_argument('--out-dir',      default='db/', help='Output directory')
    parser.add_argument('--sample-every', type=int, default=15,
                        help='Process every Nth video frame')
    parser.add_argument('--device',       default='cpu', help='PyTorch device')
    parser.add_argument('--source',       default='video',
                        choices=['video', 'gis', 'both'],
                        help='Data source: video frames, GIS satellite tiles, or both')
    parser.add_argument('--gis-dir',      default='db_gis/',
                        help='Directory of satellite patches (from map_fetcher.py)')
    parser.add_argument('--fast', action=argparse.BooleanOptionalAction, default=True,
                        help='Use ORB (fast, CPU) instead of SuperPoint (default: True)')
    args = parser.parse_args()

    if args.source in ('video', 'both') and not (args.srt and args.video):
        parser.error('--srt and --video are required when --source is video or both')

    build_database(args.srt, args.video, args.out_dir,
                   sample_every=args.sample_every, device=args.device,
                   fast=args.fast, source=args.source, gis_dir=args.gis_dir)


if __name__ == '__main__':
    main()
