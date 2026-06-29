"""
Download satellite imagery tiles for the Ariel University area and build
a GIS patch database for offline visual navigation.

Patches are 640×640-pixel crops of Esri World Imagery at zoom-18 (~0.5 m/px),
arranged on an overlapping grid that covers the campus.

Usage
-----
    python map_fetcher.py \
        --center-lat 32.1035 --center-lon 35.2075 \
        --radius-m 1500 --zoom 18 \
        --out-dir db_gis/
"""

import argparse
import json
import math
import os
import time

import cv2
import numpy as np

# Esri World Imagery — free, no API key required
_ESRI_SATELLITE = (
    'https://server.arcgisonline.com/ArcGIS/rest/services/'
    'World_Imagery/MapServer/tile/{z}/{y}/{x}'
)

_PATCH_PX = 640
_GRID_N   = 8

# Web-Mercator: one tile at zoom z covers 2π·R / 2^z metres at equator,
# divided by 256 pixels/tile.
_GSD_EQUATOR_Z18 = 40_075_016.686 / (2**18 * 256)  # ~0.596 m/px at equator


def _gsd_at_lat(lat_deg: float, zoom: int = 18) -> float:
    """Web-Mercator GSD at the given latitude and zoom level (m/px)."""
    return _GSD_EQUATOR_Z18 * math.cos(math.radians(lat_deg))


def _build_grid(center_lat: float, center_lon: float,
                n: int = _GRID_N) -> list:
    """Return list of NxN patch centres with 50% overlap.

    Each dict has: row, col, lat, lon, gsd_m_per_px, width_m
    """
    gsd     = _gsd_at_lat(center_lat)
    patch_m = _PATCH_PX * gsd     # ground extent of one patch (m)
    step_m  = patch_m * 0.5       # 50 % overlap → step = half a patch

    lat_per_m = 1.0 / 111_111.0
    lon_per_m = 1.0 / (111_111.0 * math.cos(math.radians(center_lat)))

    half = (n - 1) / 2.0
    patches = []
    for i in range(n):
        for j in range(n):
            dy = (i - half) * step_m   # metres north of centre
            dx = (j - half) * step_m   # metres east of centre
            patches.append({
                'row':          i,
                'col':          j,
                'lat':          center_lat + dy * lat_per_m,
                'lon':          center_lon + dx * lon_per_m,
                'gsd_m_per_px': gsd,
                'width_m':      patch_m,
            })
    return patches


def _fetch_patch(lat: float, lon: float,
                 zoom: int, size: int) -> np.ndarray:
    """Download a satellite image centred at (lat, lon).

    Returns
    -------
    np.ndarray, shape (size, size), dtype uint8, grayscale
    """
    from staticmap import StaticMap

    m   = StaticMap(size, size, url_template=_ESRI_SATELLITE)
    img = m.render(zoom=zoom, center=[lon, lat])   # PIL Image

    rgb = np.array(img, dtype=np.uint8)
    if rgb.ndim == 2:
        return rgb                                  # already grayscale
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]                        # drop alpha
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def fetch_all_patches(center_lat: float, center_lon: float,
                      zoom: int = 18, out_dir: str = 'db_gis/',
                      n: int = _GRID_N) -> None:
    """Download all patches and save metadata.

    Directory layout
    ----------------
    out_dir/
      patches/patch_<row>_<col>.npy   – grayscale uint8, shape (640, 640)
      metadata.json                   – list of patch dicts
    """
    os.makedirs(out_dir, exist_ok=True)
    patches_dir = os.path.join(out_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)

    grid = _build_grid(center_lat, center_lon, n=n)
    gsd  = _gsd_at_lat(center_lat, zoom)
    print(f"Grid: {n}×{n} = {len(grid)} patches  "
          f"zoom={zoom}  GSD≈{gsd:.3f} m/px  "
          f"patch≈{_PATCH_PX * gsd:.0f}×{_PATCH_PX * gsd:.0f} m")

    metadata = []
    failed   = 0

    for idx, patch in enumerate(grid):
        filename = f'patch_{patch["row"]}_{patch["col"]}.npy'
        npy_path = os.path.join(patches_dir, filename)

        if os.path.exists(npy_path):
            print(f'  [{idx+1:3d}/{len(grid)}] skip  '
                  f'({patch["lat"]:.5f}, {patch["lon"]:.5f})  cached')
            patch['filename'] = filename
            metadata.append(patch)
            continue

        try:
            gray = _fetch_patch(patch['lat'], patch['lon'],
                                zoom=zoom, size=_PATCH_PX)
            np.save(npy_path, gray)
            patch['filename'] = filename
            metadata.append(patch)
            print(f'  [{idx+1:3d}/{len(grid)}] ok    '
                  f'({patch["lat"]:.5f}, {patch["lon"]:.5f})  {gray.shape}')
        except Exception as exc:
            print(f'  [{idx+1:3d}/{len(grid)}] ERROR '
                  f'({patch["lat"]:.5f}, {patch["lon"]:.5f}): {exc}')
            failed += 1

        time.sleep(0.25)    # polite rate-limit to avoid tile-server bans

    meta_path = os.path.join(out_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    ok = len(grid) - failed
    print(f"\nDone: {ok}/{len(grid)} patches saved → {out_dir}")
    if failed:
        print(f"  {failed} failures — re-run to retry (cached patches are skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download satellite tiles for GIS-based visual navigation database.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--center-lat', type=float, required=True)
    parser.add_argument('--center-lon', type=float, required=True)
    parser.add_argument('--radius-m',   type=float, default=1500.0,
                        help='Coverage radius (for reference only; grid is fixed N×N)')
    parser.add_argument('--zoom',       type=int,   default=18)
    parser.add_argument('--grid-n',     type=int,   default=_GRID_N,
                        help='Patches per side (total = N²)')
    parser.add_argument('--out-dir',    default='db_gis/')
    args = parser.parse_args()

    fetch_all_patches(
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        zoom=args.zoom,
        out_dir=args.out_dir,
        n=args.grid_n,
    )


if __name__ == '__main__':
    main()
