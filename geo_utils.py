"""
Geodetic utility functions for the real-time drone visual navigation system.

Coordinate conventions:
  - Latitude / longitude in decimal degrees (WGS-84)
  - Altitude in metres above WGS-84 ellipsoid
  - ECEF axes: X towards (0°N, 0°E), Z towards North Pole
"""

import math
import numpy as np

# WGS-84 ellipsoid constants
_WGS84_A = 6_378_137.0          # semi-major axis (m)
_WGS84_F = 1.0 / 298.257_223_563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2   # first eccentricity squared

# DJI Air 3 camera model (used for ground footprint)
_SENSOR_W_MM = 9.6
_FOCAL_MM = 8.8
_IMAGE_W = 1920
_IMAGE_H = 1080

_LAT_PER_M = 1.0 / 111_111.0


def _lon_per_m(lat_deg: float) -> float:
    return 1.0 / (111_111.0 * math.cos(math.radians(lat_deg)))


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two GPS points.

    Parameters
    ----------
    lat1, lon1 : float
        First point in decimal degrees.
    lat2, lon2 : float
        Second point in decimal degrees.

    Returns
    -------
    float
        Distance in metres.
    """
    R = 6_371_000.0  # Earth mean radius (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def ecef_from_gps(lat: float, lon: float, alt: float) -> np.ndarray:
    """Convert WGS-84 geodetic coordinates to ECEF Cartesian.

    Parameters
    ----------
    lat : float  Latitude in decimal degrees.
    lon : float  Longitude in decimal degrees.
    alt : float  Height above WGS-84 ellipsoid in metres.

    Returns
    -------
    np.ndarray
        Shape (3,) array [X, Y, Z] in metres.
    """
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    X = (N + alt) * cos_lat * cos_lon
    Y = (N + alt) * cos_lat * sin_lon
    Z = (N * (1.0 - _WGS84_E2) + alt) * sin_lat
    return np.array([X, Y, Z], dtype=np.float64)


def gps_from_ecef(x: float, y: float, z: float) -> tuple:
    """Convert ECEF Cartesian coordinates to WGS-84 geodetic (Bowring iteration).

    Parameters
    ----------
    x, y, z : float
        ECEF coordinates in metres.

    Returns
    -------
    tuple
        (lat_deg, lon_deg, alt_m)
    """
    lon = math.atan2(y, x)
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))  # initial estimate
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
        lat_new = math.atan2(z + _WGS84_E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    sin_lat = math.sin(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    alt = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else abs(z) / abs(sin_lat) - N * (1 - _WGS84_E2)
    return math.degrees(lat), math.degrees(lon), alt


def compute_ground_footprint(lat: float, lon: float, alt_m: float,
                             pitch_deg: float = -90.0, heading_deg: float = 0.0,
                             hfov_deg: float = 82.1,
                             image_w: int = 1920, image_h: int = 1080) -> dict:
    """Compute GPS coordinates of the camera ground footprint.

    Uses the DJI Air 3 camera model:
        GSD = alt * sensor_width_mm / (focal_length_mm * image_width)

    Parameters
    ----------
    lat, lon : float
        Drone GPS position in decimal degrees.
    alt_m : float
        Relative altitude in metres (AGL).
    pitch_deg : float
        Camera pitch angle in degrees (-90 = nadir / straight down).
    heading_deg : float
        Drone heading in degrees clockwise from North.
    hfov_deg : float
        Horizontal field of view in degrees (unused — GSD formula is used instead).
    image_w, image_h : int
        Image dimensions in pixels.

    Returns
    -------
    dict with keys:
        center_lat, center_lon : float  GPS of image centre on the ground.
        width_m, height_m      : float  Footprint dimensions in metres.
        gsd                    : float  Ground sample distance (m/px).
        corners                : list   [(lat, lon), ...] NW, NE, SE, SW.
    """
    if alt_m <= 0:
        alt_m = 0.1

    gsd = (alt_m * _SENSOR_W_MM) / (_FOCAL_MM * image_w)
    width_m = gsd * image_w
    height_m = gsd * image_h

    # Ground point directly below the camera (nadir offset when pitch != -90)
    cam_lat, cam_lon = compute_camera_center(lat, lon, alt_m, pitch_deg, heading_deg)

    half_lat = (height_m / 2.0) * _LAT_PER_M
    half_lon = (width_m / 2.0) * _lon_per_m(cam_lat)

    corners = [
        (cam_lat + half_lat, cam_lon - half_lon),  # NW
        (cam_lat + half_lat, cam_lon + half_lon),  # NE
        (cam_lat - half_lat, cam_lon + half_lon),  # SE
        (cam_lat - half_lat, cam_lon - half_lon),  # SW
    ]

    return {
        'center_lat':   cam_lat,
        'center_lon':   cam_lon,
        'width_m':      width_m,
        'height_m':     height_m,
        'gsd':          gsd,
        'gsd_m_per_px': gsd,  # alias used by matcher.py
        'corners':      corners,
    }


def compute_camera_center(lat: float, lon: float, alt_m: float,
                          pitch_deg: float, heading_deg: float) -> tuple:
    """GPS coordinate where the camera's optical axis intersects the ground.

    For nadir pointing (pitch == -90) this is simply the drone's position.
    Otherwise the ground point is offset in the heading direction by
    ``alt_m * tan(90 + pitch_deg)`` metres.

    Parameters
    ----------
    lat, lon : float
        Drone GPS position in decimal degrees.
    alt_m : float
        Relative altitude AGL in metres.
    pitch_deg : float
        Camera pitch in degrees (-90 = straight down, 0 = horizontal).
    heading_deg : float
        Drone heading in degrees clockwise from North.

    Returns
    -------
    tuple
        (camera_lat, camera_lon) in decimal degrees.
    """
    if abs(pitch_deg + 90.0) < 0.5:
        return lat, lon

    angle_from_nadir = math.radians(90.0 + pitch_deg)
    offset_m = alt_m * math.tan(angle_from_nadir)
    heading_r = math.radians(heading_deg)

    # North/East components of the ground offset
    offset_north = offset_m * math.cos(heading_r)
    offset_east = offset_m * math.sin(heading_r)

    new_lat = lat + offset_north * _LAT_PER_M
    new_lon = lon + offset_east * _lon_per_m(lat)
    return new_lat, new_lon
