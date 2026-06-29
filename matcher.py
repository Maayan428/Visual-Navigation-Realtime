"""
Feature matching, homography estimation, and GPS localisation.

Contains:
  LightGlueMatcher  – accurate learned matching (requires LightGlue / GPU)
  orb_match()       – fast BFMatcher (NORM_HAMMING) for binary ORB descriptors
  estimate_homography() – RANSAC homography from matched pairs
  gps_from_homography() – map query image centre to GPS via homography + GSD
"""

import math
import numpy as np
import torch
import cv2

from lightglue import LightGlue
from lightglue.utils import rbd


class LightGlueMatcher:
    """LightGlue matcher configured for SuperPoint features.

    Parameters
    ----------
    device : str
        PyTorch device string (``'cpu'`` or ``'cuda'``).
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def match(self, kp0: np.ndarray, desc0: np.ndarray,
              kp1: np.ndarray, desc1: np.ndarray,
              image_size0: tuple, image_size1: tuple) -> dict:
        """Run LightGlue matching between two sets of SuperPoint features.

        Keypoints are in original pixel coordinates; they are normalised
        internally before passing to LightGlue.

        Parameters
        ----------
        kp0, kp1 : np.ndarray, shape (N, 2)
            Keypoint (x, y) coordinates in pixels for image 0 and 1.
        desc0, desc1 : np.ndarray, shape (N, 256)
            Per-keypoint SuperPoint descriptors.
        image_size0, image_size1 : tuple (width, height)
            Original image dimensions used for keypoint normalisation.

        Returns
        -------
        dict with keys:
            'matches0'         : np.ndarray (K,)  – indices into kp0
            'matching_scores0' : np.ndarray (K,)  – match confidence scores
            'matched_kp0'      : np.ndarray (K, 2) – matched coords in image 0
            'matched_kp1'      : np.ndarray (K, 2) – matched coords in image 1
            'num_matches'      : int
        """
        _empty = {
            'matches0': np.array([], dtype=np.int64),
            'matching_scores0': np.array([], dtype=np.float32),
            'matched_kp0': np.zeros((0, 2), dtype=np.float32),
            'matched_kp1': np.zeros((0, 2), dtype=np.float32),
            'num_matches': 0,
        }

        if kp0.shape[0] == 0 or kp1.shape[0] == 0:
            return _empty

        def _to_tensor(kp, desc, img_size):
            w, h = img_size
            # Pass raw pixel coords; LightGlue normalises internally via image_size
            # using: (kp - size/2) / (max(size)/2).  Pre-normalising here would
            # cause double-normalisation and produce incorrect matches.
            return {
                'keypoints':   torch.from_numpy(kp.astype(np.float32)).unsqueeze(0).to(self.device),
                'descriptors': torch.from_numpy(desc.astype(np.float32)).unsqueeze(0).to(self.device),
                'image_size':  torch.tensor([[w, h]], dtype=torch.float32).to(self.device),
            }

        feats0 = _to_tensor(kp0, desc0, image_size0)
        feats1 = _to_tensor(kp1, desc1, image_size1)

        with torch.no_grad():
            result = rbd(self.matcher({'image0': feats0, 'image1': feats1}))

        matches = result.get('matches', None)
        if matches is None or len(matches) == 0:
            return _empty

        matches_np = matches.cpu().numpy()  # (K, 2)
        scores = result.get('scores', torch.ones(len(matches_np))).cpu().numpy()

        idx0 = matches_np[:, 0]
        idx1 = matches_np[:, 1]

        return {
            'matches0':         idx0,
            'matching_scores0': scores.astype(np.float32),
            'matched_kp0':      kp0[idx0].astype(np.float32),
            'matched_kp1':      kp1[idx1].astype(np.float32),
            'num_matches':      len(idx0),
        }


def orb_match(kp0: np.ndarray, desc0: np.ndarray,
              kp1: np.ndarray, desc1: np.ndarray,
              ratio_thresh: float = 0.75) -> dict:
    """Fast BFMatcher matching for binary ORB descriptors.

    Uses Hamming distance with Lowe's ratio test.  Much faster than
    LightGlue on CPU; suitable for the ``--fast`` pipeline.

    Parameters
    ----------
    kp0, kp1 : np.ndarray, shape (N, 2)  float32
        Keypoint (x, y) pixel coordinates for the two frames.
    desc0, desc1 : np.ndarray, shape (N, 32)  uint8
        Binary ORB descriptors.
    ratio_thresh : float
        Lowe's ratio test threshold (default 0.75).

    Returns
    -------
    dict with keys:
        'matches0'         : np.ndarray (K,)  indices into kp0
        'matching_scores0' : np.ndarray (K,)  1 - distance/256 as proxy score
        'matched_kp0'      : np.ndarray (K, 2)
        'matched_kp1'      : np.ndarray (K, 2)
        'num_matches'      : int
    """
    _empty = {
        'matches0':         np.array([], dtype=np.int64),
        'matching_scores0': np.array([], dtype=np.float32),
        'matched_kp0':      np.zeros((0, 2), dtype=np.float32),
        'matched_kp1':      np.zeros((0, 2), dtype=np.float32),
        'num_matches':      0,
    }

    if desc0 is None or desc1 is None or len(desc0) == 0 or len(desc1) == 0:
        return _empty

    # BFMatcher(NORM_HAMMING) requires uint8.  DB descriptors may have been
    # stored as float32 (old DB built before the dtype-preservation fix);
    # rounding back to uint8 is safe because ORB bit-patterns are integers in [0,255].
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw = bf.knnMatch(desc0.astype(np.uint8), desc1.astype(np.uint8), k=2)
    except cv2.error:
        return _empty

    good_q, good_db, scores = [], [], []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_q.append(m.queryIdx)
            good_db.append(m.trainIdx)
            scores.append(1.0 - m.distance / 256.0)

    if not good_q:
        return _empty

    idx0 = np.array(good_q, dtype=np.int64)
    idx1 = np.array(good_db, dtype=np.int64)
    return {
        'matches0':         idx0,
        'matching_scores0': np.array(scores, dtype=np.float32),
        'matched_kp0':      kp0[idx0].astype(np.float32),
        'matched_kp1':      kp1[idx1].astype(np.float32),
        'num_matches':      len(idx0),
    }


def estimate_homography(matched_kp_query: np.ndarray,
                        matched_kp_db: np.ndarray,
                        min_inliers: int = 8,
                        reproj_threshold: float = 3.0) -> tuple:
    """Estimate a homography from matched keypoint pairs using RANSAC.

    Parameters
    ----------
    matched_kp_query : np.ndarray, shape (K, 2)
        Matched keypoints in the query image (x, y pixels).
    matched_kp_db : np.ndarray, shape (K, 2)
        Corresponding matched keypoints in the database frame (x, y pixels).
    min_inliers : int
        Minimum number of RANSAC inliers required to accept the result.

    Returns
    -------
    tuple : (H, mask, num_inliers)
        H          : 3×3 homography matrix (or None)
        mask       : inlier boolean array (or None)
        num_inliers: int (0 if failed)
    """
    if len(matched_kp_query) < 4:
        return None, None, 0

    H, mask = cv2.findHomography(
        matched_kp_query.reshape(-1, 1, 2),
        matched_kp_db.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
    )

    if H is None or mask is None:
        return None, None, 0

    num_inliers = int(mask.sum())
    if num_inliers < min_inliers:
        return None, None, 0

    # Reject degenerate homographies: det(H) far from 1 means severe
    # scale distortion or flip — not physically plausible for aerial views.
    det = float(np.linalg.det(H))
    if not (0.1 <= det <= 10.0):
        return None, None, 0

    return H, mask, num_inliers


def gps_from_homography(H: np.ndarray, db_record: dict,
                        image_w: int = 1920, image_h: int = 1080,
                        query_w: int | None = None,
                        query_h: int | None = None) -> tuple:
    """Estimate query GPS by mapping the query image centre through a homography.

    Algorithm
    ---------
    1. Map the QUERY image centre through H → pixel in db frame.
    2. Offset from db frame centre in db pixels → metres via db GSD.
    3. Apply offset (east/south) to the db frame's ground GPS.

    Parameters
    ----------
    H : np.ndarray, shape (3, 3)
        Homography mapping query pixels → db frame pixels.
    db_record : dict
        Database frame metadata ('camera_lat', 'camera_lon', 'gsd_m_per_px').
    image_w, image_h : int
        DB frame dimensions (used for the centre-offset reference).
    query_w, query_h : int or None
        Query frame dimensions (used as the input point to transform through H).
        If None, falls back to image_w / image_h (correct for same-resolution pairs).

    Returns
    -------
    tuple : (est_lat, est_lon, confidence)
    """
    # The point we transform through H must be the QUERY image centre.
    # The reference for computing the DB-pixel offset must be the DB centre.
    # For same-resolution pairs (video↔video) these are identical; for
    # mixed-resolution pairs (video↔GIS 640×640) they differ.
    qw = query_w if query_w is not None else image_w
    qh = query_h if query_h is not None else image_h
    qcx, qcy   = qw / 2.0, qh / 2.0       # query centre → input to H
    dbcx, dbcy = image_w / 2.0, image_h / 2.0  # db centre → offset reference

    pt     = np.array([[[qcx, qcy]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)[0][0]
    px, py = float(mapped[0]), float(mapped[1])

    gsd        = db_record.get('gsd_m_per_px', 0.3)
    offset_x_m = (px - dbcx) * gsd   # positive = east
    offset_y_m = (py - dbcy) * gsd   # positive = south (image y-axis is down)

    base_lat = db_record.get('camera_lat') or db_record.get('lat')
    base_lon = db_record.get('camera_lon') or db_record.get('lon')

    est_lat = base_lat - offset_y_m / 111_111.0
    est_lon = base_lon + offset_x_m / (111_111.0 * math.cos(math.radians(base_lat)))

    dist_from_center = math.sqrt((px - dbcx) ** 2 + (py - dbcy) ** 2)
    max_dist         = math.sqrt(dbcx ** 2 + dbcy ** 2)
    confidence       = max(0.0, 1.0 - dist_from_center / max_dist)

    return est_lat, est_lon, confidence
