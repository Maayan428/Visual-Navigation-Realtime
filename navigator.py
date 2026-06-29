"""
Real-time localization pipeline for GPS-denied drone navigation.

The RealTimeNavigator loads a pre-built GeoDatabase and processes video
frames one at a time, estimating GPS position through:
  1. SuperPoint feature extraction
  2. FAISS approximate nearest-neighbour retrieval
  3. LightGlue local feature matching
  4. RANSAC homography + GPS triangulation
"""

import os
import time

import cv2
import numpy as np
import pandas as pd
import simplekml
import matplotlib.pyplot as plt
from tqdm import tqdm

from feature_extractor import SuperPointExtractor, ORBExtractor, extract_global_descriptor
from geo_utils import haversine
from matcher import LightGlueMatcher, estimate_homography, gps_from_homography, orb_match
from retrieval import GeoDatabase
from srt_parser import parse_srt

_TOP_K_RETRIEVE = 10  # candidates from FAISS
_TOP_N_MATCH = 3      # run LightGlue on the top-N candidates


class RealTimeNavigator:
    """GPS-denied drone localizer using SuperPoint + LightGlue + FAISS.

    Parameters
    ----------
    db_path : str
        Path to the directory produced by ``preprocess.py`` / ``GeoDatabase.save()``.
    device : str
        PyTorch device (``'cpu'`` or ``'cuda'``).
    fast : bool
        If True (default), use ORB + BFMatcher for fine matching (fast, CPU).
        If False, use SuperPoint + LightGlue (accurate, GPU recommended).
        The FAISS coarse retrieval always uses the global 256-dim descriptor,
        regardless of mode.
    """

    def __init__(self, db_path: str, device: str = 'cpu', fast: bool = True,
                 min_inliers: int = 4):
        self.fast = fast
        self.min_inliers = min_inliers
        print(f"Loading database from {db_path}...")
        self.db = GeoDatabase.load(db_path)

        query_feat = 'orb' if fast else 'superpoint'
        db_feat    = getattr(self.db, 'feature_type', 'unknown')
        if db_feat != 'unknown' and db_feat != query_feat:
            raise ValueError(
                f"Feature type mismatch: DB was built with '{db_feat}' but "
                f"navigator is in '{query_feat}' mode.  "
                f"Rebuild the DB with --{'fast' if fast else 'no-fast'} "
                f"or switch mode."
            )

        if fast:
            print(f"Navigator mode: FAST (ORB + BFMatcher, min_inliers={min_inliers})")
            self.extractor = ORBExtractor()
            self._lg_matcher = None
        else:
            print(f"Navigator mode: ACCURATE (SuperPoint + LightGlue, min_inliers={min_inliers})")
            self.extractor = SuperPointExtractor(device=device)
            self._lg_matcher = LightGlueMatcher(device=device)

        self.device = device

        # Pre-compute mean altitude for fallback altitude weighting
        alts = [r.get('alt', 50.0) for r in self.db.records]
        self._mean_db_alt = float(np.mean(alts)) if alts else 50.0

        self._last_lat = None
        self._last_lon = None

    # ------------------------------------------------------------------
    # Single-frame localisation
    # ------------------------------------------------------------------

    def locate(self, frame: np.ndarray,
               query_alt: float | None = None) -> dict | None:
        """Localise a single query frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from the drone camera (any resolution).
        query_alt : float, optional
            Query altitude AGL in metres.  Used to down-rank database
            candidates that were captured at very different altitudes.
            If None, the mean database altitude is used.

        Returns
        -------
        dict or None
            On success:
              'est_lat'           : float  Estimated latitude (degrees)
              'est_lon'           : float  Estimated longitude (degrees)
              'confidence'        : float  In [0, 1]
              'num_inliers'       : int
              'num_matches'       : int
              'best_candidate_id' : int    Database frame index
              'processing_time_ms': float
            Returns None if localisation failed.
        """
        t0 = time.perf_counter()

        # 1. Extract features (ORB or SuperPoint depending on mode)
        feats = self.extractor.extract(frame)
        kp_q = feats['keypoints']
        desc_q = feats['descriptors']

        if len(kp_q) < 50:
            return None

        # 2. Global descriptor for FAISS retrieval (256-dim float32, both modes)
        global_desc = extract_global_descriptor(frame, self.extractor)

        # 3. FAISS top-K retrieval
        if hasattr(self, '_last_lat') and self._last_lat is not None:
            candidates = self.db.search_near(
                global_desc,
                center_lat=self._last_lat,
                center_lon=self._last_lon,
                radius_m=500.0,
                k=_TOP_K_RETRIEVE
            )
        else:
            candidates = self.db.search(global_desc, k=_TOP_K_RETRIEVE)
        if not candidates:
            return None

        # 4. Altitude-weighted re-ranking.
        # FAISS distances are ~0.001-0.002; use a tiny coefficient so altitude
        # acts only as a tiebreaker and never overrides visual similarity.
        q_alt = query_alt if query_alt is not None else self._mean_db_alt
        for c in candidates:
            alt_penalty = abs(q_alt - c.get('alt', self._mean_db_alt)) * 1e-5
            c['weighted_dist'] = c['distance'] + alt_penalty
        candidates.sort(key=lambda c: c['weighted_dist'])

        img_w = int(frame.shape[1])
        img_h = int(frame.shape[0])
        query_size = (img_w, img_h)

        valid_results = []  # all candidates that pass matching + geo filter
        geo_rejected = 0

        # 5–6. Fine matching + homography for top-N candidates
        for cand in candidates[:_TOP_N_MATCH]:
            db_idx = cand['db_index']
            kp_db = self.db.local_keypoints[db_idx]
            desc_db = self.db.local_descriptors[db_idx]

            db_w = cand.get('image_w', 1920)
            db_h = cand.get('image_h', 1080)
            db_size = (db_w, db_h)

            if self.fast:
                match_result = orb_match(kp_q, desc_q, kp_db, desc_db)
            else:
                match_result = self._lg_matcher.match(
                    kp_q, desc_q, kp_db, desc_db, query_size, db_size
                )

            if match_result['num_matches'] < max(4, self.min_inliers):
                continue

            H, mask, n_inliers = estimate_homography(
                match_result['matched_kp0'],
                match_result['matched_kp1'],
                min_inliers=self.min_inliers,
            )

            if H is None or n_inliers < self.min_inliers:
                continue

            # 7. GPS from homography
            # Pass query dims separately so GIS patches (640×640) work correctly.
            est_lat, est_lon, _ = gps_from_homography(H, cand,
                                                       image_w=db_w, image_h=db_h,
                                                       query_w=img_w, query_h=img_h)

            # Geographic sanity filter: estimate must be within 500 m of
            # the DB candidate's ground position.  Homographies from a
            # handful of inliers can project wildly even if RANSAC passed.
            ref_lat = cand.get('camera_lat', cand['lat'])
            ref_lon = cand.get('camera_lon', cand['lon'])
            dist_from_ref = haversine(ref_lat, ref_lon, est_lat, est_lon)
            if dist_from_ref > 200.0:
                geo_rejected += 1
                continue

            valid_results.append({
                'n_inliers':   n_inliers,
                'est_lat':     est_lat,
                'est_lon':     est_lon,
                'cand':        cand,
                'db_idx':      db_idx,
                'num_matches': match_result['num_matches'],
            })

        if not valid_results:
            return None

        # 8. Weighted position averaging when multiple candidates agree.
        if len(valid_results) == 1:
            vr = valid_results[0]
            est_lat  = vr['est_lat']
            est_lon  = vr['est_lon']
            best_inliers = vr['n_inliers']
            best_cand    = vr['cand']
            best_db_idx  = vr['db_idx']
            best_matches = vr['num_matches']
        else:
            lats = [r['est_lat'] for r in valid_results]
            lons = [r['est_lon'] for r in valid_results]
            # Pairwise spread
            max_spread = max(
                haversine(lats[i], lons[i], lats[j], lons[j])
                for i in range(len(lats))
                for j in range(i + 1, len(lats))
            )

            best_vr = max(valid_results, key=lambda r: r['n_inliers'])

            if max_spread <= 200.0:
                # Estimates agree — compute inlier-weighted average
                weights = np.array([r['n_inliers'] for r in valid_results],
                                   dtype=np.float64)
                weights /= weights.sum()
                est_lat = float(np.dot(weights, lats))
                est_lon = float(np.dot(weights, lons))
            else:
                # Estimates disagree — trust the one with most inliers
                est_lat = best_vr['est_lat']
                est_lon = best_vr['est_lon']

            best_inliers = best_vr['n_inliers']
            best_cand    = best_vr['cand']
            best_db_idx  = best_vr['db_idx']
            best_matches = best_vr['num_matches']

        confidence = min(1.0, best_inliers / 50.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._last_lat = est_lat
        self._last_lon = est_lon
        return {
            'est_lat':              est_lat,
            'est_lon':              est_lon,
            'confidence':           confidence,
            'num_inliers':          best_inliers,
            'num_matches':          best_matches,
            'best_candidate_id':    best_db_idx,
            'processing_time_ms':   elapsed_ms,
            'geo_rejected_count':   geo_rejected,
        }

    # ------------------------------------------------------------------
    # Video-stream processing
    # ------------------------------------------------------------------

    def locate_video_stream(self, video_path: str,
                            srt_path: str | None = None,
                            out_dir: str = 'out/') -> pd.DataFrame:
        """Process a video file as if it were a live stream.

        Parameters
        ----------
        video_path : str
            Path to the query MP4 video.
        srt_path : str, optional
            Path to the companion SRT for ground-truth comparison.
        out_dir : str
            Output directory for KML, CSV, and plots.

        Returns
        -------
        pd.DataFrame
            One row per processed frame with all result fields.
        """
        os.makedirs(out_dir, exist_ok=True)

        # Ground truth lookup: {frame_cnt: record}
        gt_map: dict = {}
        if srt_path:
            gt_frames = parse_srt(srt_path)
            gt_map = {f['frame_cnt']: f for f in gt_frames}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        frame_idx = 0

        pbar = tqdm(total=total, desc='Navigating')
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 30 != 0:
                pbar.update(1)
                continue
            pbar.update(1)

            # Nearest GT record for this frame
            gt = gt_map.get(frame_idx)
            query_alt = gt['rel_alt'] if gt else None

            result = self.locate(frame_bgr, query_alt=query_alt)

            # Pre-populate every column so DataFrame always has the same schema
            # regardless of whether locate() returned a result or None.
            row = {
                'frame_idx':          frame_idx,
                'gt_lat':             gt['lat'] if gt else None,
                'gt_lon':             gt['lon'] if gt else None,
                'gt_alt':             gt['rel_alt'] if gt else None,
                'est_lat':            None,
                'est_lon':            None,
                'confidence':         None,
                'num_inliers':        None,
                'num_matches':        None,
                'best_candidate_id':  None,
                'processing_time_ms': None,
                'error_m':            None,
                'geo_rejected_count': 0,
            }

            if result:
                row.update(result)
                if gt:
                    err = haversine(gt['lat'], gt['lon'], result['est_lat'], result['est_lon'])
                    row['error_m'] = err
                    print(f"  Frame {frame_idx:5d}  "
                          f"est ({result['est_lat']:.6f}, {result['est_lon']:.6f})  "
                          f"err={err:.1f}m  inliers={result['num_inliers']}  "
                          f"t={result['processing_time_ms']:.0f}ms")
                else:
                    print(f"  Frame {frame_idx:5d}  "
                          f"est ({result['est_lat']:.6f}, {result['est_lon']:.6f})  "
                          f"inliers={result['num_inliers']}  "
                          f"t={result['processing_time_ms']:.0f}ms")
            else:
                print(f"  Frame {frame_idx:5d}  no match")

            rows.append(row)

        pbar.close()
        cap.release()

        df = pd.DataFrame(rows)

        # Export results
        csv_path = os.path.join(out_dir, 'results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        self._export_kml(df, out_dir, with_gt=bool(gt_map))
        self._plot_performance(df, out_dir)

        return df

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export_kml(self, df: pd.DataFrame, out_dir: str, with_gt: bool) -> None:
        """Write estimated positions as Point Placemarks and ground-truth as LineString."""
        # Guard: columns may not exist if no frame was ever localized
        for col in ('est_lat', 'est_lon'):
            if col not in df.columns:
                df[col] = float('nan')

        kml_est = simplekml.Kml()
        est_rows = df.dropna(subset=['est_lat', 'est_lon']).sort_values('frame_idx')
        if len(est_rows) >= 2:
            ls = kml_est.newlinestring(name='Estimated path')
            ls.coords = [(row['est_lon'], row['est_lat'])
                         for _, row in est_rows.iterrows()]
            ls.style.linestyle.color = simplekml.Color.red
            ls.style.linestyle.width = 3
        for _, row in est_rows.iterrows():
            pnt = kml_est.newpoint(name=f'Frame {int(row["frame_idx"])}')
            pnt.coords = [(row['est_lon'], row['est_lat'])]
            err_str = f'{row["error_m"]:.1f} m' if pd.notna(row.get("error_m")) else 'N/A'
            pnt.description = (f'Frame: {int(row["frame_idx"])}\n'
                              f'Error: {err_str}\n'
                              f'Inliers: {int(row["num_inliers"]) if pd.notna(row.get("num_inliers")) else 0}')
            pnt.style.iconstyle.scale = 0.5
            pnt.style.iconstyle.color = simplekml.Color.red
        kml_est.save(os.path.join(out_dir, 'path_estimated.kml'))

        if with_gt:
            kml_gt = simplekml.Kml()
            ls_gt = kml_gt.newlinestring(name='Ground truth path')
            gt_rows = df.dropna(subset=['gt_lat', 'gt_lon'])
            ls_gt.coords = [(row['gt_lon'], row['gt_lat'])
                            for _, row in gt_rows.iterrows()]
            ls_gt.style.linestyle.color = simplekml.Color.blue
            ls_gt.style.linestyle.width = 3
            kml_gt.save(os.path.join(out_dir, 'path_groundtruth.kml'))

        print(f"  KML files saved to {out_dir}")

    def _plot_performance(self, df: pd.DataFrame, out_dir: str) -> None:
        """Generate error-over-time and histogram plots."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if 'error_m' not in df.columns:
            df['error_m'] = float('nan')
        err_series = df['error_m'].dropna()

        if not err_series.empty:
            axes[0].plot(df['frame_idx'], df['error_m'], color='steelblue', linewidth=0.8)
            axes[0].axhline(err_series.median(), color='red', linestyle='--',
                            label=f'Median {err_series.median():.1f} m')
            axes[0].set_xlabel('Frame')
            axes[0].set_ylabel('Error (m)')
            axes[0].set_title('Localisation error over time')
            axes[0].legend()

            axes[1].hist(err_series, bins=40, color='steelblue', edgecolor='white')
            axes[1].set_xlabel('Error (m)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Error distribution')

        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'performance_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Performance plot saved to {plot_path}")
