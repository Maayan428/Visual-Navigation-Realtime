"""
Diagnostic script for the visual navigation pipeline.

Loads the database, extracts the first frame of the query video,
and traces each step of the pipeline to find where matching fails.

Usage
-----
    python debug_db.py \
        --db db/ \
        --query-video ~/Documents/Final_Year/nav01/DJI_20260427152735_0019_D.MP4
"""

import argparse
import sys

import cv2
import numpy as np


def _sep(title: str = '') -> None:
    bar = '=' * 60
    if title:
        print(f"\n{bar}")
        print(f"  {title}")
        print(bar)
    else:
        print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Diagnose the visual navigation pipeline step by step.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--db',           required=True, help='GeoDatabase directory')
    parser.add_argument('--query-video',  required=True, help='Query MP4 file')
    parser.add_argument('--frame-number', type=int, default=30,
                        help='Which video frame to use for the test (1-based)')
    parser.add_argument('--top-k',        type=int, default=5,
                        help='Number of FAISS candidates to inspect')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1 – Load database                                              #
    # ------------------------------------------------------------------ #
    _sep('Step 1: Loading database')
    from retrieval import GeoDatabase
    from feature_extractor import ORBExtractor, extract_global_descriptor
    from matcher import orb_match, estimate_homography

    db = GeoDatabase.load(args.db)
    N = len(db.records)
    print(f"  Records in DB       : {N}")
    print(f"  Sample record [0]   : {db.records[0]}")
    kp_counts = [len(db.local_keypoints[i]) for i in range(N)]
    print(f"  Keypoints per frame : min={min(kp_counts)}  "
          f"max={max(kp_counts)}  mean={sum(kp_counts)/N:.0f}")
    desc_dtype = db.local_descriptors[0].dtype if N > 0 else '?'
    desc_shape = db.local_descriptors[0].shape if N > 0 else '?'
    print(f"  Descriptor dtype    : {desc_dtype}  shape per frame: {desc_shape}")

    # ------------------------------------------------------------------ #
    # Step 2 – Extract the requested frame from the query video           #
    # ------------------------------------------------------------------ #
    _sep(f'Step 2: Extracting frame {args.frame_number} from query video')
    cap = cv2.VideoCapture(args.query_video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.query_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames @ {fps:.1f} fps")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_number - 1)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        sys.exit(f"Could not read frame {args.frame_number}")

    print(f"  Frame shape: {frame_bgr.shape}  dtype: {frame_bgr.dtype}")

    # ------------------------------------------------------------------ #
    # Step 3 – ORB feature extraction on the query frame                  #
    # ------------------------------------------------------------------ #
    _sep('Step 3: ORB feature extraction on query frame')
    extractor = ORBExtractor()
    feats = extractor.extract(frame_bgr)
    kp_q    = feats['keypoints']
    desc_q  = feats['descriptors']
    scores_q = feats['scores']

    print(f"  Keypoints found     : {len(kp_q)}")
    if len(scores_q) > 0:
        print(f"  Score range         : [{scores_q.min():.1f}, {scores_q.max():.1f}]  "
              f"mean={scores_q.mean():.1f}")
    print(f"  Descriptor dtype    : {desc_q.dtype}  shape: {desc_q.shape}")

    if len(kp_q) < 50:
        print("  *** WARNING: < 50 keypoints — locate() will return None immediately! ***")
    if len(kp_q) == 0:
        print("  *** FATAL: 0 keypoints extracted. Check image content. ***")
        return

    # ------------------------------------------------------------------ #
    # Step 4 – Global descriptor + FAISS retrieval                        #
    # ------------------------------------------------------------------ #
    _sep(f'Step 4: Global descriptor + FAISS retrieval (top {args.top_k})')
    global_desc = extract_global_descriptor(frame_bgr, extractor)
    print(f"  Global descriptor   : norm={np.linalg.norm(global_desc):.6f}  "
          f"min={global_desc.min():.4f}  max={global_desc.max():.4f}")

    candidates = db.search(global_desc, k=args.top_k)
    if not candidates:
        print("  *** ERROR: FAISS returned no candidates! ***")
        return

    print(f"\n  {'#':>2}  {'db_idx':>6}  {'dist':>8}  {'lat':>12}  {'lon':>12}  "
          f"{'alt_m':>7}  {'frame_id':>8}")
    print(f"  {'-'*2}  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*8}")
    for i, c in enumerate(candidates):
        print(f"  {i+1:>2}  {c['db_index']:>6}  {c['distance']:>8.4f}  "
              f"{c['lat']:>12.6f}  {c['lon']:>12.6f}  "
              f"{c.get('alt', 0):>7.1f}  {c.get('frame_id', '?'):>8}")

    # ------------------------------------------------------------------ #
    # Step 5 – BFMatcher against each candidate                           #
    # ------------------------------------------------------------------ #
    _sep('Step 5: BFMatcher (NORM_HAMMING) statistics for each candidate')
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for i, c in enumerate(candidates):
        db_idx   = c['db_index']
        kp_db    = db.local_keypoints[db_idx]
        desc_db  = db.local_descriptors[db_idx]

        print(f"\n  Candidate {i+1}  (db_idx={db_idx}, "
              f"GPS=({c['lat']:.6f}, {c['lon']:.6f}), "
              f"{len(kp_db)} db keypoints)")

        if len(desc_db) == 0:
            print("    No descriptors in DB frame — skipping")
            continue

        # Cast to uint8: DB descriptors may have been saved as float32 (dtype bug
        # in old DBs); rounding back is safe because ORB values are integers in [0,255].
        try:
            raw = bf.knnMatch(desc_q.astype(np.uint8), desc_db.astype(np.uint8), k=2)
        except cv2.error as e:
            print(f"    knnMatch failed: {e}")
            continue

        print(f"    Raw knnMatch pairs returned : {len(raw)}")

        for thresh in (0.70, 0.75, 0.80, 0.85, 0.90):
            good = [pair for pair in raw
                    if len(pair) == 2 and pair[0].distance < thresh * pair[1].distance]
            print(f"    Good matches @ ratio {thresh} : {len(good)}")

        # Use 0.75 for RANSAC test
        good_075 = [pair for pair in raw
                    if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(good_075) >= 4:
            kp_q_pts  = np.float32([kp_q[m[0].queryIdx]  for m in good_075])
            kp_db_pts = np.float32([kp_db[m[0].trainIdx] for m in good_075])
            H, mask = cv2.findHomography(
                kp_q_pts.reshape(-1, 1, 2),
                kp_db_pts.reshape(-1, 1, 2),
                cv2.RANSAC, 5.0,
            )
            if H is not None and mask is not None:
                inliers = int(mask.sum())
                print(f"    RANSAC inliers (of {len(good_075)}) : {inliers}")
            else:
                print("    RANSAC: failed to find homography")
        else:
            print(f"    Too few good matches ({len(good_075)}) for RANSAC (need ≥ 4)")

    # ------------------------------------------------------------------ #
    # Step 6 – What does the navigator actually do?                       #
    # ------------------------------------------------------------------ #
    _sep('Step 6: Simulating navigator.locate() with current thresholds')
    print("  Current hard-coded thresholds: num_matches < 8 → skip, n_inliers < 8 → skip")
    print()
    passed = 0
    for i, c in enumerate(candidates[:3]):
        db_idx = c['db_index']
        kp_db  = db.local_keypoints[db_idx]
        desc_db = db.local_descriptors[db_idx]
        mr = orb_match(kp_q, desc_q, kp_db, desc_db)
        H, mask, n_inliers = estimate_homography(mr['matched_kp0'], mr['matched_kp1'])
        status = '✓ PASS' if (H is not None and n_inliers >= 8) else '✗ FAIL'
        print(f"  Candidate {i+1}: good={mr['num_matches']}  inliers={n_inliers}  → {status}")
        if H is not None and n_inliers >= 4:
            passed += 1
    print(f"\n  Would pass with min_inliers=4: {passed}/3 candidates")

    _sep()
    print("Diagnosis complete.")


if __name__ == '__main__':
    main()
