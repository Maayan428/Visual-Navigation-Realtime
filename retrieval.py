"""
FAISS-based geo-referenced visual database for drone localization.

Each database entry stores:
  - frame metadata (GPS, altitude, GSD, source video)
  - a global descriptor (256-dim float32) for fast approximate retrieval
  - local keypoints and descriptors for precise LightGlue matching
"""

import json
import os
import numpy as np
import faiss


class GeoDatabase:
    """Geo-referenced visual feature database with FAISS indexing.

    Build the database offline with ``add_frame`` + ``build_index``,
    then save to disk with ``save``.  At runtime, load with
    ``GeoDatabase.load`` and call ``search`` to retrieve candidates.
    """

    def __init__(self):
        self.records: list = []                    # list of frame metadata dicts
        self.global_descriptors: list = []         # list of (256,) float32 arrays
        self.local_keypoints: list = []            # list of (N, 2) float32 arrays
        self.local_descriptors: list = []          # list of (N, 256) float32 arrays
        self.index = None                          # FAISS index (built by build_index)
        self._desc_matrix: np.ndarray | None = None
        self.feature_type: str = 'unknown'         # 'orb' or 'superpoint'

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_frame(self, frame_meta: dict, global_desc: np.ndarray,
                  keypoints: np.ndarray, descriptors: np.ndarray) -> None:
        """Add one reference frame to the database.

        Parameters
        ----------
        frame_meta : dict
            Must include: frame_id, lat, lon, alt, timestamp,
            camera_lat, camera_lon, source_video.
        global_desc : np.ndarray, shape (256,), float32
            Mean-pooled SuperPoint descriptor for FAISS retrieval.
        keypoints : np.ndarray, shape (N, 2), float32
            Local SuperPoint keypoints (x, y) in original pixel coordinates.
        descriptors : np.ndarray, shape (N, 256), float32
            Per-keypoint SuperPoint descriptors.
        """
        self.records.append(frame_meta)
        self.global_descriptors.append(global_desc.astype(np.float32))
        self.local_keypoints.append(keypoints.astype(np.float32))
        # Preserve original dtype: uint8 for ORB (required by BFMatcher NORM_HAMMING),
        # float32 for SuperPoint.
        self.local_descriptors.append(descriptors.copy())

    def build_index(self) -> None:
        """Build a FAISS index over the global descriptors.

        Uses ``IndexFlatL2`` for N ≤ 1000 frames; otherwise switches to
        ``IndexIVFFlat`` with ``nlist = min(100, N // 10)`` for faster
        approximate search.
        """
        N = len(self.records)
        if N == 0:
            raise ValueError("Database is empty; add frames before building the index.")

        self._desc_matrix = np.vstack(self.global_descriptors).astype(np.float32)
        # Normalise to unit length so L2 distance ~ cosine distance
        faiss.normalize_L2(self._desc_matrix)
        dim = self._desc_matrix.shape[1]

        if N > 1000:
            nlist = min(100, N // 10)
            quantiser = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_L2)
            self.index.train(self._desc_matrix)
            self.index.add(self._desc_matrix)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self._desc_matrix)

        print(f"  FAISS index built: {N} frames, dim={dim}, "
              f"type={'IVFFlat' if N > 1000 else 'FlatL2'}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query_desc: np.ndarray, k: int = 10) -> list:
        """Retrieve the top-k most similar database frames.

        Parameters
        ----------
        query_desc : np.ndarray, shape (256,), float32
            Global descriptor of the query frame.
        k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list of dict
            Each dict is a copy of the matching record plus:
              'db_index'  : int   – position in self.records
              'distance'  : float – FAISS L2 distance (lower = more similar)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = min(k, len(self.records))

        q = query_desc.astype(np.float32).copy().reshape(1, -1)
        faiss.normalize_L2(q)

        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = 10

        distances, indices = self.index.search(q, k)
        distances = distances[0]
        indices = indices[0]

        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0:
                continue
            rec = dict(self.records[idx])
            rec['db_index'] = int(idx)
            rec['distance'] = float(dist)
            results.append(rec)
        return results

    def search_near(self, query_desc: np.ndarray,
                    center_lat: float, center_lon: float,
                    radius_m: float = 1000.0, k: int = 10) -> list:
        """Like search() but restricts candidates to frames within radius_m.

        Parameters
        ----------
        query_desc : np.ndarray, shape (256,)
            Global descriptor of the query frame.
        center_lat, center_lon : float
            Geographic center of the search region (degrees).
        radius_m : float
            Radius of the search region in metres.
        k : int
            Maximum number of candidates to return.

        Returns
        -------
        list of dict
            Same format as search().  Falls back to full search() if no DB
            frames exist within the radius.
        """
        from geo_utils import haversine

        valid_indices = [
            i for i, r in enumerate(self.records)
            if haversine(center_lat, center_lon,
                         r.get('lat', 0.0), r.get('lon', 0.0)) <= radius_m
        ]

        if not valid_indices:
            return self.search(query_desc, k=k)

        sub_matrix = self._desc_matrix[valid_indices].copy()  # (M, 256), already normalised
        q = query_desc.astype(np.float32).copy().reshape(1, -1)
        faiss.normalize_L2(q)

        temp_index = faiss.IndexFlatL2(sub_matrix.shape[1])
        temp_index.add(sub_matrix)

        actual_k = min(k, len(valid_indices))
        distances, local_indices = temp_index.search(q, actual_k)

        results = []
        for dist, local_idx in zip(distances[0], local_indices[0]):
            if local_idx < 0:
                continue
            global_idx = valid_indices[int(local_idx)]
            rec = dict(self.records[global_idx])
            rec['db_index'] = global_idx
            rec['distance'] = float(dist)
            results.append(rec)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, feature_type: str = 'unknown') -> None:
        """Save the database to disk.

        Creates the directory if necessary. Files written:
          records.json
          meta.json              – feature type and frame count
          global_descriptors.npy
          local_keypoints_<i>.npy   (one per frame)
          local_descriptors_<i>.npy (one per frame)
          index.faiss
        """
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'records.json'), 'w') as f:
            json.dump(self.records, f, indent=2)

        meta = {'feature_type': feature_type, 'num_frames': len(self.records)}
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        np.save(os.path.join(path, 'global_descriptors.npy'), self._desc_matrix)

        kp_dir = os.path.join(path, 'keypoints')
        os.makedirs(kp_dir, exist_ok=True)
        for i, (kp, desc) in enumerate(zip(self.local_keypoints, self.local_descriptors)):
            np.save(os.path.join(kp_dir, f'kp_{i}.npy'), kp)
            np.save(os.path.join(kp_dir, f'desc_{i}.npy'), desc)

        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        print(f"  Database saved to {path}  ({len(self.records)} frames, {feature_type})")

    @classmethod
    def load(cls, path: str) -> 'GeoDatabase':
        """Load a previously saved GeoDatabase from disk.

        Parameters
        ----------
        path : str
            Directory that was passed to ``save()``.

        Returns
        -------
        GeoDatabase
            Fully initialised instance ready for ``search()``.
        """
        db = cls()

        with open(os.path.join(path, 'records.json')) as f:
            db.records = json.load(f)

        db._desc_matrix = np.load(os.path.join(path, 'global_descriptors.npy'))
        db.global_descriptors = list(db._desc_matrix)

        kp_dir = os.path.join(path, 'keypoints')
        for i in range(len(db.records)):
            kp = np.load(os.path.join(kp_dir, f'kp_{i}.npy'))
            desc = np.load(os.path.join(kp_dir, f'desc_{i}.npy'))
            db.local_keypoints.append(kp)
            db.local_descriptors.append(desc)

        db.index = faiss.read_index(os.path.join(path, 'index.faiss'))

        meta_path = os.path.join(path, 'meta.json')
        db.feature_type = 'unknown'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            db.feature_type = meta.get('feature_type', 'unknown')

        print(f"  Database loaded from {path}  "
              f"({len(db.records)} frames, {db.feature_type})")
        return db
