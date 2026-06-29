"""
Feature extractors for the real-time visual navigation system.

Two modes:
  SuperPointExtractor – high-accuracy learned features (requires LightGlue / GPU)
  ORBExtractor        – fast CPU-only classical features for the --fast pipeline
"""

import numpy as np
import torch
import cv2

from lightglue import SuperPoint
from lightglue.utils import rbd

_MAX_SIDE = 640  # longest image dimension passed to SuperPoint

# Fixed random projection matrix: maps 32-byte ORB mean descriptor → 256-dim float.
# Seeded so that the same projection is used at build time and query time.
_ORB_PROJ = np.random.RandomState(42).randn(256, 32).astype(np.float32)


class SuperPointExtractor:
    """Wrapper around LightGlue's SuperPoint for keypoint extraction.

    Parameters
    ----------
    device : str
        PyTorch device string, e.g. ``'cpu'`` or ``'cuda'``.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = SuperPoint(max_num_keypoints=2048).eval().to(self.device)

    def extract(self, image: np.ndarray) -> dict:
        """Extract SuperPoint keypoints and descriptors from an image.

        Parameters
        ----------
        image : np.ndarray
            Grayscale (H, W) or BGR (H, W, 3) image as a uint8 numpy array.

        Returns
        -------
        dict with keys:
            'keypoints'   : np.ndarray, shape (N, 2), float32  (x, y) pixel coords
            'descriptors' : np.ndarray, shape (N, 256), float32
            'scores'      : np.ndarray, shape (N,), float32
        Returns empty arrays of correct shape if fewer than 1 keypoint is found.
        """
        _empty = {
            'keypoints':   np.zeros((0, 2), dtype=np.float32),
            'descriptors': np.zeros((0, 256), dtype=np.float32),
            'scores':      np.zeros((0,), dtype=np.float32),
        }

        if image is None or image.size == 0:
            return _empty

        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize so longest side ≤ _MAX_SIDE
        h, w = gray.shape[:2]
        scale = min(1.0, _MAX_SIDE / max(h, w))
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Build (1, 1, H, W) float32 tensor in [0, 1]
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)

        with torch.no_grad():
            feats = rbd(self.model.extract(tensor))  # remove batch dim

        kps = feats.get('keypoints')
        descs = feats.get('descriptors')
        scores = feats.get('keypoint_scores')

        if kps is None or len(kps) == 0:
            return _empty

        # Scale keypoints back to original image coordinates
        kps_np = kps.cpu().numpy().astype(np.float32)
        if scale < 1.0:
            kps_np = kps_np / scale

        descs_np = descs.cpu().numpy().astype(np.float32)
        scores_np = scores.cpu().numpy().astype(np.float32)

        return {
            'keypoints':   kps_np,
            'descriptors': descs_np,
            'scores':      scores_np,
        }


def extract_global_descriptor(image: np.ndarray, extractor) -> np.ndarray:
    """Compute a 256-dim unit-normalised global descriptor for FAISS retrieval.

    Dispatches to the extractor-specific implementation so that both
    SuperPointExtractor and ORBExtractor produce compatible 256-dim vectors.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale image.
    extractor : SuperPointExtractor | ORBExtractor
        Initialised extractor instance.

    Returns
    -------
    np.ndarray
        Shape (256,) float32, L2-normalised.
    """
    if isinstance(extractor, ORBExtractor):
        return extractor.extract_global_descriptor(image)

    # SuperPoint path: mean-pool the 256-dim float descriptors
    feats = extractor.extract(image)
    descs = feats['descriptors']  # (N, 256)

    if descs.shape[0] == 0:
        return np.zeros(256, dtype=np.float32)

    global_desc = descs.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(global_desc)
    if norm > 1e-6:
        global_desc /= norm
    return global_desc


# -----------------------------------------------------------------------
# ORB fast extractor
# -----------------------------------------------------------------------

class ORBExtractor:
    """CPU-only ORB feature extractor for the fast preprocessing pipeline.

    Produces keypoints and 32-byte binary descriptors (uint8) that are
    ~10-50× faster to extract than SuperPoint on CPU.  A fixed random
    projection maps the mean binary descriptor to a 256-dim float32 vector
    so that the same FAISS index dimension is used regardless of mode.
    """

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)

    def extract(self, image: np.ndarray) -> dict:
        """Detect ORB keypoints and compute descriptors.

        Parameters
        ----------
        image : np.ndarray
            Grayscale (H, W) or BGR (H, W, 3) uint8 image.

        Returns
        -------
        dict with keys:
            'keypoints'   : np.ndarray shape (N, 2) float32  (x, y) pixel coords
            'descriptors' : np.ndarray shape (N, 32) uint8   binary ORB descriptors
            'scores'      : np.ndarray shape (N,) float32    keypoint response scores
        Returns empty arrays if no keypoints are found.
        """
        _empty = {
            'keypoints':   np.zeros((0, 2), dtype=np.float32),
            'descriptors': np.zeros((0, 32), dtype=np.uint8),
            'scores':      np.zeros((0,), dtype=np.float32),
        }

        if image is None or image.size == 0:
            return _empty

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kps, descs = self.orb.detectAndCompute(gray, None)

        if kps is None or descs is None or len(kps) == 0:
            return _empty

        coords = np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)
        scores = np.array([k.response for k in kps], dtype=np.float32)
        return {
            'keypoints':   coords,
            'descriptors': descs,
            'scores':      scores,
        }

    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Compute a 256-dim global descriptor via random projection of ORB.

        Each ORB descriptor (32 bytes, uint8) is cast to float32, mean-pooled
        across all keypoints to a (32,) vector, then projected with the fixed
        random matrix ``_ORB_PROJ`` (shape 256×32) and L2-normalised.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale image.

        Returns
        -------
        np.ndarray
            Shape (256,) float32, L2-normalised.  Zero vector on failure.
        """
        feats = self.extract(image)
        descs = feats['descriptors']  # (N, 32) uint8

        if descs.shape[0] == 0:
            return np.zeros(256, dtype=np.float32)

        mean_desc = descs.astype(np.float32).mean(axis=0)  # (32,)
        global_desc = _ORB_PROJ @ mean_desc                # (256,)
        norm = np.linalg.norm(global_desc)
        if norm > 1e-6:
            global_desc /= norm
        return global_desc
