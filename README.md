# Real-Time Visual Navigation for Drones (GNSS-Denied)

GPS-free drone localization using SuperPoint feature extraction, FAISS-based retrieval, and LightGlue matching against a pre-built geo-referenced visual database.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE PHASE                            │
│                                                                 │
│  Reference flight                                               │
│  (DJI MP4 + SRT)  →  SuperPoint  →  FAISS index  →  GeoDatabase│
│                        features       (global)      (on disk)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                         db/ directory
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        ONLINE PHASE                             │
│                                                                 │
│  Live video frame                                               │
│       ↓                                                         │
│  SuperPoint  →  FAISS top-10  →  LightGlue  →  RANSAC  →  GPS  │
│  (features)      (retrieval)     (matching)   (homography)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Algorithm Description

### Offline Phase (preprocess.py)

1. **SRT parsing** – read DJI telemetry (GPS, altitude) for every video frame.
2. **Video sampling** – extract one frame every N video frames (default N=15, ≈ 2 fps from 30 fps footage).
3. **SuperPoint extraction** – detect up to 2048 keypoints per frame and compute 256-dim descriptors.
4. **Ground footprint** – compute GSD and GPS bounding box of each frame's ground coverage.
5. **FAISS indexing** – mean-pool descriptors into a global 256-dim vector; build a flat L2 (or IVFFlat for > 1000 frames) index for fast retrieval.
6. **GeoDatabase** – serialise records, per-frame keypoints/descriptors, and the FAISS index to disk.

### Online Phase (navigator.py)

1. **Feature extraction** – run SuperPoint on the incoming frame (resized to max 640 px).
2. **Global descriptor** – mean-pool to get a 256-dim retrieval vector.
3. **FAISS retrieval** – find the top-10 most similar database frames.
4. **Altitude re-ranking** – penalise candidates whose altitude differs significantly from the query altitude.
5. **LightGlue matching** – run full attention-based matching against the top-3 candidates.
6. **RANSAC homography** – estimate the planar homography from matched keypoint pairs (minimum 8 inliers).
7. **GPS estimation** – map the query image centre through the homography, convert pixel offset to metres using GSD, add to the database frame's GPS.

---

## Installation

```bash
# Clone and enter the project
cd visual_navigation_realtime

# Install all dependencies (requires git for LightGlue)
pip install -r requirements.txt
```

> **Note:** LightGlue is installed directly from GitHub. You need `git` and internet access.
> On CPU-only machines the system works but is slower (≈ 200–500 ms per frame pair).

---

## Usage

### 1. Build the geo-referenced database (offline, run once)

```bash
python main.py preprocess \
    --srt /path/to/DJI_0017.SRT \
    --video "/path/to/DJI 0017.MP4" \
    --out-dir db/ \
    --sample-every 15
```

### 2. Real-time navigation on a query video

```bash
python main.py navigate \
    --db db/ \
    --video "/path/to/DJI 0007.mp4" \
    --srt /path/to/DJI_0007.SRT \
    --out-dir out/
```

The `--srt` flag is optional; supply it to print per-frame ground-truth error.

### 3. Formal evaluation experiment

```bash
python main.py experiment \
    --db db/ \
    --query-srt /path/to/DJI_0019.SRT \
    --query-video /path/to/DJI_0019.MP4 \
    --out-dir out/
```

---

## Expected Performance

| Metric                    | Target   |
|---------------------------|----------|
| Median localization error | < 50 m   |
| Processing time per frame | < 100 ms (GPU) / ~500 ms (CPU) |
| Localization rate         | > 90 %   |

---

## Output Files

| File                      | Description                               |
|---------------------------|-------------------------------------------|
| `out/results.csv`         | Per-frame estimates and errors            |
| `out/path_estimated.kml`  | Estimated flight path (open in Google Earth) |
| `out/path_groundtruth.kml`| Ground-truth GPS path                    |
| `out/performance_plot.png`| Error over time + processing times        |
| `out/error_histogram.png` | Distribution of localization errors       |
| `out/error_over_time.png` | Error as a function of frame index        |

---

## Known Limitations

1. **Viewpoint change** – the system assumes near-nadir camera orientation. Significant pitch/roll reduces matching quality.
2. **Altitude mismatch** – if the query flight altitude differs greatly from the database, GSD-based GPS correction will be less accurate.
3. **Repetitive textures** – agricultural fields or large uniform surfaces can fool global retrieval.
4. **CPU speed** – LightGlue runs at ≈ 200–500 ms/pair on CPU; use a GPU for real-time operation.
5. **Database coverage** – localisation fails for areas not covered by the reference flight.

---

## File Structure

```
visual_navigation_realtime/
├── README.md               This file
├── requirements.txt        Python dependencies
├── srt_parser.py           DJI SRT telemetry parser (reused from Ex1)
├── geo_utils.py            Haversine, ECEF conversions, ground footprint
├── feature_extractor.py    SuperPoint keypoint/descriptor extraction
├── retrieval.py            GeoDatabase + FAISS indexing and search
├── matcher.py              LightGlue matching, homography, GPS estimation
├── preprocess.py           Offline: build geo-referenced database
├── navigator.py            Online: real-time localisation pipeline
├── experiment.py           Evaluation: metrics, plots, KML export
└── main.py                 CLI entry point (preprocess / navigate / experiment)
```
