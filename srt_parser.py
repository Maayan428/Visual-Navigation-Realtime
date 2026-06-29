import os
import re


def parse_srt(path: str) -> list:
    """Parse a DJI SRT telemetry file and return 1fps-sampled frame records."""
    with open(path, encoding='utf-8-sig') as f:
        text = f.read()

    # Split on one or more blank lines
    blocks = re.split(r'\n\s*\n', text.strip())

    frames = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 5:
            continue

        # Line 2: strip HTML font tag → "FrameCnt: N, ..." or "SrtCnt : N, ..."
        # Both DJI formats are handled; spaces around the colon are optional.
        fc_line = re.sub(r'<[^>]+>', '', lines[2]).strip()
        m = re.search(r'(?:FrameCnt|SrtCnt)\s*:\s*(\d+)', fc_line)
        if not m:
            continue
        frame_cnt = int(m.group(1))

        # Sample at 1fps: keep frames 1, 31, 61 ... (every 30th starting at 1)
        if (frame_cnt - 1) % 30 != 0:
            continue

        timestamp = lines[3].strip()

        # Line 4: strip </font> → telemetry
        tel_line = re.sub(r'<[^>]+>', '', lines[4]).strip()

        try:
            record = _parse_telemetry(tel_line)
        except (ValueError, AttributeError):
            continue

        record['frame_cnt'] = frame_cnt
        record['timestamp'] = timestamp
        frames.append(record)

    # Filter out frames where GPS has not yet been acquired (lat/lon == 0).
    # This happens when the drone is still on the ground before lock.
    before = len(frames)
    frames = [f for f in frames if f['lat'] != 0.0 and f['lon'] != 0.0]
    dropped = before - len(frames)
    if dropped:
        print(f"  [{os.path.basename(path)}] Dropped {dropped} zero-GPS frames "
              f"({before} sampled → {len(frames)} kept)")

    return frames


def _parse_telemetry(line: str) -> dict:
    """Extract all telemetry fields from a single telemetry line."""
    record = {}

    # Simple bracketed fields: [key: value] or [key : value] (both DJI formats).
    # Commas are excluded from the value so [dzoom_ratio: 10000, delta:0] is skipped.
    for m in re.finditer(r'\[(\w+)\s*:\s*([^\]\s,]+)\]', line):
        key, val = m.group(1), m.group(2)
        if key == 'iso':
            record['iso'] = int(val)
        elif key == 'shutter':
            record['shutter'] = val
        elif key == 'fnum':
            record['fnum'] = float(val)
        elif key == 'ev':
            record['ev'] = int(val)
        elif key == 'focal_len':
            record['focal_len'] = float(val)
        elif key == 'latitude':
            record['lat'] = float(val)
        elif key == 'longitude':
            record['lon'] = float(val)
        elif key == 'ct':
            record['ct'] = int(val)
        # color_md and other unknown fields are ignored

    # Compound field: [rel_alt: X abs_alt: Y]
    m = re.search(r'\[rel_alt:\s*([\d.]+)\s+abs_alt:\s*([\d.]+)\]', line)
    if m:
        record['rel_alt'] = float(m.group(1))
        record['abs_alt'] = float(m.group(2))

    for field in ('lat', 'lon', 'rel_alt'):
        if field not in record:
            raise ValueError(f"Missing field: {field}")

    return record
