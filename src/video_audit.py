"""Source manifests and timestamped, per-video feature extraction (schema v2)."""
import hashlib
from pathlib import Path
import numpy as np
from src.dataset import find_all_video_files, parse_video_metadata


def discover(roots):
    import cv2
    rows = []
    for path in sorted({str(Path(v).resolve()) for root in roots for v in find_all_video_files(root)}):
        meta = parse_video_metadata(path)
        cap = cv2.VideoCapture(path)
        row = dict(path=path, video_id=hashlib.sha256(path.encode()).hexdigest()[:20],
                   bytes=Path(path).stat().st_size, mtime_ns=Path(path).stat().st_mtime_ns,
                   metadata=meta, opened=bool(cap.isOpened()),
                   width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   fps=float(cap.get(cv2.CAP_PROP_FPS)),
                   frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        if not np.isfinite(row['fps']):
            row['fps'] = 0.0
        row['duration_seconds_nominal'] = row['frame_count'] / row['fps'] if row['fps'] > 0 else None
        rows.append(row)
    return rows


def manifest_summary(rows):
    known = [r['metadata'] for r in rows if r['metadata']]
    present = {(m['subject_id'], m['label']) for m in known}
    return dict(videos=len(rows), unparsed=[r['path'] for r in rows if r['metadata'] is None],
                unreadable=[r['path'] for r in rows if not r['opened']],
                missing_subject_classes=[dict(subject=s, label=c) for s in range(1, 61)
                                         for c in range(3) if (s, c) not in present])


def extract_record(row, pipeline, max_frames=None):
    """Keep all decoded source frames; detection failures are explicit NaNs."""
    import cv2
    from src.feature_extractor import resize_preserving_aspect
    cap = cv2.VideoCapture(row['path'])
    if not cap.isOpened():
        raise ValueError(f"Cannot open {row['path']}")
    features, timestamps, indices = [], [], []
    index = 0
    try:
        while max_frames is None or index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
            indices.append(index)
            frame = resize_preserving_aspect(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            values = pipeline.process_frame(rgb, frame.shape[1], frame.shape[0])
            features.append(values if values is not None else [np.nan] * 5)
            index += 1
    finally:
        cap.release()
    if not features:
        raise ValueError('No decoded frames')
    if max_frames is None and row['frame_count'] > len(features) + 2:
        raise ValueError(f"Decoder stopped at {len(features)} of {row['frame_count']} reported frames; inspect source rather than silently truncating")
    t = np.asarray(timestamps, dtype=np.float64)
    clock = 'decoder_pos_msec'
    if not np.isfinite(t).all() or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        if row['fps'] <= 0:
            raise ValueError('Neither monotonic decoder timestamps nor usable nominal FPS')
        t = np.asarray(indices, dtype=np.float64) / row['fps']
        clock = 'nominal_fps_fallback'  # Explicitly NOT a VFR-accurate clock.
    t -= t[0]
    x = np.asarray(features, dtype=np.float32)
    valid = np.isfinite(x).all(axis=1)
    x[~valid] = np.nan
    metadata = dict(row, timestamp_source=clock, backend=pipeline.mode,
                    extractor_version=2, resize='aspect_fit_640x480', coordinates='float',
                    source_frame_skip=1, truncated=max_frames is not None,
                    detection_rate=float(valid.mean()), decoded_frames=len(x))
    return dict(features=x, timestamps=t, frame_indices=np.asarray(indices), detected=valid), metadata


def compare_frames(row, pipeline, out, count=4):
    """Paired diagnostic: same decoded frames, legacy warp vs uniform resize."""
    import cv2
    from src.feature_extractor import resize_preserving_aspect
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(row['path'])
    results = []
    try:
        for index in np.linspace(0, max(0, row['frame_count'] - 1), count, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            for mode, resized in [('legacy_warp', cv2.resize(frame, (640, 480))),
                                  ('aspect_fit', resize_preserving_aspect(frame))]:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                values = pipeline.process_frame(rgb, resized.shape[1], resized.shape[0])
                landmarks = getattr(pipeline, 'last_landmarks', None)
                if landmarks:
                    for i in [33,160,158,133,153,144,362,385,387,263,373,380,
                              78,81,13,311,308,402,14,178]:
                        pt = landmarks[i]
                        cv2.circle(resized, (round(pt.x * resized.shape[1]), round(pt.y * resized.shape[0])),
                                   2, (0,255,0), -1)
                cv2.putText(resized, mode, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
                filename = f"{row['video_id']}_{index}_{mode}.jpg"
                if not cv2.imwrite(str(out / filename), resized):
                    raise IOError(f'Could not save {filename}')
                results.append(dict(video_id=row['video_id'], frame=int(index), mode=mode,
                                    features=list(values) if values is not None else None, image=filename))
    finally:
        cap.release()
    return results
