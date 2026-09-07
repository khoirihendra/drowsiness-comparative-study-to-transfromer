"""Extract source-frame features into independent video caches; build windows separately."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from src.provenance import new_directory, provenance, sha256, write_json
from src.video_audit import discover, extract_record, manifest_summary


def worker(task):
    row, model, out, max_frames = task
    try:
        source = Path(row['path'])
        fingerprint = sha256(source)
        if source.stat().st_size != row['bytes'] or source.stat().st_mtime_ns != row['mtime_ns']:
            raise ValueError('Source changed after discovery')
        from src.feature_extractor import FacialLandmarkerPipeline
        pipeline = FacialLandmarkerPipeline(model)
        try:
            arrays, metadata = extract_record(row, pipeline, max_frames)
        finally:
            pipeline.close()
        if source.stat().st_size != row['bytes'] or source.stat().st_mtime_ns != row['mtime_ns']:
            raise ValueError('Source changed during extraction')
        metadata['source_sha256'] = fingerprint
        if not arrays['detected'].any():
            raise ValueError('No valid face detection in this video')
        path = Path(out) / (row['video_id'] + '.npz')
        np.savez_compressed(path, **arrays, metadata=json.dumps(metadata))
        return dict(video_id=row['video_id'], path=row['path'], status='ok',
                    cache=path.name, sha256=sha256(path), metadata=metadata)
    except Exception as exc:
        return dict(video_id=row['video_id'], path=row['path'], status='failed', error=str(exc))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_path', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-path', required=True, help='Existing face_landmarker.task; shared by workers.')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-frames', type=int, help='Smoke tests only; marks caches as truncated.')
    args = p.parse_args()
    if not Path(args.model_path).is_file() or args.num_workers < 1 or (args.max_frames is not None and args.max_frames < 1):
        p.error('An existing model, positive worker count and positive max-frames are required.')
    rows = discover(args.dataset_path)
    if not rows or any(not r['opened'] or r['metadata'] is None for r in rows):
        raise ValueError(f'Source audit failed: {manifest_summary(rows)}. Run audit_dataset.py first.')
    out = new_directory(args.output)
    record = dict(provenance=provenance(), model_sha256=sha256(args.model_path),
                  summary=manifest_summary(rows), videos=rows, results=[])
    write_json(out / 'manifest.json', record)
    tasks = [(r, args.model_path, str(out), args.max_frames) for r in rows]
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for result in executor.map(worker, tasks):
            record['results'].append(result)
            write_json(out / 'manifest.json', record)
            print(f"[{len(record['results'])}/{len(rows)}] {result['status']}: {result['path']}", flush=True)
    if any(r['status'] != 'ok' for r in record['results']):
        raise RuntimeError('Some videos failed; see manifest.json. No failures were silently dropped.')


if __name__ == '__main__':
    main()
