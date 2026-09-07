"""Build quality-filtered windows in seconds from reusable source-frame caches."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.provenance import new_directory, provenance, sha256, write_json
from src.research_data import video_windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--rate',type=float,default=15)
    p.add_argument('--duration',type=float,default=10)
    p.add_argument('--stride',type=float,default=5)
    p.add_argument('--max-interpolation-gap',type=float,default=.25)
    p.add_argument('--max-missing-fraction',type=float,default=.2)
    p.add_argument('--max-unfilled-gap',type=float,default=.5)
    p.add_argument('--allow-nominal-timestamps',action='store_true',help='Explicit fallback for unavailable decoder timestamps; not VFR-accurate.')
    p.add_argument('--allow-truncated',action='store_true',help='Smoke tests only.')
    args=p.parse_args()
    cache=Path(args.cache_dir)
    manifest=json.loads((cache/'manifest.json').read_text())
    results=manifest['results']
    if not results or len(results)!=len(manifest['videos']) or any(r['status']!='ok' for r in results):
        raise ValueError('Cache extraction incomplete or failed; inspect manifest before windowing')
    out=new_directory(args.output)
    chunks=[]; reports=[]; ids=[]; labels=[]; subjects=[]; folds=[]; starts=[]; observed=[]; usable=[]
    seen=set(); fingerprints={}
    for result in results:
        path=cache/result['cache']
        if sha256(path)!=result['sha256']:
            raise ValueError(f'Cache fingerprint changed: {path}')
        with np.load(path,allow_pickle=False) as a:
            meta=json.loads(str(a['metadata']))
            if meta['video_id'] in seen:
                raise ValueError('Duplicate video ID in cache manifest')
            seen.add(meta['video_id'])
            fingerprint=meta.get('source_sha256')
            if fingerprint and fingerprint in fingerprints:
                raise ValueError(f'Duplicate source content: {meta["video_id"]} and {fingerprints[fingerprint]}')
            if fingerprint:
                fingerprints[fingerprint]=meta['video_id']
            if meta['truncated'] and not args.allow_truncated:
                raise ValueError('Truncated extraction requires --allow-truncated for smoke tests')
            if meta['timestamp_source']!='decoder_pos_msec' and not args.allow_nominal_timestamps:
                raise ValueError('Nominal FPS clock requires explicit --allow-nominal-timestamps')
            windows,report=video_windows(a,args.rate,args.duration,args.stride,
                args.max_interpolation_gap,args.max_missing_fraction,args.max_unfilled_gap)
            report.update(video_id=meta['video_id'],subject=meta['metadata']['subject_id'],
                          label=meta['metadata']['label'],fold=meta['metadata']['fold_id'],
                          low_source_rate=bool(1/np.median(np.diff(a['timestamps'])) < args.rate))
            reports.append(report)
            for x,o,u,start in windows:
                chunks.append(x); observed.append(o); usable.append(u); starts.append(start)
                ids.append(meta['video_id']); labels.append(meta['metadata']['label'])
                subjects.append(meta['metadata']['subject_id']); folds.append(meta['metadata']['fold_id'])
    write_json(out/'quality_report.json',reports)
    if not chunks:
        raise ValueError('No accepted windows; inspect quality_report.json')
    metadata=dict(schema_version=2,video_identity_verified=True,temporal_rate_hz=args.rate,
                  duration_seconds=args.duration,stride_seconds=args.stride,
                  parameters=vars(args),cache_manifest_sha256=sha256(cache/'manifest.json'),provenance=provenance())
    np.savez_compressed(out/'windows.npz',X=np.stack(chunks),y=np.asarray(labels,dtype=np.int32),
        subjects=np.asarray(subjects,dtype=np.int32),folds=np.asarray(folds,dtype=np.int32),
        observed=np.stack(observed),usable=np.stack(usable),video_ids=np.asarray(ids),
        start_seconds=np.asarray(starts),metadata=json.dumps(metadata))
    write_json(out/'metadata.json',metadata)
    print(f'{len(chunks)} windows saved to {out}; per-video/class rejection counts in quality_report.json')


if __name__=='__main__':
    main()
