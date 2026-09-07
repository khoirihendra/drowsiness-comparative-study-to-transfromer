"""Audit video coverage/resolutions and optionally inspect paired resize/landmark images."""
import argparse
from src.provenance import new_directory, provenance, write_json
from src.video_audit import discover, manifest_summary, compare_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_path', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--compare-videos', type=int, default=0,
                   help='Select a bounded sample across resolution, class and fold; 0 only probes files.')
    p.add_argument('--model-path')
    args = p.parse_args()
    if args.compare_videos < 0:
        p.error('--compare-videos must be nonnegative')
    out = new_directory(args.output)
    rows = discover(args.dataset_path)
    summary = manifest_summary(rows)
    write_json(out / 'manifest.json', dict(provenance=provenance(), videos=rows, summary=summary))
    if not rows:
        raise ValueError('No source videos found; inspect manifest.json')
    if args.compare_videos:
        from src.feature_extractor import FacialLandmarkerPipeline
        buckets = {}
        for row in rows:
            if row['opened'] and row['metadata']:
                key = (row['width'], row['height'], row['metadata']['label'], row['metadata']['fold_id'])
                buckets.setdefault(key, []).append(row)
        selected = []
        while buckets and len(selected) < args.compare_videos:
            for key in list(buckets):
                selected.append(buckets[key].pop(0))
                if not buckets[key]:
                    del buckets[key]
                if len(selected) == args.compare_videos:
                    break
        comparisons = []
        for row in selected:
            pipeline = FacialLandmarkerPipeline(args.model_path)
            try:
                comparisons.extend(compare_frames(row, pipeline, out / 'comparison'))
            finally:
                pipeline.close()
        write_json(out / 'comparison.json', comparisons)
    print(summary)
    print(f'Audit saved to {out}')


if __name__ == '__main__':
    main()
