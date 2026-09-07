"""Snapshot historical metrics and fingerprint their datasets/checkpoints."""
import argparse
import shutil
from pathlib import Path
from src.provenance import new_directory, provenance, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='output')
    p.add_argument('--output', required=True, help='New directory; existing output is never replaced.')
    args = p.parse_args()
    source = Path(args.source).resolve()
    dest = Path(args.output).resolve()
    # Collect before creating destination; do not recursively snapshot older snapshots.
    files = [p for folder in ('metrics', 'checkpoints', 'features', 'extracted_features')
             for p in (source / folder).rglob('*') if p.is_file() and p.suffix in ('.json', '.keras', '.npz')]
    new_directory(dest)
    records = []
    for path in files:
        records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)))
        if path.suffix == '.json':
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    write_json(dest / 'baseline.json', dict(provenance=provenance(), files=records,
        evaluation_note='Historical test results have already informed development; not an untouched holdout.'))
    print(f'Baseline snapshot: {dest} ({len(records)} fingerprinted files)')


if __name__ == '__main__':
    main()
