"""Subject-disjoint, validation-first experiments. Never overwrite historical outputs."""
import argparse
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from uuid import uuid4
import numpy as np
from src.provenance import new_directory, provenance, sha256, write_json
from src.research_data import WindowData, FeatureTransform
from src.research_metrics import evaluation_report
from src.research_training import fit_classifier, predict_bundle


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path', '--data-path', required=True)
    p.add_argument('--models', nargs='+', choices=['logistic', 'bilstm', 'lstm', 'bigru', 'cnn1d', 'transformer', 'xgboost'], default=['logistic'])
    p.add_argument('--features', nargs='+', choices=['ear', 'ear_mar', 'all'], default=['ear_mar'])
    p.add_argument('--folds', nargs='+', type=int, default=[1], help='Reserved test fold(s); next fold is validation.')
    p.add_argument('--phase', choices=['validation', 'final'], default='validation')
    p.add_argument('--output-dir', default='output/runs')
    p.add_argument('--run-id', help='New directory name; existing runs cannot be overwritten.')
    p.add_argument('--train-thinning', nargs='+', type=int, default=[1], help='Random fraction 1/factor within subject/class; NOT verified non-overlap.')
    p.add_argument('--train-stride', type=float, help='Seconds, v2 only; default archive stride.')
    p.add_argument('--eval-stride', type=float, help='Seconds, v2 only; defaults to duration (non-overlap).')
    p.add_argument('--pose', choices=['raw', 'sincos'], default='sincos')
    p.add_argument('--dynamics', action='store_true', help='Append valid adjacent EAR/MAR differences.')
    p.add_argument('--no-mask-channels', action='store_true')
    p.add_argument('--no-positional-encoding', action='store_true', help='Transformer ablation only.')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', '--batch-size', type=int, default=64)
    p.add_argument('--steps-per-epoch', type=int)
    p.add_argument('--max-updates', type=int, help='NN fixed update budget; disables early stopping.')
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--logistic-c', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-plots', action='store_true', help='Save numerical artifacts without PNG figures.')
    args = p.parse_args()
    if any(v < 1 for v in [args.epochs, args.batch_size, args.patience, *args.train_thinning]):
        p.error('Counts and thinning factors must be positive')
    if any(v is not None and v < 1 for v in (args.max_updates, args.steps_per_epoch)):
        p.error('Update counts must be positive')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.logistic_c) or args.logistic_c <= 0:
        p.error('Learning rate and logistic C must be finite and positive')
    if args.max_updates and any(m in ('logistic', 'xgboost') for m in args.models):
        p.error('--max-updates compares neural models only; run tabular baselines separately')
    if len(args.train_thinning) > 1 and any(m not in ('logistic', 'xgboost') for m in args.models):
        if not args.max_updates or not args.steps_per_epoch:
            p.error('Neural thinning grids require --max-updates AND --steps-per-epoch for matched budgets and validation frequency')
    for values in (args.models, args.features, args.train_thinning, args.folds):
        if len(values) != len(set(values)):
            p.error('Duplicate grid values would create duplicate experiment paths')
    if args.phase == 'final' and (len(args.models)*len(args.features)*len(args.train_thinning) != 1):
        p.error('Final phase accepts one frozen configuration, not a tuning grid')
    if args.run_id and (Path(args.run_id).name != args.run_id or args.run_id in ('.', '..')):
        p.error('--run-id must be a simple directory name')
    return args


def describe_split(data, ids):
    return dict(n=len(ids), subjects=np.unique(data.subjects[ids]).tolist(),
                folds=np.unique(data.folds[ids]).tolist(),
                class_counts=np.bincount(data.y[ids], minlength=3).tolist())


def aggregate_folds(summaries):
    groups = {}
    for row in summaries:
        key = (row['model'], row['features'], row['thinning'], row['phase'])
        groups.setdefault(key, []).append(row)
    result = []
    for (model, features, thinning, phase), rows in groups.items():
        result.append(dict(model=model, features=features, thinning=thinning, phase=phase,
            folds=[r['reserved_test_fold'] for r in rows],
            metrics={key: dict(mean=float(np.mean([r[key] for r in rows])),
                               std_population=float(np.std([r[key] for r in rows])))
                     for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss')},
            note='Unweighted fold mean/std, not a confidence interval or independent window estimate.'))
    return result


def main():
    args = parse_args()
    data = WindowData.load(args.data_path)
    verified = data.metadata.get('video_identity_verified', False)
    if not verified and (args.train_stride is not None or args.eval_stride is not None):
        raise ValueError('Legacy NPZ cannot verify stride; use --train-thinning')
    eval_stride = args.eval_stride if args.eval_stride is not None else data.metadata.get('duration_seconds')
    if verified and eval_stride < data.metadata['duration_seconds']:
        raise ValueError('Evaluation stride must be >= window duration (non-overlap)')
    splits = {fold: data.split_indices(fold) for fold in args.folds}
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    out = new_directory(Path(args.output_dir) / run_id)
    write_json(out / 'run.json', dict(arguments=vars(args), provenance=provenance(),
        data_sha256=sha256(args.data_path), data_metadata=data.metadata,
        evaluation_stride_seconds=eval_stride, verified_nonoverlap_evaluation=bool(verified),
        note='Historical test folds have already informed development; do not describe them as untouched.'))
    summaries = []
    for name, subset, factor, fold in product(args.models, args.features, args.train_thinning, args.folds):
        folder = new_directory(out / f'{name}_{subset}_thin{factor}_fold{fold}')
        original_train, original_val, original_test = splits[fold]
        train = data.select_stride(original_train, args.train_stride)
        val = data.select_stride(original_val, eval_stride)
        train = train[data.usable[train].any(1)]
        val = val[data.usable[val].any(1)]
        train = data.thin_training(train, factor, args.seed)
        test = data.select_stride(original_test, eval_stride) if args.phase == 'final' else np.array([], dtype=int)
        test = test[data.usable[test].any(1)]
        for ids in ([train, val, test] if args.phase == 'final' else [train, val]):
            if set(np.unique(data.y[ids])) != {0, 1, 2}:
                raise ValueError('Selection/filtering removed an entire class from a split')
        split_info = dict(train=describe_split(data, train), validation=describe_split(data, val),
            reserved_test=describe_split(data, original_test),
            evaluated_test=describe_split(data, test) if args.phase == 'final' else None,
            selection_kind='subject_class_random_thinning' if factor > 1 else 'all_selected_windows',
            all_missing_before_selection={key: int((~data.usable[ids].any(1)).sum())
                for key, ids in [('train', original_train), ('validation', original_val), ('reserved_test', original_test)]})
        write_json(folder / 'splits.json', split_info)
        np.savez_compressed(folder / 'split_indices.npz', train=train, validation=val,
                            reserved_test=original_test, evaluated_test=test)
        print(f'{folder.name}: train={len(train)}, validation={len(val)}, phase={args.phase}', flush=True)
        transform = FeatureTransform(subset, args.pose, args.dynamics, not args.no_mask_channels)
        transform.fit(data.X[train], data.usable[train])
        transform.save(folder / 'preprocessing.json')
        x = transform.transform(data.X[train], data.usable[train], data.observed[train])
        xv = transform.transform(data.X[val], data.usable[val], data.observed[val])
        write_json(folder / 'inference.json', dict(input_shape=list(x.shape[1:]),
            raw_feature_order=['EAR', 'MAR', 'Pitch_degrees', 'Yaw_degrees', 'Roll_degrees'],
            window_metadata=data.metadata, predictor=name, requires_preprocessing_and_masks=True))
        fit_classifier(name, x, data.y[train], xv, data.y[val], folder, args)
        del x, xv
        evaluation_ids = test if args.phase == 'final' else val
        probabilities = predict_bundle(folder, data.X[evaluation_ids], data.usable[evaluation_ids],
                                       data.observed[evaluation_ids], args.batch_size)
        metrics = evaluation_report(data, evaluation_ids, probabilities)
        prior = np.bincount(data.y[train], minlength=3).astype(float)
        prior /= prior.sum()
        majority = np.eye(3)[np.argmax(prior)]
        metrics['train_majority_baseline'] = evaluation_report(data, evaluation_ids,
                                                   np.tile(majority, (len(evaluation_ids), 1)))
        metrics['train_prior_baseline'] = evaluation_report(data, evaluation_ids,
                                                   np.tile(prior, (len(evaluation_ids), 1)))
        metrics['phase'] = args.phase
        write_json(folder / 'metrics.json', metrics)
        if not args.no_plots:
            from src.research_plots import save_figures
            save_figures(folder, metrics)
        np.savez_compressed(folder / 'predictions.npz', indices=evaluation_ids,
                            y=data.y[evaluation_ids], probabilities=probabilities,
                            subjects=data.subjects[evaluation_ids], videos=data.videos[evaluation_ids])
        summaries.append(dict(model=name, features=subset, thinning=factor, reserved_test_fold=fold,
                              phase=args.phase, directory=folder.name, **metrics['window']))
        write_json(out / 'summary.json', summaries)
        write_json(out / 'aggregate.json', aggregate_folds(summaries))
        print(f"{args.phase}: accuracy={metrics['window']['accuracy']:.4f}, macro F1={metrics['window']['macro_f1']:.4f}", flush=True)
    print(f'Artifacts saved to {out.resolve()}')


if __name__ == '__main__':
    main()
