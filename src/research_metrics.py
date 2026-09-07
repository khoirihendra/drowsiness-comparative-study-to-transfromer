"""Three-class metrics with explicit aggregation units (values in [0, 1])."""
import numpy as np


def classification_metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not len(y) or p.shape != (len(y), 3) or not np.isfinite(p).all():
        raise ValueError('Expected nonempty labels and finite (N,3) probabilities')
    if np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid class probabilities')
    pred = p.argmax(1)
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support > 0)
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3), where=precision + recall > 0)
    return dict(n=len(y), accuracy=float(np.mean(pred == y)),
                balanced_accuracy=float(recall[support > 0].mean()),
                macro_f1=float(f1.mean()),
                log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
                confusion_matrix=cm.tolist(), support=support.tolist(),
                predicted_counts=cm.sum(0).tolist(), recall=recall.tolist(),
                precision=precision.tolist(), f1=f1.tolist())


def evaluation_report(data, ids, probabilities):
    report = dict(window=classification_metrics(data.y[ids], probabilities))
    subjects = {}
    for subject in np.unique(data.subjects[ids]):
        mask = data.subjects[ids] == subject
        subjects[str(subject)] = classification_metrics(data.y[ids][mask], probabilities[mask])
    report['per_subject'] = subjects
    report['subject_macro'] = {key: float(np.mean([v[key] for v in subjects.values()]))
                               for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss')}
    report['video'] = None
    if data.metadata.get('video_identity_verified'):
        labels, scores, videos = [], [], []
        for video in np.unique(data.videos[ids]):
            mask = data.videos[ids] == video
            labels.append(int(data.y[ids][mask][0]))
            scores.append(probabilities[mask].mean(0))
            videos.append(str(video))
        report['video'] = classification_metrics(labels, scores)
        report['video_predictions'] = dict(video_ids=videos, labels=labels,
                                            probabilities=np.asarray(scores).tolist())
    return report
