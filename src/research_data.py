"""Time windows, group-safe selection and train-only preprocessing for research runs."""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from config import FEATURE_SUBSETS, MISSING_FEATURE_VECTOR


def longest_gap(mask, rate):
    edges = np.diff(np.r_[False, ~np.asarray(mask, dtype=bool), False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(lengths.max() / rate) if len(lengths) else 0.0


def resample(t, x, detected, rate, max_gap_seconds):
    """Uniform time grid with bounded interpolation, no extrapolation over gaps.

    `observed` follows the nearest source frame's detection. `usable` also includes
    short interpolated gaps. Low-source-rate interpolation is recorded by callers.
    Angles use shortest-arc interpolation; this does not repair PnP ambiguity.
    """
    t, x, detected = np.asarray(t), np.asarray(x), np.asarray(detected, dtype=bool)
    if len(t) < 2 or x.shape != (len(t), 5) or detected.shape != t.shape:
        raise ValueError('Expected >=2 timestamps, features (N,5), detected (N,)')
    if rate <= 0 or max_gap_seconds < 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Rate must be positive; timestamps must be finite and strictly increasing')
    t = t - t[0]
    grid = np.arange(int(np.floor(t[-1] * rate + 1e-8)) + 1) / rate
    right = np.minimum(np.searchsorted(t, grid), len(t)-1)
    left = np.maximum(right-1, 0)
    nearest = np.where(abs(t[left]-grid) <= abs(t[right]-grid), left, right)
    # A timestamp outage must not be interpreted as a long run of observations.
    close = abs(t[nearest]-grid) <= max(0.5/rate, 0.75*np.median(np.diff(t))) + 1e-8
    source_good = detected & np.isfinite(x).all(axis=1)
    observed = source_good[nearest] & close
    values = np.full((len(grid), 5), np.nan, dtype=np.float32)
    values[observed] = x[nearest[observed]]
    good = np.flatnonzero(source_good)
    if len(good) >= 2:
        gtime = t[good]
        hi = np.clip(np.searchsorted(gtime, grid, side='right'), 1, len(good)-1)
        lo = hi-1
        span = gtime[hi]-gtime[lo]
        bounded = (grid >= gtime[lo]) & (grid <= gtime[hi]) & (span <= max_gap_seconds + 1e-8)
        weight = ((grid-gtime[lo])/span)[:, None]
        delta = x[good[hi]]-x[good[lo]]
        delta[:, 2:] = (delta[:, 2:] + 180) % 360 - 180
        interp = x[good[lo]] + weight * delta
        interp[:, 2:] = (interp[:, 2:] + 180) % 360 - 180
        values[bounded] = interp[bounded]
    usable = np.isfinite(values).all(axis=1)
    return grid, values, observed, usable


def seconds_to_ticks(seconds, rate, name):
    ticks = seconds * rate
    if seconds <= 0 or not np.isfinite(ticks) or not np.isclose(ticks, round(ticks)):
        raise ValueError(f'{name} must be positive and an integer number of target timesteps')
    return int(round(ticks))


def video_windows(arrays, rate, duration, stride, max_gap, max_missing, max_unfilled):
    n = seconds_to_ticks(duration, rate, 'duration')
    step = stride * rate
    if not np.isfinite(step) or step < 1:
        raise ValueError('Stride must span at least one target timestep')
    if not 0 <= max_missing <= 1 or max_unfilled < 0:
        raise ValueError('Invalid missing-data limits')
    t, x, observed, usable = resample(arrays['timestamps'], arrays['features'],
                                     arrays['detected'], rate, max_gap)
    accepted = []
    counts = dict(candidates=0, accepted=0, rejected_missing=0, rejected_gap=0)
    # A half-open window [start, start+duration) contains n target time bins.
    # Fractional strides (e.g. 2.5 s at 15 Hz) alternate integer step lengths.
    # Anchor each start to absolute time so rounding never accumulates drift.
    count = max(0, int(np.floor((len(t)-n) / step)) + 1)
    for ordinal in range(count):
        start = int(round(ordinal * step))
        counts['candidates'] += 1
        sl = slice(start, start+n)
        if np.mean(~observed[sl]) > max_missing:
            counts['rejected_missing'] += 1
        elif longest_gap(usable[sl], rate) > max_unfilled or not usable[sl].any():
            counts['rejected_gap'] += 1
        else:
            counts['accepted'] += 1
            accepted.append((x[sl], observed[sl], usable[sl], float(t[start])))
    return accepted, counts


@dataclass
class WindowData:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    folds: np.ndarray
    observed: np.ndarray
    usable: np.ndarray
    videos: np.ndarray
    starts: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            x = a['X'].astype(np.float32)
            legacy = 'metadata' not in a or 'video_ids' not in a
            if legacy:
                valid = np.isfinite(x).all(axis=-1) & ~np.all(x == np.asarray(MISSING_FEATURE_VECTOR, dtype=x.dtype), axis=-1)
                observed, usable = valid, valid.copy()
                videos = np.full(len(x), '', dtype='U1')
                starts = np.full(len(x), np.nan)
                metadata = dict(schema_version=1, video_identity_verified=False,
                                temporal_rate_hz=None, missing_mask='inferred_exact_padding')
            else:
                observed, usable = a['observed'].astype(bool), a['usable'].astype(bool)
                videos, starts = a['video_ids'].astype(str), a['start_seconds']
                metadata = json.loads(str(a['metadata']))
            result = cls(x, a['y'], a['subjects'], a['folds'], observed, usable, videos, starts, metadata)
        result.validate()
        return result

    def validate(self):
        if self.X.ndim != 3 or self.X.shape[1] < 2 or self.X.shape[2] != 5 or len(self.X) == 0:
            raise ValueError('Expected nonempty X (N,T>=2,5)')
        n, t, _ = self.X.shape
        for a in (self.y, self.subjects, self.folds, self.videos, self.starts):
            if a.shape != (n,):
                raise ValueError('Metadata must be one-dimensional and aligned with windows')
        if self.usable.shape != (n,t) or self.observed.shape != (n,t):
            raise ValueError('Detection masks must have shape (N,T)')
        if np.any(self.observed & ~self.usable):
            raise ValueError('Observed frames must also be usable')
        for a, valid in [(self.y, range(3)), (self.folds, range(1,6)), (self.subjects, range(1,61))]:
            if not np.isin(a, list(valid)).all():
                raise ValueError('Invalid class, fold or subject metadata')
        if not self.usable.any() or not np.isfinite(self.X[self.usable]).all():
            raise ValueError('No usable signal or nonfinite features marked usable')
        if np.all(np.ptp(self.X[self.usable], axis=0) < 1e-8):
            raise ValueError('All usable features are constant; re-extract videos')
        for subject in np.unique(self.subjects):
            if len(np.unique(self.folds[self.subjects == subject])) != 1:
                raise ValueError(f'Subject {subject} appears in multiple folds')
        if self.metadata.get('video_identity_verified'):
            if not np.isfinite(self.starts).all() or np.any(self.starts < 0) or np.any(self.videos == ''):
                raise ValueError('Verified video windows require IDs and finite nonnegative timestamps')
            for video in np.unique(self.videos):
                m = self.videos == video
                if len(np.unique(self.starts[m])) != m.sum():
                    raise ValueError(f'Duplicate window start for video {video}')
                if any(len(np.unique(a[m])) != 1 for a in (self.subjects, self.folds, self.y)):
                    raise ValueError(f'Video {video} crosses metadata boundaries')

    def split_indices(self, test_fold, val_fold=None):
        folds = sorted(np.unique(self.folds).tolist())
        if len(folds) < 3 or test_fold not in folds:
            raise ValueError('Need >=3 present folds including requested test fold')
        val_fold = folds[(folds.index(test_fold)+1) % len(folds)] if val_fold is None else val_fold
        if val_fold not in folds or val_fold == test_fold:
            raise ValueError('Validation fold must be present and distinct from test fold')
        masks = [(self.folds != test_fold) & (self.folds != val_fold),
                 self.folds == val_fold, self.folds == test_fold]
        if any(set(np.unique(self.y[m])) != {0,1,2} for m in masks):
            raise ValueError('Each split must contain all three classes')
        return tuple(np.flatnonzero(m) for m in masks)

    def select_stride(self, ids, seconds):
        if seconds is None:
            return ids
        if not self.metadata.get('video_identity_verified'):
            raise ValueError('Legacy NPZ has no verified video starts; use --train-thinning, not stride')
        base = self.metadata['stride_seconds']
        if not np.isfinite(seconds) or seconds < base or not np.isclose(seconds/base, round(seconds/base)):
            raise ValueError('Selection stride must be an integer multiple of archive stride')
        # Select on original clock lattice, not on index among accepted windows.
        rate = self.metadata['temporal_rate_hz']
        ordinal = np.rint(self.starts[ids] / seconds)
        target = np.rint(ordinal * seconds * rate) / rate
        selected = ids[np.isclose(self.starts[ids], target, rtol=0, atol=1e-6)]
        if len(selected) == 0:
            raise ValueError('Requested stride leaves no windows')
        return selected

    def thin_training(self, ids, factor, seed):
        if factor < 1:
            raise ValueError('Training thinning factor must be >=1')
        if factor == 1:
            return ids
        rng = np.random.default_rng(seed)
        selected = []
        for subject in np.unique(self.subjects[ids]):
            for label in range(3):
                group = ids[(self.subjects[ids] == subject) & (self.y[ids] == label)]
                if len(group):
                    selected.extend(rng.choice(group, max(1,len(group)//factor), replace=False))
        return np.sort(selected)


class FeatureTransform:
    """Serializable preprocessing; no statistics are fitted on validation/test."""
    def __init__(self, subset='ear_mar', pose='raw', dynamics=False, mask=True):
        if subset not in FEATURE_SUBSETS or pose not in ('raw','sincos'):
            raise ValueError('Unknown feature subset or pose representation')
        self.subset, self.pose, self.dynamics, self.mask = subset, pose, dynamics, mask
        self.mean = self.scale = None

    def encode(self, x, usable):
        indices = FEATURE_SUBSETS[self.subset]['indices']
        z = x[:,:,indices].astype(np.float32, copy=True)
        z[~usable] = np.nan
        if self.subset == 'all' and self.pose == 'sincos':
            angles = np.deg2rad(z[:,:,2:])
            z = np.concatenate([z[:,:,:2], np.sin(angles), np.cos(angles)], axis=2)
        if self.dynamics:
            delta = np.full_like(z[:,:,:min(2,len(indices))], np.nan)
            pairs = usable[:,1:] & usable[:,:-1]
            delta[:,1:] = np.where(pairs[:,:,None], np.diff(z[:,:,:delta.shape[2]],axis=1), np.nan)
            z = np.concatenate([z,delta],axis=2)
        return z

    def fit(self, x, usable):
        sums = squares = count = None
        for start in range(0,len(x),1024):
            z = self.encode(x[start:start+1024],usable[start:start+1024]).reshape(-1,self.encode(x[:1],usable[:1]).shape[-1]).astype(np.float64)
            finite = np.isfinite(z)
            clean = np.where(finite,z,0)
            if sums is None:
                sums = np.zeros(z.shape[-1]); squares=sums.copy(); count=sums.copy()
            sums += clean.sum(0); squares += (clean*clean).sum(0); count += finite.sum(0)
        if count is None or np.any(count == 0):
            raise ValueError('A feature has no valid training observations')
        self.mean = sums/count
        self.scale = np.sqrt(np.maximum(squares/count-self.mean**2,0))
        self.scale[self.scale < 1e-6] = 1
        return self

    def transform(self, x, usable, observed):
        if self.mean is None:
            raise ValueError('Fit preprocessing on training first')
        z = self.encode(x,usable)
        z = np.where(np.isfinite(z),(z-self.mean)/self.scale,0).astype(np.float32)
        if self.mask:
            z = np.concatenate([z,observed[:,:,None],usable[:,:,None]],axis=2).astype(np.float32)
        return z

    def save(self, path):
        from src.provenance import write_json
        write_json(path, dict(subset=self.subset,pose=self.pose,dynamics=self.dynamics,mask=self.mask,
                             mean=self.mean.tolist(),scale=self.scale.tolist()))

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text()); mean=d.pop('mean'); scale=d.pop('scale')
        obj=cls(**d); obj.mean=np.asarray(mean); obj.scale=np.asarray(scale)
        return obj


def summary_features(z):
    """Simple classifier input: level, spread, extrema and temporal changes."""
    return np.concatenate([z.mean(1),z.std(1),z.min(1),z.max(1),
                           np.mean(np.abs(np.diff(z,axis=1)),axis=1)],axis=1)
