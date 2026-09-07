"""Figures derived from the same saved metrics/history as the numerical report."""
import json
import warnings
import numpy as np


def save_figures(folder, metrics):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn('matplotlib unavailable; metrics/history remain available without PNGs')
        return
    cm = np.asarray(metrics['window']['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    labels = ['Alert', 'Low vigilant', 'Drowsy']
    ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
           xlabel='Predicted', ylabel='True', title=f"{metrics['phase']} — window confusion")
    fig.tight_layout()
    fig.savefig(folder / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    training = json.loads((folder / 'training.json').read_text())
    history = training.get('history', {})
    if 'loss' not in history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ('loss', 'accuracy')):
        for series in (key, 'val_' + key):
            if series in history:
                ax.plot(range(1, len(history[series]) + 1), history[series], label=series)
        ax.set(xlabel='Epoch', ylabel=key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'learning_curves.png', dpi=160)
    plt.close(fig)
