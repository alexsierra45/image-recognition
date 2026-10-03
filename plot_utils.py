"""Reusable plots for the experiment sweeps. Figures are saved to experiments/plots/ (names in English, phase prefix).

Convention in every figure: one colour per experiment / family, train = dashed line, validation = solid line.
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_utils import CSV_PATH, EXP_DIR, HISTORY_DIR
from plot_theme import GREY, INK, INK2, PALETTE, RC

PLOTS_DIR = os.path.join(EXP_DIR, 'plots')

matplotlib.rcParams.update(RC)

METRIC_LABELS = {'accuracy': 'accuracy', 'loss': 'loss', 'mean_recall': 'mean recall (macro)', 'val_acc': 'val accuracy',
                 'mean_recall_val': 'mean recall (macro)'}


def color(i):
    return PALETTE[i] if i < len(PALETTE) else GREY


def load_history(run_id):
    with open(os.path.join(HISTORY_DIR, run_id + '.json')) as ifs:
        return json.load(ifs)


def load_results(exp_ids=None, phase=None):
    df = pd.read_csv(CSV_PATH)
    if exp_ids is not None:
        df = df[df['exp_id'].isin([exp_ids] if isinstance(exp_ids, str) else exp_ids)]
    if phase is not None:
        df = df[df['phase'].astype(str) == str(phase)]
    return df.reset_index(drop=True)


def expand_config(df):
    """Adds columns parsed from the JSON config: res, lr, optimizer, batch_size, hidden, family (# hidden layers),
    activation, dropout, l2, bn, gap (train - val mean recall), gap_acc (train - val accuracy)."""
    df = df.copy()
    cfgs = [json.loads(c) for c in df['config']]
    df['res'] = [c['res'] for c in cfgs]
    df['lr'] = [c['optimizer']['lr'] for c in cfgs]
    df['optimizer'] = [c['optimizer']['name'] for c in cfgs]
    df['batch_size'] = [c['batch_size'] for c in cfgs]
    df['hidden'] = ['-'.join(map(str, c['hidden'])) or 'none' for c in cfgs]
    df['family'] = ['%d hidden' % len(c['hidden']) for c in cfgs]
    df['activation'] = [c['activation'] for c in cfgs]
    df['dropout'] = [c['dropout'] for c in cfgs]
    df['l2'] = [c['l2'] for c in cfgs]
    df['bn'] = [c['bn'] or 'none' for c in cfgs]
    df['variant'] = df['id'].str.replace(r'_seed\d+$', '', regex=True)
    df['seed'] = [c['seed'] for c in cfgs]
    df['gap'] = df['train_mean_recall'] - df['mean_recall']
    df['gap_acc'] = df['train_acc'] - df['val_acc']
    return df


def seed_summary(df, by='variant', cols=('mean_recall', 'val_acc', 'min_recall', 'train_mean_recall', 'train_acc')):
    """Mean and std over seeds per variant (expects expand_config columns)."""
    g = df.groupby(by, sort=False)
    out = g[list(cols)].agg(['mean', 'std'])
    out.columns = ['%s_%s' % c for c in out.columns]
    out.insert(0, 'n_seeds', g.size())
    return out.reset_index()


def plot_seed_bars(summary, y='mean_recall', label='variant', name=None, title=None, ax=None):
    """Bars = mean over seeds of the validation metric, error bars = std; train mean as a dashed tick."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(max(5.0, 0.75 * len(summary)), 3.8))
    xs = np.arange(len(summary))
    ax.bar(xs, summary[y + '_mean'], 0.6, yerr=summary[y + '_std'], color=PALETTE[0], ecolor=INK2, capsize=3,
           label='val (mean ± std over seeds)')
    train_col = {'mean_recall': 'train_mean_recall', 'val_acc': 'train_acc'}.get(y)
    if train_col:
        ax.hlines(summary[train_col + '_mean'], xs - 0.3, xs + 0.3, colors=INK, linestyles='--', lw=1.2, label='train (mean)')
    for x_, v in zip(xs, summary[y + '_mean']):
        ax.text(x_ + 0.32, v, '%.3f' % v, ha='left', va='center', fontsize=6.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(summary[label], rotation=30, ha='right')
    ax.set_ylabel(METRIC_LABELS.get(y, y))
    ax.grid(axis='x', visible=False)
    ax.set_title(title or METRIC_LABELS.get(y, y))
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    return _save(fig, name)


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if name:
        fig.savefig(os.path.join(PLOTS_DIR, name + '.png'))
    return fig


def plot_learning_curves(run_ids, metric='loss', labels=None, name=None, title=None, ax=None, logy=None):
    """Train (dashed) and validation (solid) curves per epoch. metric: 'loss' | 'accuracy' | 'mean_recall' | 'learning_rate'."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 3.8))
    for i, rid in enumerate(run_ids):
        h = load_history(rid)['history']
        lab = labels[i] if labels else rid
        if metric == 'learning_rate':
            ax.plot(np.arange(1, len(h['learning_rate']) + 1), h['learning_rate'], color=color(i), label=lab)
            continue
        val = h['val_' + metric]
        ep = np.arange(1, len(val) + 1)
        ax.plot(ep, val, color=color(i), label=lab + ' (val)')
        if metric in h:
            ax.plot(ep, h[metric], color=color(i), ls='--', lw=1.1, label=lab + ' (train)')
    ax.set_xlabel('epoch')
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    if logy or (logy is None and metric in ('loss', 'learning_rate')):
        ax.set_yscale('log')
    ax.set_title(title or ('Learning curves: ' + METRIC_LABELS.get(metric, metric)))
    ax.legend(fontsize=7, ncol=2 if len(run_ids) > 3 else 1)
    return _save(fig, name)


def plot_sweep(df, x, y='mean_recall', logx=False, group=None, name=None, title=None, xlabel=None, annotate=None, ax=None):
    """Validation metric (solid) and the train counterpart (dashed) vs a hyperparameter. One line per `group`."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 3.8))
    train_col = {'mean_recall': 'train_mean_recall', 'val_acc': 'train_acc'}.get(y)
    groups = [(None, df)] if group is None else list(df.groupby(group, sort=False))
    for i, (g, d) in enumerate(groups):
        d = d.sort_values(x)
        lab = '' if g is None else '%s ' % g
        ax.plot(d[x], d[y], marker='o', color=color(i), label=lab + 'val')
        if train_col:
            ax.plot(d[x], d[train_col], marker='o', mfc='white', ls='--', lw=1.1, color=color(i), label=lab + 'train')
        if annotate:
            for _, r in d.iterrows():
                ax.annotate(str(r[annotate]), (r[x], r[y]), textcoords='offset points', xytext=(0, -12),
                            ha='center', fontsize=7, color=INK2)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(METRIC_LABELS.get(y, y))
    ax.set_title(title or '%s vs %s' % (METRIC_LABELS.get(y, y), x))
    ax.legend(fontsize=7)
    return _save(fig, name)


def plot_capacity(df, group='family', metric='val_acc', name=None, title=None, ax=None):
    """Train (dashed) and validation (solid) metric vs number of parameters (log x), one line per family."""
    return plot_sweep(df, 'params', metric, logx=True, group=group, name=name, ax=ax,
                      title=title or 'Capacity: %s vs parameters' % METRIC_LABELS.get(metric, metric),
                      xlabel='parameters (log)')


def plot_per_class_recall(run_ids, labels=None, name=None, title=None):
    """Grouped bars: validation recall per class, one colour per run."""
    hs = [load_history(r) for r in run_ids]
    classes = hs[0]['class_names']
    n = len(run_ids)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(classes) * max(1, n / 2)), 3.8))
    xs = np.arange(len(classes))
    for i, h in enumerate(hs):
        lab = labels[i] if labels else run_ids[i]
        ax.bar(xs + (i - (n - 1) / 2) * width, h['val']['recall'], width * 0.9, color=color(i),
               label='%s (mean %.3f)' % (lab, h['val']['mean_recall']))
    ax.set_xticks(xs)
    ax.set_xticklabels(classes, rotation=40, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('validation recall')
    ax.grid(axis='x', visible=False)
    ax.set_title(title or 'Per-class recall (validation)')
    ax.legend(fontsize=7)
    return _save(fig, name)


def plot_confusion_matrix(run_id, name=None, title=None):
    """Row-normalised validation confusion matrix (rows = annotation)."""
    h = load_history(run_id)
    cm = np.array(h['val']['cm'], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    classes = h['class_names']
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Annotation')
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, '%.2f' % cmn[i, j], ha='center', va='center', fontsize=6.5,
                        color='white' if cmn[i, j] > 0.5 else INK)
    ax.set_title(title or 'Confusion matrix (validation): %s' % run_id)
    return _save(fig, name)


def plot_bars(df, y='mean_recall', label='id', name=None, title=None, ax=None):
    """Validation metric per variant (bars) with the train counterpart as a dashed tick."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(max(5.0, 0.7 * len(df)), 3.8))
    train_col = {'mean_recall': 'train_mean_recall', 'val_acc': 'train_acc'}.get(y)
    xs = np.arange(len(df))
    ax.bar(xs, df[y], 0.6, color=PALETTE[0], label='val')
    if train_col:
        ax.hlines(df[train_col], xs - 0.3, xs + 0.3, colors=INK, linestyles='--', lw=1.2, label='train')
    for x_, v in zip(xs, df[y]):
        ax.text(x_, v + 0.01, '%.3f' % v, ha='center', fontsize=7, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(df[label], rotation=30, ha='right')
    ax.set_ylabel(METRIC_LABELS.get(y, y))
    ax.grid(axis='x', visible=False)
    ax.set_title(title or METRIC_LABELS.get(y, y))
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    return _save(fig, name)
