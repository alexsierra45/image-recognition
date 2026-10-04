"""Curated figures for a delivery, redrawn from experiments/ (read-only) following <delivery>/figure_specs.yaml.

Same style as the experiment plots (plot_theme.py), English labels, PNG. Writes <delivery>/overleaf/figures/<id>.png,
<delivery>/overleaf/generated/figure_captions.tex (\\figcaption{id}: Spanish caption + selection note) and
<delivery>/figure_log.md (what was included / omitted and why).

Usage: python scripts/build_figures.py reports/delivery_01
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402
from build_results import ROOT, Data, tex  # noqa: E402

C = ps.CONCEPT


def cfg_of(data, run_id):
    return json.loads(data.rows[run_id]['config'])


def variant_of(run_id):
    return re.sub(r'_seed\d+$', '', run_id)


def seeds_curve(data, variant, key):
    curves = [data.history(r)['history'][key] for r in data.runs(variant)]
    n = min(len(c) for c in curves)
    a = np.array([c[:n] for c in curves])
    return a.mean(0), a.min(0), a.max(0)


def smooth(y, k):
    if k <= 1:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode='edge')
    return np.convolve(pad, np.ones(k) / k, mode='valid')


def seed_bars(ax, data, items, metric, train_col):
    """Bars = mean over seeds of the validation metric (error bar = std); train mean as a dashed tick
    (the same encoding as plot_utils.plot_seed_bars)."""
    xs = np.arange(len(items))
    means = [data.stat(it['variant'], metric)[0] for it in items]
    stds = [data.stat(it['variant'], metric)[1] for it in items]
    cols = [C['baseline'] if it.get('role') == 'baseline' else C['final'] for it in items]
    ax.bar(xs, means, 0.6, yerr=[0 if np.isnan(s) else s for s in stds], color=cols, ecolor=ps.INK2, capsize=3)
    ax.hlines([data.stat(it['variant'], train_col)[0] for it in items], xs - 0.3, xs + 0.3, colors=ps.INK,
              linestyles='--', lw=1.2, label='train (mean)')
    ax.bar([0], [0], color=C['final'], label='val (mean ± std over seeds)')   # legend swatch in the accent colour
    if any(it.get('role') == 'baseline' for it in items):
        ax.bar([0], [0], color=C['baseline'], label='reference (val)')
    for x, v in zip(xs, means):
        ax.text(x + 0.32, v, '%.3f' % v, ha='left', va='center', fontsize=6.5, color=ps.INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([it['label'] for it in items], rotation=30, ha='right')
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))


# ---------------------------------------------------------------- figure types
def fig_class_distribution(data, spec):
    ann = json.load(open(os.path.join(ROOT, 'xview_recognition', 'xview_ann_train.json')))
    counts = {}
    for a in ann['annotations'].values():
        counts[a['category_id']] = counts.get(a['category_id'], 0) + 1
    names = sorted(counts, key=counts.get)
    fig, ax = ps.new_figure(spec['layout'], 1.1)
    vals = [counts[n] for n in names]
    ax.barh(range(len(names)), vals, color=ps.PALETTE[0], height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i, v in enumerate(vals):
        ax.text(v, i, ' %d' % v, va='center', fontsize=6.5, color=ps.INK2)
    ax.set_xlabel('labelled images (train + validation)')   # counts of the whole annotations file, before the split
    ax.set_title('Images per class')
    ax.grid(axis='y', visible=False)
    return fig, {'included': 'all 13 classes (annotations file)', 'omitted': 'none', 'note': ''}


def fig_sweep(data, spec):
    sel, opt = spec['select'], spec.get('options', {})
    runs = [r for r in data.rows if data.rows[r]['exp_id'] == sel['exp_id']]
    xk = sel['x']
    pts = []
    for r in runs:
        c = cfg_of(data, r)
        pts.append({'id': r, 'x': c[xk], 'lr': c['optimizer']['lr'], 'val': float(data.rows[r]['mean_recall']),
                    'train': float(data.rows[r]['train_mean_recall'])})
    main = sorted([p for p in pts if abs(p['lr'] - sel['filter']['lr']) < 1e-12], key=lambda p: p['x'])
    fig, ax = ps.new_figure(spec['layout'])
    xs = [p['x'] for p in main]
    ax.plot(xs, [p['val'] for p in main], marker='o', color=ps.PALETTE[0], label='val (lr 1e-3)')
    if opt.get('show_train'):
        ax.plot(xs, [p['train'] for p in main], marker='o', mfc='white', ls='--', lw=1.1, color=ps.PALETTE[0], label='train (lr 1e-3)')
    extra, omitted, note = [], [], ''
    if sel.get('best_lr_per_x'):
        best = {}
        for p in pts:
            if p['x'] not in best or p['val'] > best[p['x']]['val']:
                best[p['x']] = p
        multi = sorted(x for x in best if sum(1 for p in pts if p['x'] == x) > 1)
        ax.plot(multi, [best[x]['val'] for x in multi], ls='none', marker='D', color=ps.PALETTE[1], label='val, best lr per size')
        extra = [best[x]['id'] for x in multi if best[x]['id'] not in {p['id'] for p in main}]
        shown = {p['id'] for p in main} | set(extra)
        omitted = sorted(p['id'] for p in pts if p['id'] not in shown)
        note = ('Se muestran %d de %d runs: todas las resoluciones con lr = 1e-3 y, en las resoluciones con varias lr '
                'probadas, la de mejor mean recall de validación.' % (len(main) + len(extra), len(pts)))
    if opt.get('logx'):
        ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(opt.get('xlabel', xk))
    ax.set_ylabel('mean recall (macro)')
    ax.set_title(opt.get('title', 'Mean recall vs %s' % xk))
    ax.legend(fontsize=7)
    return fig, {'included': [p['id'] for p in main] + extra, 'omitted': omitted,
                 'criterion': 'main sweep at lr 1e-3; per size, the lr with the best validation mean recall', 'note': note}


def fig_capacity(data, spec):
    sel = spec['select']
    runs = [r for r in data.rows if data.rows[r]['exp_id'] == sel['exp_id']]
    groups = {}
    for r in runs:
        groups.setdefault(len(cfg_of(data, r)['hidden']), []).append(r)
    fig, ax = ps.new_figure(spec['layout'])
    for i, n in enumerate(sorted(groups)):
        g = sorted(groups[n], key=lambda r: float(data.rows[r]['params']))
        P = [float(data.rows[r]['params']) for r in g]
        ax.plot(P, [float(data.rows[r]['val_acc']) for r in g], marker='o', color=ps.PALETTE[i], label='%d hidden val' % n)
        ax.plot(P, [float(data.rows[r]['train_acc']) for r in g], marker='o', mfc='white', ls='--', lw=1.1,
                color=ps.PALETTE[i], label='%d hidden train' % n)
    ax.set_xscale('log')
    ax.set_xlabel('parameters (log)')
    ax.set_ylabel('accuracy')
    ax.set_title('Capacity: accuracy vs parameters (train dashed)')
    ax.legend(fontsize=6.5, ncol=2)
    return fig, {'included': runs, 'omitted': [], 'criterion': 'all architectures of EXP-006 (1 seed each), one line per depth',
                 'note': ''}


def fig_strength(data, spec):
    sel, opt = spec['select'], spec.get('options', {})
    xs, val, tr, inc = [0.0], [data.stat(sel['base'], 'val_acc')[0]], [data.stat(sel['base'], 'train_acc')[0]], [sel['base']]
    for v in sorted({variant_of(r) for r in data.rows if r.startswith(sel['prefix']) and '_lr' not in r}):
        xs.append(cfg_of(data, data.runs(v)[0])[sel['param']])
        val.append(data.stat(v, 'val_acc')[0])
        tr.append(data.stat(v, 'train_acc')[0])
        inc.append(v)
    o = np.argsort(xs)
    xs, val, tr = np.array(xs)[o], np.array(val)[o], np.array(tr)[o]
    fig, ax = ps.new_figure(spec['layout'])
    ax.plot(xs, val, marker='o', color=ps.PALETTE[0], label='val')
    ax.plot(xs, tr, marker='o', mfc='white', ls='--', lw=1.1, color=ps.PALETTE[0], label='train')
    if opt.get('logx'):  # 0 (no penalty) is drawn one decade below the smallest value and labelled "none"
        x0 = xs[1] / 10
        labels = ['none'] + ['%g' % x for x in xs[1:]]
        xs = np.where(xs == 0, x0, xs)
        ax.lines[0].set_xdata(xs)
        ax.lines[1].set_xdata(xs)
        ax.set_xscale('log')
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.minorticks_off()
    ax.set_xlabel(opt.get('xlabel', sel['param']))
    ax.set_ylabel('accuracy')
    ax.set_title(opt.get('title', 'Validation curve: accuracy vs %s (mean of 3 seeds)' % sel['param']))
    ax.legend(fontsize=7)
    return fig, {'included': inc, 'omitted': ['EXP-013 L2 and BatchNorm variants (other techniques)'],
                 'criterion': 'dropout variants of EXP-013 + their base, mean of 3 seeds', 'note': ''}


def fig_bars(data, spec):
    opt = spec.get('options', {})
    m = opt.get('metric', 'mean_recall')
    fig, ax = ps.new_figure(spec['layout'], 1.05)
    seed_bars(ax, data, spec['select']['variants'], m, {'mean_recall': 'train_mean_recall', 'val_acc': 'train_acc'}[m])
    ax.set_ylabel('mean recall (macro)' if m == 'mean_recall' else 'accuracy')
    ax.set_title(opt.get('title', ''))
    return fig, {'included': [it['variant'] for it in spec['select']['variants']], 'omitted': [],
                 'criterion': 'all variants of the step and their base, 3 seeds each', 'note': ''}


def fig_learning_curves(data, spec):
    opt = spec['options']
    fig, ax = ps.new_figure(spec['layout'])
    m = opt['metric']
    for i, it in enumerate(spec['select']['runs']):
        col = ps.PALETTE[i]
        mean, lo, hi = seeds_curve(data, it['variant'], 'val_' + m)
        ep = np.arange(1, len(mean) + 1)
        if opt.get('show_val', True):
            ax.fill_between(ep, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(ep, smooth(mean, opt.get('smooth', 1)), color=col, label=it['label'] + ' (val)')
        if opt.get('show_train'):
            tm, _, _ = seeds_curve(data, it['variant'], m)
            ax.plot(ep, smooth(tm, opt.get('smooth', 1)), color=col, ls='--' if opt.get('show_val', True) else '-',
                    lw=1.1 if opt.get('show_val', True) else 1.6, label=it['label'] + ' (train)')
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel('epoch')
    ax.set_ylabel(m)
    if opt.get('logy'):
        ax.set_yscale('log')
    ax.set_title(opt.get('title', 'Learning curves: %s (mean of 3 seeds)' % m))
    ax.legend(fontsize=7)
    return fig, {'included': [r for it in spec['select']['runs'] for r in data.runs(it['variant'])], 'omitted': [],
                 'criterion': opt.get('criterion', 'the runs listed in the spec, all their seeds'), 'note': ''}


def fig_progress(data, spec, cfg):
    ph = cfg['phases']
    mr = [data.stat(p['variant'], 'mean_recall')[0] for p in ph]
    acc = [data.stat(p['variant'], 'val_acc')[0] for p in ph]
    xs = [p['phase'] for p in ph]
    fig, ax = ps.new_figure(spec['layout'])
    ax.plot(xs, mr, marker='o', color=ps.PALETTE[0], label='val mean recall')
    ax.plot(xs, acc, marker='o', color=ps.PALETTE[1], label='val accuracy')
    for x, v in zip(xs, mr):
        ax.annotate('%.3f' % v, (x, v), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7, color=ps.INK2)
    ax.set_xticks(xs)
    ax.set_xlabel('phase (0 = baseline)')
    ax.set_ylabel('validation metric')
    ax.set_title(spec.get('options', {}).get('title', 'Progress: validation metric after each phase'))
    ax.legend(fontsize=7)
    return fig, {'included': [p['variant'] for p in ph], 'omitted': [], 'criterion': 'winner of each phase (delivery.yaml)', 'note': ''}


def fig_ablation(data, spec, cfg):
    variants = {p['key']: p['variant'] for p in cfg['phases']}
    variants.update(cfg['variants'])
    items = [{'variant': variants['phase8'], 'label': 'combined (EXP-015_d4)', 'role': 'baseline'}] + \
            [{'variant': variants[a['variant']], 'label': a['label']} for a in cfg['ablation']]
    fig, ax = ps.new_figure(spec['layout'], 1.05)
    seed_bars(ax, data, items, 'mean_recall', 'train_mean_recall')
    ax.set_ylabel('mean recall (macro)')
    ax.set_title('Ablation of the combined model (3 seeds)')
    return fig, {'included': [it['variant'] for it in items], 'omitted': [],
                 'criterion': 'every ablation of EXP-017 + the no-augmentation run of EXP-015', 'note': ''}


def fig_per_class_recall(data, spec):
    sel = spec['select']
    names, rf, _ = data.recall_per_class(sel['final'])
    _, rb, _ = data.recall_per_class(sel['baseline'])
    fig, ax = ps.new_figure(spec['layout'], 1.05)
    xs = np.arange(len(names))
    w = 0.4
    ax.bar(xs - w / 2, rb, w * 0.9, color=ps.PALETTE[0], label='baseline (mean %.3f)' % np.mean(rb))
    ax.bar(xs + w / 2, rf, w * 0.9, color=ps.PALETTE[1], label='final, 3 seeds (mean %.3f)' % np.mean(rf))
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=40, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('validation recall')
    ax.set_title('Per-class recall (validation)')
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=7)
    return fig, {'included': [sel['baseline']] + data.runs(sel['final']), 'omitted': [], 'criterion': 'all 13 classes', 'note': ''}


def fig_confusion(data, spec):
    h = data.history(spec['select']['run'])
    cm = np.array(h['val']['cm'], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = h['class_names']
    fig, ax = ps.new_figure(spec['layout'], 1.9)
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Annotation')
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, '%.2f' % cmn[i, j], ha='center', va='center', fontsize=6,
                        color='white' if cmn[i, j] > 0.5 else ps.INK)
    ax.set_title('Confusion matrix (validation): %s' % spec['select']['run'])
    return fig, {'included': [spec['select']['run']], 'omitted': ['seeds 2 and 3 (one matrix per model)'],
                 'criterion': 'seed 1 = the model trained in ffNN_example.ipynb', 'note': ''}


def cfg_path(c, path):
    for k in path.split('.'):
        c = c[k]
    return c


def fig_group_sweep(data, spec):
    """Validation (solid) and train (dashed) metric vs a hyperparameter, one line per group (as plot_utils.plot_sweep)."""
    sel, opt = spec['select'], spec.get('options', {})
    runs = [r for r in data.rows if data.rows[r]['exp_id'] == sel['exp_id'] and not re.search(r'_seed\d+$', r)]
    groups = {}
    for r in runs:
        c = cfg_of(data, r)
        groups.setdefault(cfg_path(c, sel['group']), []).append((cfg_path(c, sel['x']), r))
    fig, ax = ps.new_figure(spec['layout'])
    for i, (g, pts) in enumerate(sorted(groups.items())):
        pts.sort()
        xs = [p[0] for p in pts]
        label = sel.get('group_labels', {}).get(g, g)
        ax.plot(xs, [float(data.rows[p[1]]['mean_recall']) for p in pts], marker='o', color=ps.PALETTE[i], label='%s val' % label)
        ax.plot(xs, [float(data.rows[p[1]]['train_mean_recall']) for p in pts], marker='o', mfc='white', ls='--', lw=1.1,
                color=ps.PALETTE[i], label='%s train' % label)
    if opt.get('logx'):
        ax.set_xscale('log')
    ax.set_xlabel(opt.get('xlabel', sel['x']))
    ax.set_ylabel('mean recall (macro)')
    ax.set_title(opt.get('title', ''))
    ax.legend(fontsize=7)
    return fig, {'included': sorted(runs), 'omitted': ['seed repetitions'] if any(re.search(r'_seed', r) for r in data.rows) else [],
                 'criterion': 'every run of %s (seed 1), one line per %s' % (sel['exp_id'], sel['group']), 'note': ''}


def fig_stopping(data, spec):
    """Why the stopping point: train vs validation accuracy (and validation mean recall) per epoch of one run, with the
    learning-rate reductions (dotted lines), the best epoch kept by the checkpoint (star) and the EarlyStopping epoch (x).
    The epochs after the best one are shaded: there only the training metric improves."""
    sel, opt = spec['select'], spec.get('options', {})
    r = sel['run']
    h = data.history(r)
    H = h['history']
    ep = np.arange(1, len(H['accuracy']) + 1)
    b = int(h['best_epoch'])
    fig, ax = ps.new_figure(spec['layout'])
    ax.axvspan(b, ep[-1], color=C['other'], alpha=0.35, lw=0, label='after the best epoch')
    lr = H['learning_rate']
    for k in range(1, len(lr)):
        if lr[k] < lr[k - 1] * 0.5:
            ax.axvline(ep[k] - 0.5, color=ps.INK2, ls=':', lw=1.0, label='lr / 10' if k == next(
                j for j in range(1, len(lr)) if lr[j] < lr[j - 1] * 0.5) else None)
    ax.plot(ep, H['accuracy'], color=ps.PALETTE[0], ls='--', lw=1.1, label='train accuracy')
    ax.plot(ep, H['val_accuracy'], color=ps.PALETTE[0], label='val accuracy')
    ax.plot(ep, H['val_mean_recall'], color=ps.PALETTE[1], label='val mean recall (monitored)')
    ax.plot(b, H['val_mean_recall'][b - 1], marker='*', markersize=12, color=ps.PALETTE[1], zorder=4, ls='none',
            label='best epoch (kept)')
    ax.plot(ep[-1], H['val_mean_recall'][-1], marker='x', markersize=8, color=ps.INK, zorder=4, ls='none',
            label='EarlyStopping')
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation / train metric')
    ax.set_title(opt.get('title', 'Stopping point: %s' % r))
    ax.legend(fontsize=6.5, loc='lower right')
    return fig, {'included': [r], 'omitted': ['seeds 2 and 3 (best epochs 31 and 32; same pattern)'],
                 'criterion': 'seed 1 of the final model = the run reproduced by ffNN_example.ipynb', 'note': ''}


def fig_activation(data, spec):
    """Two panels. Left (why): % of inactive units of the hidden layer (pre-activation <= 0 on every validation image;
    with ReLU, dead units) vs the input resolution, one line per learning rate (the runs of `sweep_exp`); hollow
    diamonds = best lr of each size. Right (what was tried): validation mean recall per activation (filled, mean +- std
    over seeds), training mean recall (hollow) and the inactive units of each model printed under it."""
    sel, opt = spec['select'], spec.get('options', {})
    fig, (a1, a2) = ps.plt.subplots(1, 2, figsize=opt.get('size', (8.4, 3.4)), gridspec_kw={'width_ratios': [1, 1.15]})
    # left: inactive units vs resolution, by lr
    pts = []
    for r in data.rows:
        if data.rows[r]['exp_id'] == sel['sweep_exp'] and data.inactive(r) is not None:
            c = cfg_of(data, r)
            pts.append((c['res'], c['optimizer']['lr'], 100 * data.inactive(r), float(data.rows[r]['mean_recall']), r))
    lrs = sorted({p[1] for p in pts}, reverse=True)
    lines = [lr for lr in lrs if sum(1 for p in pts if p[1] == lr) > 1]
    for i, lr in enumerate(lines):
        g = sorted(p for p in pts if p[1] == lr)
        a1.plot([p[0] for p in g], [p[2] for p in g], marker='o', color=ps.PALETTE[[0, 2, 3, 4][i]],
                label='lr ' + ('%.0e' % lr).replace('e-0', 'e-'))
    single = [p for p in pts if p[1] not in lines]
    if single:
        a1.plot([p[0] for p in single], [p[2] for p in single], ls='none', marker='o', color=C['baseline'], label='other lr')
    best = {}
    for p in pts:
        if p[0] not in best or p[3] > best[p[0]][3]:
            best[p[0]] = p
    multi = sorted(x for x in best if sum(1 for p in pts if p[0] == x) > 1)
    a1.plot(multi, [best[x][2] for x in multi], ls='none', marker='D', ms=9, mfc='none', mec=ps.INK, mew=1.1,
            label='best lr per size')
    xs = sorted({p[0] for p in pts})
    a1.set_xscale('log')
    a1.set_xticks(xs)
    a1.set_xticklabels([str(x) for x in xs])
    a1.minorticks_off()
    a1.set_ylim(0, 100)
    a1.set_xlabel('input resolution (side, log)')
    a1.set_ylabel('inactive hidden units (%)')
    a1.set_title(opt.get('title_left', 'Phase 2 (ReLU): inactive units'))
    a1.legend(fontsize=6.5, loc='lower right')
    # right: activations
    items = sel['variants']
    x = np.arange(len(items))
    for k, it in enumerate(items):
        col = C['baseline'] if it.get('role') == 'baseline' else C['final']
        m, s, n = data.stat(it['variant'], 'mean_recall')
        a2.errorbar(k, m, yerr=None if n == 1 else s, marker='o', ms=7, color=col, ecolor=ps.INK2, capsize=3, ls='none')
        a2.plot(k, data.stat(it['variant'], 'train_mean_recall')[0], marker='o', ms=7, mfc='white', color=col, ls='none')
        a2.text(k, m - 0.035, '%.3f' % m, ha='center', va='top', fontsize=6.5, color=ps.INK2)
        ina = data.inactive(it['variant'])
        a2.text(k, opt.get('inactive_row', 0.415), '%.0f%%' % (100 * ina) if ina is not None else '--', ha='center',
                va='center', fontsize=7, color=ps.INK, fontweight='bold' if it.get('role') == 'baseline' else 'normal')
        if it.get('role') == 'chosen':
            a2.axvspan(k - 0.4, k + 0.4, color=C['other'], alpha=0.35, lw=0)
    a2.text(-0.5, opt.get('inactive_row', 0.415) + 0.035, 'inactive units of the 1st hidden layer:', ha='left', va='center',
            fontsize=6.5, color=ps.INK2)
    a2.plot([], [], marker='o', ms=6, color=C['final'], ls='none', label='val (mean ± std over seeds)')
    a2.plot([], [], marker='o', ms=6, mfc='white', color=C['final'], ls='none', label='train')
    a2.plot([], [], marker='o', ms=6, color=C['baseline'], ls='none', label='ReLU (phase 3 model)')
    a2.set_xticks(x)
    a2.set_xticklabels([it['label'] for it in items], rotation=30, ha='right')
    a2.set_xlim(-0.6, len(items) - 0.4)
    a2.set_ylim(*opt.get('ylim', (0.38, 1.05)))
    a2.grid(axis='x', visible=False)
    a2.set_ylabel('mean recall (macro)')
    a2.set_title(opt.get('title_right', 'Phase 4: activation of the hidden layers'))
    a2.legend(fontsize=6.5, loc='upper left')
    fig.tight_layout()
    inc = sorted(p[4] for p in pts) + [r for it in items for r in data.runs(it['variant'])]
    return fig, {'included': inc, 'omitted': [],
                 'criterion': 'left: every run of %s (inactive units measured on the saved models); right: the '
                              'activations listed in figure_specs.yaml, every seed of each' % sel['sweep_exp'],
                 'note': ''}


TYPES = {'class_distribution': fig_class_distribution, 'sweep': fig_sweep, 'capacity': fig_capacity, 'strength': fig_strength,
         'bars': fig_bars, 'learning_curves': fig_learning_curves, 'per_class_recall': fig_per_class_recall,
         'confusion': fig_confusion, 'group_sweep': fig_group_sweep, 'stopping': fig_stopping, 'activation': fig_activation}
NEEDS_CFG = {'progress': fig_progress, 'ablation': fig_ablation}


def main(delivery_dir):
    cfg = yaml.safe_load(open(os.path.join(delivery_dir, 'delivery.yaml'), encoding='utf-8'))
    specs = yaml.safe_load(open(os.path.join(delivery_dir, 'figure_specs.yaml'), encoding='utf-8'))
    fig_dir = os.path.join(delivery_dir, 'overleaf', 'figures')
    gen_dir = os.path.join(delivery_dir, 'overleaf', 'generated')
    shutil.rmtree(fig_dir, ignore_errors=True)  # figures/ is always rebuilt from scratch
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    data = Data()
    ps.apply()
    log = ['# Figure log (generated by scripts/build_figures.py)', '']
    captions = ['% >>> ARCHIVO GENERADO por scripts/build_figures.py. Se sobrescribe siempre (se carga desde results_macros.tex). <<<',
                r'\renewcommand{\figcaption}[1]{\ifcsname figcap@#1\endcsname\csname figcap@#1\endcsname\else\textcolor{red}{\textbf{[??caption \detokenize{#1}]}}\fi}',
                '% \\figcaptionshort{id}: short caption for the slides (caption_slides_es; falls back to the long one)',
                r'\providecommand{\figcaptionshort}[1]{\ifcsname figcapshort@#1\endcsname\csname figcapshort@#1\endcsname\else\figcaption{#1}\fi}']
    for spec in specs:
        fn = NEEDS_CFG.get(spec['type'])
        fig, info = fn(data, spec, cfg) if fn else TYPES[spec['type']](data, spec)
        fig.savefig(os.path.join(fig_dir, '%s.png' % spec['id']), dpi=ps.DPI)
        ps.plt.close(fig)
        cap = tex(spec['caption_es']) + ((' ' + tex(info['note'])) if info.get('note') else '')
        captions.append(r'\expandafter\def\csname figcap@%s\endcsname{%s}' % (spec['id'], cap))
        if spec.get('caption_slides_es'):
            captions.append(r'\expandafter\def\csname figcapshort@%s\endcsname{%s}' % (spec['id'], tex(spec['caption_slides_es'])))
        log += ['## %s (%s)' % (spec['id'], spec['type']),
                '- Criterion: %s' % info.get('criterion', '-'),
                '- Included: %s' % (', '.join(info['included']) if isinstance(info['included'], list) else info['included']),
                '- Omitted: %s' % (', '.join(info['omitted']) if isinstance(info['omitted'], list) and info['omitted'] else (info['omitted'] or 'none')),
                '']
    with open(os.path.join(gen_dir, 'figure_captions.tex'), 'w', encoding='utf-8') as ofs:
        ofs.write('\n'.join(captions) + '\n')
    with open(os.path.join(delivery_dir, 'figure_log.md'), 'w', encoding='utf-8') as ofs:
        ofs.write('\n'.join(log))
    print('%d figures -> %s' % (len(specs), fig_dir))


if __name__ == '__main__':
    main(sys.argv[1])
