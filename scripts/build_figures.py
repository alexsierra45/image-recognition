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
    ax.set_xlabel('training images')
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
        ax.fill_between(ep, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(ep, smooth(mean, opt.get('smooth', 1)), color=col, label=it['label'] + ' (val)')
        if opt.get('show_train'):
            tm, _, _ = seeds_curve(data, it['variant'], m)
            ax.plot(ep, smooth(tm, opt.get('smooth', 1)), color=col, ls='--', lw=1.1, label=it['label'] + ' (train)')
    ax.set_xlabel('epoch')
    ax.set_ylabel(m)
    ax.set_title(opt.get('title', 'Learning curves: %s (mean of 3 seeds)' % m))
    ax.legend(fontsize=7)
    return fig, {'included': [r for it in spec['select']['runs'] for r in data.runs(it['variant'])], 'omitted': [],
                 'criterion': 'reference without augmentation vs D4, 3 seeds each', 'note': ''}


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
    """Validation metric per epoch for each seed of a variant: star = best epoch (restored checkpoint),
    end of the line = epoch where EarlyStopping stopped."""
    sel, opt = spec['select'], spec.get('options', {})
    key = 'val_' + opt.get('metric', 'mean_recall')
    fig, ax = ps.new_figure(spec['layout'])
    runs = data.runs(sel['variant'])
    for i, r in enumerate(runs):
        h = data.history(r)
        y = h['history'][key]
        ep = np.arange(1, len(y) + 1)
        b = int(h['best_epoch']) - 1
        seed = cfg_of(data, r)['seed']
        ax.plot(ep, y, color=ps.PALETTE[i], lw=1.3, label='seed %d (best epoch %d, stop %d)' % (seed, b + 1, len(y)))
        ax.plot(ep[b], y[b], marker='*', markersize=11, color=ps.PALETTE[i], zorder=4)
        ax.plot(ep[-1], y[-1], marker='x', markersize=7, color=ps.PALETTE[i], zorder=4)
    ax.set_xlabel('epoch')
    ax.set_ylabel(key.replace('_', ' '))
    ax.set_title(opt.get('title', 'Stopping point: %s per epoch' % key.replace('_', ' ')))
    ax.legend(fontsize=7, loc='lower right')
    return fig, {'included': runs, 'omitted': [], 'criterion': 'every seed of %s; star = checkpoint kept, x = EarlyStopping' % sel['variant'],
                 'note': ''}


TYPES = {'class_distribution': fig_class_distribution, 'sweep': fig_sweep, 'capacity': fig_capacity, 'strength': fig_strength,
         'bars': fig_bars, 'learning_curves': fig_learning_curves, 'per_class_recall': fig_per_class_recall,
         'confusion': fig_confusion, 'group_sweep': fig_group_sweep, 'stopping': fig_stopping}
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
                r'\renewcommand{\figcaption}[1]{\ifcsname figcap@#1\endcsname\csname figcap@#1\endcsname\else\textcolor{red}{\textbf{[??caption \detokenize{#1}]}}\fi}']
    for spec in specs:
        fn = NEEDS_CFG.get(spec['type'])
        fig, info = fn(data, spec, cfg) if fn else TYPES[spec['type']](data, spec)
        fig.savefig(os.path.join(fig_dir, '%s.png' % spec['id']), dpi=ps.DPI)
        ps.plt.close(fig)
        cap = tex(spec['caption_es']) + ((' ' + tex(info['note'])) if info.get('note') else '')
        captions.append(r'\expandafter\def\csname figcap@%s\endcsname{%s}' % (spec['id'], cap))
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
