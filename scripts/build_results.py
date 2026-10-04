"""Numbers -> LaTeX. Reads experiments/ (read-only) and writes <delivery>/overleaf/generated/:
results_macros.tex (\\res{key}), tables/*.tex (booktabs), meta.tex, and <delivery>/results_index.json (key -> source).

Usage: python scripts/build_results.py reports/delivery_01
"""
import csv
import datetime
import json
import math
import os
import re
import shutil
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'experiments')
# Class names are kept as in the dataset (English), the same as in the figures.

UNICODE_TEX = {'β': r'$\beta$', '₁': r'$_1$', '₂': r'$_2$', '×': r'$\times$', '°': r'$^\circ$', '±': r'$\pm$',
               '→': r'$\rightarrow$', '≈': r'$\approx$', '−': '--', 'α': r'$\alpha$', 'σ': r'$\sigma$'}


def tex(s):
    """Escape LaTeX specials and map the Unicode symbols used in labels to math (pdfLaTeX-safe)."""
    s = str(s)
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('$', r'\$')]:
        s = s.replace(a, b)
    for a, b in UNICODE_TEX.items():
        s = s.replace(a, b)
    return s.replace('$$', '')


def num(x, decimals):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '--'
    s = ('%.*f' % (decimals, abs(x))).replace('.', ',')
    return ('$-$' + s) if x < 0 and s.strip('0,') else s


def sci(x):
    """3.6e-3 -> '3,6\\cdot 10^{-3}' in math mode, Spanish comma."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    ms = ('%.1f' % m).rstrip('0').rstrip('.').replace('.', '{,}')
    return '$%s\\cdot 10^{%d}$' % (ms, e) if ms != '1' else '$10^{%d}$' % e


def hidden_tex(h):
    if not h:
        return 'ninguna'
    if len(set(h)) == 1:
        return '%d $\\times$ %d' % (len(h), h[0])
    return '--'.join(map(str, h))


def describe_model(c, n_classes):
    """Spanish one-line description of a ffNN config (pdfLaTeX-safe)."""
    act = {'relu': 'ReLU', 'leaky_relu': 'LeakyReLU(%s)' % num(c.get('leaky_alpha', 0.1), 1), 'elu': 'ELU', 'gelu': 'GELU',
           'selu': 'SELU', 'tanh': 'tanh'}[c['activation']]
    parts = ['Rescaling'] if c['norm'] == 'scale' else []
    parts.append('Flatten')
    if c.get('input_relu'):
        parts.append('ReLU')
    if c['hidden']:
        block = 'Dense' + (' $\\rightarrow$ BatchNorm' if c['bn'] == 'before' else '') + ' $\\rightarrow$ ' + act + \
                (' $\\rightarrow$ BatchNorm' if c['bn'] == 'after' else '') + \
                (' $\\rightarrow$ Dropout(%s)' % num(c['dropout'], 1) if c['dropout'] else '')
        parts.append('%s [%s]' % (hidden_tex(c['hidden']), block))
    parts.append('Dense(%d) $\\rightarrow$ softmax' % n_classes)
    return ' $\\rightarrow$ '.join(parts)


def describe_training(c):
    o = c['optimizer']
    name = {'adam': 'Adam' + (' (AMSGrad)' if o.get('amsgrad') else ''), 'adamw': 'AdamW', 'sgd': 'SGD', 'rmsprop': 'RMSprop'}[o['name']]
    extra = []
    if c.get('augment'):
        extra.append('augmentation: ' + ', '.join({'flip': 'volteos', 'rot90': 'rotaciones de 90$^\\circ$', 'color': 'brillo y contraste'}[a]
                                                  for a in c['augment']))
    if c.get('l2'):
        extra.append('L2 %s' % sci(c['l2']))
    if c.get('class_weight'):
        extra.append('pesos de clase (%s)' % c['class_weight'])
    return '%s, lr %s, batch %d, hasta %d épocas' % (name, sci(o['lr']), c['batch_size'], c['epochs']) + \
        ''.join('; ' + e for e in extra)


def thousands(n):
    return '{:,}'.format(int(n)).replace(',', '\\,')


class Data:
    def __init__(self):
        with open(os.path.join(EXP, 'experiments.csv'), newline='') as ifs:
            self.rows = {r['id']: r for r in csv.DictReader(ifs)}

    def runs(self, variant):
        """Run ids of a variant (the id itself + its _seedN repetitions), or an explicit list of run ids (for seed
        repetitions logged under another id, e.g. [EXP-008_sgdm_lr1e-2, EXP-009_sgd_m0.9_seed2, ...])."""
        if isinstance(variant, list):
            missing = [i for i in variant if i not in self.rows]
            if missing:
                raise KeyError('runs %s not in experiments.csv' % missing)
            return list(variant)
        ids = [i for i in self.rows if i == variant or re.fullmatch(re.escape(variant) + r'_seed\d+', i)]
        if not ids:
            raise KeyError('variant %s not in experiments.csv' % variant)
        return sorted(ids)

    def inactive(self, variant):
        """Mean over the runs of val.inactive_units (CHG-009), or None if a run has no value."""
        vals = [self.history(r)['val'].get('inactive_units') for r in self.runs(variant)]
        return None if any(v is None for v in vals) else sum(vals) / len(vals)

    def stat(self, variant, col):
        vals = [float(self.rows[i][col]) for i in self.runs(variant)]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else float('nan')
        return mean, std, len(vals)

    def history(self, run_id):
        with open(os.path.join(EXP, 'history', run_id + '.json')) as ifs:
            return json.load(ifs)

    def recall_per_class(self, variant):
        hs = [self.history(i) for i in self.runs(variant)]
        names = hs[0]['class_names']
        rec = [sum(h['val']['recall'][k] for h in hs) / len(hs) for k in range(len(names))]
        support = [sum(row) for row in hs[0]['val']['cm']]
        return names, rec, support


class Macros:
    def __init__(self, fmt):
        self.fmt, self.defs, self.index = fmt, {}, {}

    def add(self, key, value, source):
        assert re.fullmatch(r'[a-z0-9-]+', key), key
        self.defs[key] = value
        self.index[key] = {'value': value, **source}

    def write(self, path):
        lines = ['% >>> ARCHIVO GENERADO por scripts/build_results.py. Se sobrescribe siempre. <<<',
                 '% \\res{} is defined in common/results_interface.tex. Origin of every key: results_index.json.']
        for k in sorted(self.defs):
            lines.append(r'\expandafter\def\csname res@%s\endcsname{%s}' % (k, self.defs[k]))
        lines += ['% Figure captions (\\figcaption{id}), generated by scripts/build_figures.py',
                  r'\providecommand{\figcaption}[1]{\textcolor{red}{\textbf{[??caption \detokenize{#1}]}}}',
                  r'\InputIfFileExists{generated/figure_captions}{}{}']
        with open(path, 'w', encoding='utf-8') as ofs:
            ofs.write('\n'.join(lines) + '\n')


def table(path, header, rows, align, caption_note=None):
    out = ['% >>> ARCHIVO GENERADO por scripts/build_results.py (solo el tabular). Se sobrescribe siempre. <<<', r'\begin{tabular}{%s}' % align, r'\toprule',
           ' & '.join(header) + r' \\', r'\midrule']
    out += [' & '.join(r) + r' \\' for r in rows]
    out += [r'\bottomrule', r'\end{tabular}']
    with open(path, 'w', encoding='utf-8') as ofs:
        ofs.write('\n'.join(out) + '\n')


def pm(data, variant, col, d, ds):
    m, s, n = data.stat(variant, col)
    return num(m, d) if n == 1 else '%s $\\pm$ %s' % (num(m, d), num(s, ds))


def join_es(items):
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' y ' + items[-1]


TABLE_COLUMNS = {   # column id -> (header, alignment)
    'step': ('Paso', 'c'), 'label': ('Variante', 'l'), 'seeds': ('Semillas', 'c'), 'mean_recall': ('Mean recall', 'c'),
    'val_acc': ('Accuracy', 'c'), 'train_acc': ('Accuracy train', 'c'), 'train_mean_recall': ('Mean recall train', 'c'),
    'min_recall': ('Recall mín.', 'c'), 's_per_epoch': ('s/época', 'c'), 'epochs': ('Mejor época / épocas', 'c'),
    'inactive': ('Inactivas (\\%)', 'c'), 'params': ('Parámetros', 'r'), 'cap': ('Épocas máx.', 'c')}


def write_variant_tables(data, cfg, fmt, out):
    """delivery.yaml `tables`: one row per variant (mean +- std over its seeds), columns from TABLE_COLUMNS."""
    for t in cfg.get('tables', []):
        cols = t['columns']
        rows = []
        for r in t['rows']:
            v = r['variant']
            cells = []
            for c in cols:
                if c == 'step':
                    cells.append(tex(r.get('step', '')))
                elif c == 'label':
                    cells.append(tex(r['label']))
                elif c == 'seeds':
                    cells.append(str(len(data.runs(v))))
                elif c == 'epochs':
                    cells.append('%s / %s' % (num(data.stat(v, 'best_epoch')[0], 0), num(data.stat(v, 'actual_epochs')[0], 0)))
                elif c == 'inactive':
                    cells.append(num(100 * data.inactive(v), fmt.get('pct', 0)) if data.inactive(v) is not None else '--')
                elif c == 'params':
                    cells.append(thousands(data.stat(v, 'params')[0]))
                elif c == 'cap':
                    cells.append(str(json.loads(data.rows[data.runs(v)[0]]['config'])['epochs']))
                elif c == 's_per_epoch':
                    cells.append(num(data.stat(v, c)[0], 2))
                else:
                    cells.append(pm(data, v, c, fmt['metric'], fmt['std']))
            rows.append(cells)
        table(os.path.join(out, 'tables', t['name'] + '.tex'), [TABLE_COLUMNS[c][0] for c in cols], rows,
              t.get('align', ''.join(TABLE_COLUMNS[c][1] for c in cols)))


def history_value(data, spec):
    """delivery.yaml `history_values`: one number from the history of a run, or over the runs of an experiment.
    op: at (epoch, 1-based) | median_first (n epochs) | std_last (n epochs) | min / max of a val.* field over the runs
    whose id starts with `prefix`."""
    op, field = spec['op'], spec['field']
    if op in ('min', 'max'):
        runs = sorted(r for r in data.rows if r.startswith(spec['prefix']))
        vals = [data.history(r)['val'].get(field.split('.', 1)[1]) for r in runs]
        vals = [v for v in vals if v is not None]
        return (min if op == 'min' else max)(vals), {'runs': runs}
    h = data.history(spec['run'])['history'][field.split('.', 1)[1]]
    if op == 'at':
        return h[spec['epoch'] - 1], {'epoch': spec['epoch']}
    if op == 'median_first':
        s = sorted(h[:spec['n']])
        return (s[(len(s) - 1) // 2] + s[len(s) // 2]) / 2, {'epochs': '1-%d' % spec['n']}
    if op == 'std_last':
        s = h[-spec['n']:]
        m = sum(s) / len(s)
        return math.sqrt(sum((x - m) ** 2 for x in s) / (len(s) - 1)), {'epochs': 'last %d' % spec['n']}
    raise ValueError(op)


def write_ablation(data, cfg, variants, fmt, out):
    """Ablation table: each row removes one ingredient from the combined model (cfg['ablation_base'] key)."""
    base_key = cfg.get('ablation_base', 'phase8')
    base_m, _, _ = data.stat(variants[base_key], 'mean_recall')
    rows = [['Referencia (%s)' % tex(variants[base_key]), pm(data, variants[base_key], 'mean_recall', fmt['metric'], fmt['std']), '--']]
    for a in cfg['ablation']:
        m, _, _ = data.stat(variants[a['variant']], 'mean_recall')
        rows.append([tex(a['label']), pm(data, variants[a['variant']], 'mean_recall', fmt['metric'], fmt['std']),
                     num(100 * (m - base_m), fmt['pp'])])
    table(os.path.join(out, 'tables', 'ablation.tex'), ['Quitando', 'Mean recall', 'Diferencia (pp)'], rows, 'lcc',
          'Cada fila quita un solo ingrediente; 3 semillas por fila.')


def main(delivery_dir):
    cfg = yaml.safe_load(open(os.path.join(delivery_dir, 'delivery.yaml'), encoding='utf-8'))
    fmt = cfg['format']
    out = os.path.join(delivery_dir, 'overleaf', 'generated')
    shutil.rmtree(os.path.join(out, 'tables'), ignore_errors=True)  # generated/ is always rebuilt from scratch
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    data = Data()
    M = Macros(fmt)
    variants = {p['key']: p['variant'] for p in cfg['phases']}
    variants.update(cfg.get('variants', {}))
    variants['baseline'] = cfg['baseline_experiment']

    # Per-variant metrics
    for key, var in variants.items():
        src = {'source': 'experiments.csv', 'variant': var, 'runs': data.runs(var)}
        for col, name, d in [('mean_recall', 'mean-recall', fmt['metric']), ('val_acc', 'val-acc', fmt['metric']),
                             ('min_recall', 'min-recall', fmt['metric']), ('train_acc', 'train-acc', fmt['metric']),
                             ('train_mean_recall', 'train-mean-recall', fmt['metric']),
                             ('s_per_epoch', 's-per-epoch', fmt['seconds']), ('actual_epochs', 'actual-epochs', 0)]:
            m, s, n = data.stat(var, col)
            M.add('%s-%s' % (key, name), num(m, d), dict(src, column=col, stat='mean'))
            if n > 1 and col not in ('s_per_epoch', 'actual_epochs'):
                M.add('%s-%s-std' % (key, name), num(s, fmt['std']), dict(src, column=col, stat='std'))
        if data.inactive(var) is not None:   # % of units of the 1st hidden layer never active on validation (CHG-009)
            M.add('%s-inactive' % key, num(100 * data.inactive(var), fmt.get('pct', 0)),
                  dict(src, source='history/*.json', field='val.inactive_units', stat='mean x 100'))
        tr_m, _, _ = data.stat(var, 'train_acc')
        va_m, _, _ = data.stat(var, 'val_acc')
        M.add('%s-gap-acc' % key, num(100 * (tr_m - va_m), fmt['pp']), dict(src, formula='100*(train_acc-val_acc)', stat='mean'))
        m, _, n = data.stat(var, 'params')
        M.add('%s-params' % key, thousands(m), dict(src, column='params'))
        M.add('%s-seeds' % key, str(n), dict(src, column='id', stat='count'))
        c = json.loads(data.rows[data.runs(var)[0]]['config'])
        M.add('%s-res' % key, str(c['res']), dict(src, column='config.res'))
        M.add('%s-inputs' % key, thousands(3 * c['res'] ** 2), dict(src, column='config.res', formula='3 * res^2'))
        M.add('%s-lr' % key, sci(c['optimizer']['lr']), dict(src, column='config.optimizer.lr'))
        M.add('%s-batch' % key, str(c['batch_size']), dict(src, column='config.batch_size'))
        M.add('%s-hidden' % key, hidden_tex(c['hidden']), dict(src, column='config.hidden'))
        M.add('%s-n-hidden' % key, str(len(c['hidden'])), dict(src, column='config.hidden'))
        M.add('%s-width' % key, str(c['hidden'][0]) if c['hidden'] else '0', dict(src, column='config.hidden[0]'))
        for ck, name in [('epochs', 'epochs-cap'), ('es_patience', 'es-patience'), ('rlr_patience', 'rlr-patience')]:
            M.add('%s-%s' % (key, name), str(c[ck]), dict(src, column='config.' + ck))
        M.add('%s-beta1' % key, num(c['optimizer']['beta_1'], 2), dict(src, column='config.optimizer.beta_1'))
        if c['activation'] == 'leaky_relu':
            M.add('%s-leaky-alpha' % key, num(c.get('leaky_alpha', 0.1), 1), dict(src, column='config.leaky_alpha'))
        M.add('%s-dropout' % key, num(c['dropout'], 1), dict(src, column='config.dropout'))
        M.add('%s-l2' % key, sci(c['l2']) if c['l2'] else '0', dict(src, column='config.l2'))
        m, _, _ = data.stat(var, 'best_epoch')
        M.add('%s-best-epoch' % key, num(m, 0), dict(src, column='best_epoch', stat='mean'))

    for d in cfg.get('deltas', []):
        a, _, _ = data.stat(variants[d['a']], d['metric'])
        b, _, _ = data.stat(variants[d['b']], d['metric'])
        M.add(d['key'], num(100 * (a - b), fmt['pp']), {'source': 'experiments.csv', 'formula': '100*(%s-%s)' % (d['a'], d['b']),
                                                         'column': d['metric'], 'variants': [variants[d['a']], variants[d['b']]]})
    for r in cfg.get('ratios', []):
        a, _, _ = data.stat(variants[r['a']], r['metric'])
        b, _, _ = data.stat(variants[r['b']], r['metric'])
        M.add(r['key'], num(a / b, fmt['ratio']), {'source': 'experiments.csv', 'formula': '%s/%s' % (r['a'], r['b']),
                                                   'column': r['metric'], 'variants': [variants[r['a']], variants[r['b']]]})
    for hv in cfg.get('history_values', []):
        v, extra = history_value(data, hv)
        M.add(hv['key'], num(hv.get('scale', 1) * v, hv.get('decimals', fmt['metric'])),
              dict({'source': 'history/*.json', 'field': hv['field'], 'op': hv['op'], 'run': hv.get('run', hv.get('prefix'))}, **extra))
    for cf in cfg.get('confusions', []):   # share of the true class predicted as another one (final model, mean over seeds)
        hs = [data.history(r) for r in data.runs(variants['final'])]
        names = hs[0]['class_names']
        t, p = names.index(cf['actual']), names.index(cf['pred'])
        share = sum(h['val']['cm'][t][p] / sum(h['val']['cm'][t]) for h in hs) / len(hs)
        M.add('final-conf-%s-%s' % tuple(n.lower().replace(' ', '-') for n in (cf['actual'], cf['pred'])), num(share, fmt['metric']),
              {'source': 'history/*.json', 'variant': variants['final'], 'field': 'val.cm', 'stat': 'row share, mean over seeds'})

    # Seed noise: pooled standard deviation of the mean recall over every variant with >= 3 seeds
    groups = {}
    for i in data.rows:
        groups.setdefault(re.sub(r'_seed\d+$', '', i), []).append(float(data.rows[i]['mean_recall']))
    multi = [v for v in groups.values() if len(v) >= 3]
    pooled = math.sqrt(sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in multi) / sum(len(v) - 1 for v in multi))
    M.add('seed-noise-pp', num(100 * pooled, fmt['pp']), {'source': 'experiments.csv', 'stat': 'pooled std of mean_recall',
                                                          'n_variants': len(multi)})
    M.add('n-runs', str(len(data.rows)), {'source': 'experiments.csv', 'stat': 'row count'})
    M.add('n-variants-3seeds', str(len(multi)), {'source': 'experiments.csv', 'stat': 'variants with >= 3 seeds'})

    # Dataset (read-only from the annotations file) and split sizes (from the history of the baseline)
    ann = json.load(open(os.path.join(ROOT, 'xview_recognition', 'xview_ann_train.json')))
    counts = {}
    for a in ann['annotations'].values():
        counts[a['category_id']] = counts.get(a['category_id'], 0) + 1
    big = max(counts, key=counts.get)
    small = min(counts, key=counts.get)
    names, _, support = data.recall_per_class(cfg['baseline_experiment'])
    n_valid = sum(support)
    for k, v, s in [('n-images', thousands(sum(counts.values())), 'xview_ann_train.json'),
                    ('n-classes', str(len(counts)), 'xview_ann_train.json'),
                    ('n-valid', thousands(n_valid), 'confusion matrix of the baseline'),
                    ('n-train', thousands(sum(counts.values()) - n_valid), 'images - validation'),
                    ('class-max-name', big, 'xview_ann_train.json'), ('class-max-count', thousands(counts[big]), 'xview_ann_train.json'),
                    ('class-min-name', small, 'xview_ann_train.json'), ('class-min-count', thousands(counts[small]), 'xview_ann_train.json'),
                    ('imbalance-ratio', num(counts[big] / counts[small], 0), 'xview_ann_train.json'),
                    ('chance-mean-recall', num(1 / len(counts), fmt['metric']), '1 / number of classes'),
                    ('majority-acc', num(counts[big] / sum(counts.values()), fmt['metric']), 'largest class share'),
                    ('class-entropy', num(-sum(c / sum(counts.values()) * math.log(c / sum(counts.values())) for c in counts.values()), 2),
                     'entropy (nats) of the class frequencies = cross-entropy of a model that predicts the priors')]:
        M.add(k, v, {'source': s})
    n_test = len([f for f in os.listdir(os.path.join(ROOT, 'xview_recognition', 'xview_test')) if f.lower().endswith('.tif')])
    M.add('valid-share', num(100 * n_valid / sum(counts.values()), 0), {'source': 'validation / images, %'})
    M.add('n-test', thousands(n_test), {'source': 'xview_recognition/xview_test (*.tif)'})

    # Training loss of the baseline (median over epochs: the saturated softmax of EXP-000/EXP-001)
    h0 = data.history(data.runs(cfg['baseline_experiment'])[0])['history']['loss']
    M.add('baseline-train-loss', thousands(round(sorted(h0)[len(h0) // 2], -2)),
          {'source': 'history/%s.json' % cfg['baseline_experiment'], 'field': 'history.loss', 'stat': 'median over epochs, rounded to 100'})
    M.add('chance-loss', num(math.log(len(counts)), 1), {'source': 'ln(number of classes)'})

    # Per-class recall of the final model (mean over seeds)
    names, rec_f, support = data.recall_per_class(variants['final'])
    _, rec_b, _ = data.recall_per_class(cfg['baseline_experiment'])
    for n_, r in zip(names, rec_f):
        M.add('final-recall-%s' % n_.lower().replace(' ', '-'), num(r, fmt['metric']),
              {'source': 'history/*.json', 'variant': variants['final'], 'field': 'val.recall', 'class': n_})
    worst = min(range(len(names)), key=lambda k: rec_f[k])
    M.add('final-worst-class', names[worst], {'source': 'history/*.json', 'variant': variants['final']})
    M.add('final-worst-recall', num(rec_f[worst], fmt['metric']), {'source': 'history/*.json', 'variant': variants['final']})

    # Tables
    rows = []
    for p in cfg['phases']:
        rows.append([str(p['phase']), tex(p['label_es']), pm(data, p['variant'], 'mean_recall', fmt['metric'], fmt['std']),
                     pm(data, p['variant'], 'val_acc', fmt['metric'], fmt['std']), str(data.stat(p['variant'], 'mean_recall')[2])])
    table(os.path.join(out, 'tables', 'phase_summary.tex'), ['Fase', 'Cambio que se mantiene', 'Mean recall', 'Accuracy', 'Semillas'],
          rows, r'cp{0.50\linewidth}ccc', 'Métricas de validación; media $\\pm$ desviación típica cuando hay varias semillas.')
    rows = []
    for col, label in [('mean_recall', 'Mean recall (macro)'), ('val_acc', 'Accuracy'), ('min_recall', 'Recall mínimo por clase'),
                       ('train_acc', 'Accuracy en train')]:
        rows.append([label, pm(data, cfg['baseline_experiment'], col, fmt['metric'], fmt['std']),
                     pm(data, variants['final'], col, fmt['metric'], fmt['std'])])
    rows.append(['Parámetros', thousands(data.stat(cfg['baseline_experiment'], 'params')[0]), thousands(data.stat(variants['final'], 'params')[0])])
    table(os.path.join(out, 'tables', 'baseline_vs_final.tex'),
          ['', 'Punto de partida (%s)' % tex(cfg['baseline_experiment']), 'Modelo final (%s)' % tex(variants['final'])], rows, 'lcc')
    order = sorted(range(len(names)), key=lambda k: rec_f[k] - rec_b[k], reverse=True)
    rows = [[names[k], str(support[k]), num(rec_b[k], fmt['metric']), num(rec_f[k], fmt['metric']),
             ('+' if rec_f[k] >= rec_b[k] else '') + num(rec_f[k] - rec_b[k], fmt['metric'])] for k in order]
    table(os.path.join(out, 'tables', 'per_class_recall.tex'), ['Clase', 'Casos (val)', 'Baseline', 'Final', 'Diferencia'], rows, 'lrccc',
          'Recall de validación; el modelo final es la media de %d semillas.' % data.stat(variants['final'], 'mean_recall')[2])
    if cfg.get('ablation'):
        write_ablation(data, cfg, variants, fmt, out)
    write_variant_tables(data, cfg, fmt, out)

    # Keys used by the template (aliases of the final model and run count)
    for a, b in [('best-mean-recall', 'final-mean-recall'), ('best-mean-recall-std', 'final-mean-recall-std'),
                 ('best-val-acc', 'final-val-acc'), ('best-val-acc-std', 'final-val-acc-std'), ('n-experiments', 'n-runs')]:
        M.add(a, M.defs[b], dict(M.index[b], alias_of=b))

    # Baseline summary table (template: report_03_baseline.tex), described from the run config
    b = cfg['baseline_experiment']
    bc = json.loads(data.rows[data.runs(b)[0]]['config'])
    rows = [['Modelo', describe_model(bc, len(names))],
            ['Entrada', '%d $\\times$ %d $\\times$ 3, %s' % (bc['res'], bc['res'],
                        'píxeles en [0, 255]' if bc['norm'] == 'raw' else 'escalada a [0, 1]')],
            ['Parámetros', thousands(data.stat(b, 'params')[0])],
            ['Entrenamiento', describe_training(bc)],
            ['Mean recall (val)', pm(data, b, 'mean_recall', fmt['metric'], fmt['std'])],
            ['Accuracy (val)', pm(data, b, 'val_acc', fmt['metric'], fmt['std'])],
            ['Accuracy (train)', pm(data, b, 'train_acc', fmt['metric'], fmt['std'])],
            ['Recall mínimo por clase', pm(data, b, 'min_recall', fmt['metric'], fmt['std'])]]
    table(os.path.join(out, 'tables', 'baseline_summary.tex'), ['', 'Punto de partida (%s)' % tex(b)], rows, 'lp{0.62\\linewidth}')

    # Meta (macro names of the template: generated/meta.tex)
    date = cfg.get('date', 'auto')
    with open(os.path.join(out, 'meta.tex'), 'w', encoding='utf-8') as ofs:
        ofs.write('% >>> ARCHIVO GENERADO desde delivery.yaml. Se sobrescribe siempre. <<<\n')
        ofs.write('\\newcommand{\\DeliveryNumber}{%d}\n' % cfg['number'])
        ofs.write('\\newcommand{\\DeliveryTitle}{%s}\n' % tex(cfg['title_es']))
        ofs.write('\\newcommand{\\DeliveryShortTitle}{%s}\n' % tex(cfg.get('short_title_es', cfg['title_es'])))
        ofs.write('\\newcommand{\\DeliverySubtitle}{%s}\n' % tex(cfg.get('subtitle_es', 'Entrega \\DeliveryNumber')).replace('\\textbackslash{}', '\\'))
        ofs.write('\\newcommand{\\DeliveryDate}{%s}\n' % ('\\today' if date == 'auto' else tex(date)))
    # Stopping point of one run (delivery.yaml: stopping_run), for the stopping figure
    if cfg.get('stopping_run'):
        r = cfg['stopping_run']
        h = data.history(r)
        H = h['history']
        b = int(h['best_epoch'])
        src = {'source': 'history/%s.json' % r}
        drops = [k + 1 for k in range(1, len(H['learning_rate'])) if H['learning_rate'][k] < 0.5 * H['learning_rate'][k - 1]]
        M.add('stop-best-epoch', str(b), dict(src, field='best_epoch'))
        M.add('stop-epoch', str(len(H['accuracy'])), dict(src, field='len(history)'))
        M.add('stop-last-joint-epoch', str(drops[0] - 1), dict(src, field='last epoch before the first lr reduction'))
        M.add('stop-first-lr-drop', str(drops[0]), dict(src, field='first epoch with the reduced lr'))
        M.add('stop-n-lr-drops', str(len(drops)), dict(src, field='number of lr reductions'))
        M.add('stop-lr-drops', join_es(drops), dict(src, field='epochs with a reduced lr (first epoch of each)'))
        if len(drops) > 1:
            M.add('stop-later-lr-drops', join_es(drops[1:]), dict(src, field='lr reductions after the first one'))
        for name, i in [('best', b - 1), ('end', len(H['accuracy']) - 1)]:
            M.add('stop-train-acc-%s' % name, num(H['accuracy'][i], fmt['metric']), dict(src, field='history.accuracy[%d]' % i))
            M.add('stop-val-acc-%s' % name, num(H['val_accuracy'][i], fmt['metric']), dict(src, field='history.val_accuracy[%d]' % i))
            M.add('stop-gap-%s' % name, num(100 * (H['accuracy'][i] - H['val_accuracy'][i]), fmt['pp']),
                  dict(src, field='train - val accuracy at epoch %d, pp' % (i + 1)))
        M.add('stop-val-recall-best', num(H['val_mean_recall'][b - 1], fmt['metric']), dict(src, field='history.val_mean_recall'))
    M.write(os.path.join(out, 'results_macros.tex'))
    with open(os.path.join(delivery_dir, 'results_index.json'), 'w', encoding='utf-8') as ofs:
        json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'keys': M.index}, ofs, indent=1, ensure_ascii=False)
    print('%d macros, %d tables -> %s' % (len(M.defs), len(os.listdir(os.path.join(out, 'tables'))), out))


if __name__ == '__main__':
    main(sys.argv[1])
