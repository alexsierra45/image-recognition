"""Re-select the best epoch of each run by val_accuracy instead of the metric it was trained with (DEC-019).

Reads experiments/history/<id>.json (per-epoch val_accuracy / val_mean_recall) and writes, per variant (run id without
_seedN) of the given phases, mean +- std over seeds of:
  - acc_at_acc, recall_at_acc: the epoch with the highest val_accuracy (the new criterion);
  - acc_at_rec, recall_at_rec: the epoch selected during training (val_mean_recall in delivery 3).
Post-hoc only: EarlyStopping and ReduceLROnPlateau still followed the training monitor, so a run re-trained with
monitor='val_accuracy' can stop at another epoch. Usage: python tools/reselect_by_accuracy.py [phase ...] (default d3).
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'experiments')


def per_run(run_id):
    h = json.load(open(os.path.join(EXP, 'history', run_id + '.json')))['history']
    acc, rec = np.array(h['val_accuracy']), np.array(h['val_mean_recall'])
    i_acc, i_rec = int(np.argmax(acc)), int(np.argmax(rec))
    return {'epoch_acc': i_acc + 1, 'acc_at_acc': acc[i_acc], 'recall_at_acc': rec[i_acc],
            'epoch_rec': i_rec + 1, 'acc_at_rec': acc[i_rec], 'recall_at_rec': rec[i_rec], 'epochs': len(acc)}


def main(phases):
    runs = pd.read_csv(os.path.join(EXP, 'experiments.csv')).drop_duplicates('id', keep='last')
    runs = runs[runs.phase.isin(phases) & ~runs.id.str.startswith('CHG-')]
    rows = [dict(id=r.id, exp_id=r.exp_id, variant=re.sub(r'_seed\d+$', '', r.id), **per_run(r.id)) for r in runs.itertuples()]
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(EXP, 'reselect_by_accuracy_runs.csv'), index=False, float_format='%.4f')
    metrics = ['acc_at_acc', 'recall_at_acc', 'acc_at_rec', 'recall_at_rec']
    agg = per.groupby(['exp_id', 'variant'])[metrics].agg(['mean', 'std'])
    agg.columns = ['%s_%s' % c for c in agg.columns]
    agg.insert(0, 'n_seeds', per.groupby(['exp_id', 'variant']).size())
    agg = agg.reset_index()
    agg.to_csv(os.path.join(EXP, 'reselect_by_accuracy.csv'), index=False, float_format='%.4f')
    for exp_id, g in agg.groupby('exp_id'):
        by_acc, by_rec = g.sort_values('acc_at_acc_mean', ascending=False), g.sort_values('recall_at_rec_mean', ascending=False)
        flag = '' if by_acc.variant.iloc[0] == by_rec.variant.iloc[0] else '   <-- the winner changes'
        print('\n== %s: best by accuracy %s, by mean recall %s%s' % (exp_id, by_acc.variant.iloc[0], by_rec.variant.iloc[0], flag))
        for r in by_acc.itertuples():
            print('  %-32s n=%d  acc %.4f (rec %.4f) | trained: rec %.4f (acc %.4f)'
                  % (r.variant, r.n_seeds, r.acc_at_acc_mean, r.recall_at_acc_mean, r.recall_at_rec_mean, r.acc_at_rec_mean))


if __name__ == '__main__':
    main(sys.argv[1:] or ['d3'])
