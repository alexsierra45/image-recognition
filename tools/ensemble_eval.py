"""Average of the predicted probabilities of several trained models (e.g. the 3 seeds of one variant) on validation.

Model averaging reduces the variance between runs [R8 §7.11]. No training: it loads experiments/models/<run>.keras,
predicts the validation (and training, for the gap) split of the same cache and split, averages the softmax outputs
and computes the same metrics as experiment_utils.run_experiment. Each member is also recomputed alone, to check it
matches its row in experiments.csv. Result: experiments/ensembles/<id>.json.

With --tta rot90, each model's prediction is itself the mean over the 4 rotations of 90 degrees of the image
(test-time augmentation, IDEA-012). With a single run, the result is that model with TTA.

Usage: python tools/ensemble_eval.py EXP-023_ens_EXP-021 EXP-021_final EXP-021_final_seed2 EXP-021_final_seed3
       python tools/ensemble_eval.py --tta rot90 EXP-026_tta_EXP-021_final EXP-021_final
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keras  # noqa: E402
from experiment_utils import CSV_PATH, EXP_DIR, MODELS_DIR, classification_metrics, get_data, predict_probs  # noqa: E402
from xview_data import CATEGORIES  # noqa: E402


def predict_tta(model, x, idx, tta):
    if not tta:
        return predict_probs(model, x, idx)
    assert tta == 'rot90', tta
    xs = np.asarray(x[idx], dtype=np.float32)
    return np.mean([model.predict(np.rot90(xs, k, axes=(1, 2)), batch_size=256, verbose=0) for k in range(4)], 0)


def main(ens_id, runs, tta=None):
    rows = pd.read_csv(CSV_PATH).set_index('id')
    res = {json.loads(rows.loc[r, 'config'])['res'] for r in runs}
    assert len(res) == 1, 'members with different input resolutions: %s' % res
    data = get_data(res.pop(), True)
    x, y, tr, va = data['x'], data['y'], data['train'], data['valid']
    members, pv, pt = [], [], []
    for r in runs:
        model = keras.models.load_model(os.path.join(MODELS_DIR, r + '.keras'))
        m = classification_metrics(y[va], predict_probs(model, x, va))   # the member alone, without TTA
        pv.append(predict_tta(model, x, va, tta))
        pt.append(predict_tta(model, x, tr, tta))
        members.append({'run': r, 'val_acc': m['acc'], 'mean_recall': m['mean_recall'], 'min_recall': m['min_recall'],
                        'csv_val_acc': float(rows.loc[r, 'val_acc']), 'csv_mean_recall': float(rows.loc[r, 'mean_recall'])})
        print('%-28s val acc %.4f (csv %.4f)  mean recall %.4f (csv %.4f)' % (
            r, m['acc'], rows.loc[r, 'val_acc'], m['mean_recall'], rows.loc[r, 'mean_recall']), flush=True)
    val = classification_metrics(y[va], np.mean(pv, 0))
    trn = classification_metrics(y[tr], np.mean(pt, 0))
    out = {'id': ens_id, 'members': members, 'tta': tta, 'combine': 'mean of softmax probabilities',
           'class_names': list(CATEGORIES.values()), 'val': val, 'train': {k: trn[k] for k in ('acc', 'mean_recall', 'recall')}}
    os.makedirs(os.path.join(EXP_DIR, 'ensembles'), exist_ok=True)
    path = os.path.join(EXP_DIR, 'ensembles', ens_id + '.json')
    with open(path, 'w', encoding='utf-8') as ofs:
        json.dump(out, ofs, indent=1)
    print('ENSEMBLE %s: val acc %.4f  mean recall %.4f  min recall %.4f  train acc %.4f -> %s' % (
        ens_id, val['acc'], val['mean_recall'], val['min_recall'], trn['acc'], path))


if __name__ == '__main__':
    args = sys.argv[1:]
    tta = None
    if args[0] == '--tta':
        tta, args = args[1], args[2:]
    main(args[0], args[1:], tta)
