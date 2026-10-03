"""Hyperparameter sweeps of the ffNN, one function per phase of the plan. Usage: python sweep.py phase1 [phase2 ...]

Each run is logged by experiment_utils.run_experiment (experiments/experiments.csv, history/, models/).
The base of each phase is fixed below after analysing the previous one; the reasoning is in LOGBOOK.md.
Runs whose id is already in experiments.csv are skipped (resume after an interruption).
"""
import os
import sys

import pandas as pd

from experiment_utils import CSV_PATH, run_experiment

# Phase 0 (EXP-001): the assignment's model, raw pixels, original callbacks (ReduceLROnPlateau patience 10,
# EarlyStopping patience 40, checkpoint on val_accuracy), 20 epochs. Only seed + stratified split are new.
BASELINE = {'res': 224, 'norm': 'raw', 'input_relu': True, 'hidden': [], 'epochs': 20,
            'monitor': 'val_accuracy', 'es_patience': 40, 'rlr_patience': 10}


def done():
    return set(pd.read_csv(CSV_PATH)['id']) if os.path.exists(CSV_PATH) else set()


def run(run_id, exp_id, phase, change, cfg, notes=''):
    if run_id in done():
        print('skip', run_id, flush=True)
        return
    run_experiment(run_id, exp_id, phase, change, cfg, notes)


def phase0():
    run('EXP-001', 'EXP-001', 0, 'baseline (seed 1, stratified split)', BASELINE)


def phase1():
    # Cumulative ladder, one factor per step (LOGBOOK rule), so that the accumulated effect is also visible.
    b2 = dict(BASELINE, norm='scale')
    run('EXP-002', 'EXP-002', 1, '+ Rescaling(1/255)', b2)
    b3 = dict(b2, input_relu=False)
    run('CHG-008', 'CHG-008', 1, '- ReLU after Flatten (no-op check)', b3)
    fixed = dict(b3, epochs=60, es_patience=10, rlr_patience=4)
    for mon in ['val_loss', 'val_accuracy', 'val_mean_recall']:
        run('EXP-003_%s' % mon, 'EXP-003', 1, 'callbacks ES10/RLR4, 60 ep, monitor %s' % mon, dict(fixed, monitor=mon))
    # EXP-003 -> monitor val_mean_recall (best mean recall; it is also the decision metric)
    run('EXP-004', 'EXP-004', 1, 'per-channel standardisation (train mean/std)', dict(PHASE1_BEST, norm='standardize'))


# Phase 1 result (EXP-003_val_mean_recall): Rescaling(1/255), no input ReLU, ES 10 / RLR 4, 60 epochs, monitor val_mean_recall.
PHASE1_BEST = dict(BASELINE, norm='scale', input_relu=False, epochs=60, es_patience=10, rlr_patience=4, monitor='val_mean_recall')


# EXP-004: standardisation is worse than Rescaling(1/255) with lr 1e-3 -> keep 'scale'.
PHASE2_BASE = dict(PHASE1_BEST, hidden=[256])


def phase2():
    # Input resolution, same small MLP for every size: Flatten -> Dense(256, relu, he_normal) -> Dense(13).
    for res in [32, 64, 112, 224]:
        run('EXP-005_r%d' % res, 'EXP-005', 2, 'resolution %dx%d, Dense(256)' % (res, res), dict(PHASE2_BASE, res=res))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        print('==== %s ====' % name, flush=True)
        globals()[name]()
