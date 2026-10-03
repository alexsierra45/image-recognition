"""Hyperparameter sweeps of the ffNN, one function per phase of the plan. Usage: python sweep.py phase1 [phase2 ...]

Each run is logged by experiment_utils.run_experiment (experiments/experiments.csv, history/, models/).
The base of each phase is fixed below after analysing the previous one; the reasoning is in LOGBOOK.md.
Runs whose id is already in experiments.csv are skipped (resume after an interruption).
"""
import os
import sys

import pandas as pd

from experiment_utils import CSV_PATH, DEFAULT_CFG, run_experiment


def merge_cfg(base, **changes):
    """dict(base, **changes), but 'optimizer' is merged into the default optimizer instead of replacing it."""
    cfg = dict(base, **changes)
    if 'optimizer' in changes:
        cfg['optimizer'] = dict(DEFAULT_CFG['optimizer'], **base.get('optimizer', {}), **changes['optimizer'])
    return cfg

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
    # The best size (32) was at the edge of the grid: bracket it.
    for res in [16, 48, 8, 24]:
        run('EXP-005_r%d' % res, 'EXP-005', 2, 'resolution %dx%d, Dense(256)' % (res, res), dict(PHASE2_BASE, res=res))
    # Robustness: train accuracy also falls with resolution at lr 1e-3 (optimisation, not only variance).
    # Does the ranking hold at lr 1e-4?
    # It does not (64 wins at 1e-4, 16 at 1e-3): compare each resolution at its own best lr (small joint grid).
    for res, lr in [(16, 1e-4), (32, 1e-4), (64, 1e-4), (16, 3e-3), (16, 3e-4), (32, 3e-4), (64, 3e-4), (64, 3e-5)]:
        tag = '%.0e' % lr
        tag = tag.replace('e-0', 'e-')
        run('EXP-005_r%d_lr%s' % (res, tag), 'EXP-005', 2, 'resolution %dx%d, Dense(256), lr %s' % (res, res, tag),
            merge_cfg(PHASE2_BASE, res=res, optimizer={'lr': lr}))


# EXP-005: best resolution 16x16 at lr 1e-3 (each size compared at its best lr). Second candidate: 32x32 at lr 3e-4.
PHASE3_BASE = dict(PHASE2_BASE, res=16)

PHASE3_ARCHS = [[64], [128], [256], [512], [1024],                      # 1 hidden layer
                [256, 128], [512, 128], [512, 256], [1024, 256],         # 2 hidden layers
                [512, 256, 128], [1024, 512, 256],                       # 3 hidden layers (funnel)
                [256, 256, 256, 256]]                                    # narrow and deep


def phase3():
    for hidden in PHASE3_ARCHS:
        tag = '-'.join(map(str, hidden))
        run('EXP-006_h%s' % tag, 'EXP-006', 3, 'hidden %s' % tag, dict(PHASE3_BASE, hidden=hidden))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        print('==== %s ====' % name, flush=True)
        globals()[name]()
