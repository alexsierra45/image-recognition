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
        cfg['optimizer'] = {**DEFAULT_CFG['optimizer'], **base.get('optimizer', {}), **changes['optimizer']}
    return cfg

# Phase 0 (EXP-001): the assignment's model, raw pixels, original callbacks (ReduceLROnPlateau patience 10,
# EarlyStopping patience 40, checkpoint on val_accuracy), 20 epochs. Only seed + stratified split are new.
BASELINE = {'res': 224, 'norm': 'raw', 'input_relu': True, 'hidden': [], 'epochs': 20,
            'monitor': 'val_accuracy', 'es_patience': 40, 'rlr_patience': 10}


def done():
    return set(pd.read_csv(CSV_PATH)['id']) if os.path.exists(CSV_PATH) else set()


def run_isolated(run_id, exp_id, phase, change, cfg, notes=''):
    """As run(), but in a fresh Python process, so the GPU memory of a run is fully released before the next one
    (CHG-028: MobileNetV2 ran out of memory after 10 runs in one process). Same result as run(): same seed, same code."""
    if run_id in done():
        print('skip', run_id, flush=True)
        return
    import json
    import subprocess
    subprocess.run([sys.executable, os.path.abspath(__file__), '--one', json.dumps([run_id, exp_id, phase, change, cfg, notes])],
                   check=True)


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


# EXP-006: plateau at ~0.56-0.57 above ~200k params; 256-256-256-256 ties the best (0.572) with 3.6x fewer
# parameters than 1024-512-256 and a much smaller train-val gap.
PHASE4_BASE = dict(PHASE3_BASE, hidden=[256, 256, 256, 256])

PHASE4_ACTS = [('leaky0.1', {'activation': 'leaky_relu', 'leaky_alpha': 0.1}),     # he_normal
               ('leaky0.3', {'activation': 'leaky_relu', 'leaky_alpha': 0.3}),     # he_normal
               ('elu', {'activation': 'elu'}),                                    # he_normal
               ('gelu', {'activation': 'gelu'}),                                  # he_normal
               ('selu', {'activation': 'selu'}),                                  # lecun_normal, AlphaDropout if dropout
               ('tanh', {'activation': 'tanh'})]                                  # glorot_uniform


def phase4():
    # Only the hidden activation changes; the ReLU point is the phase-3 winner itself (same config, same seed).
    for tag, change in PHASE4_ACTS:
        run('EXP-007_%s' % tag, 'EXP-007', 4, 'activation %s' % tag, dict(PHASE4_BASE, **change))


# EXP-007: LeakyReLU(0.3) best on mean recall (0.591) and accuracy (0.635). Phase 5 explores with short runs (<= 20 epochs).
PHASE5_BASE = dict(PHASE4_BASE, activation='leaky_relu', leaky_alpha=0.3, epochs=20)


def lr_tag(lr):
    return ('%.0e' % lr).replace('e-0', 'e-')


def opt_run(run_id, exp_id, change, base, optimizer=None, **cfg):
    """Phase 5 helper: optimizer dict merged on top of the base optimizer."""
    run(run_id, exp_id, 5, change, merge_cfg(base, optimizer=optimizer or {}, **cfg))


def phase5_1():
    # Learning rate. Short exploration runs (max 20 epochs, as the plan asks); the lr 1e-3 Adam point is the reference.
    for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        opt_run('EXP-008_adam_lr%s' % lr_tag(lr), 'EXP-008', 'Adam lr %s' % lr_tag(lr), PHASE5_BASE, {'lr': lr})
    for lr in [1e-3, 1e-2, 1e-1]:
        opt_run('EXP-008_sgdm_lr%s' % lr_tag(lr), 'EXP-008', 'SGD momentum 0.9 lr %s' % lr_tag(lr), PHASE5_BASE,
                {'name': 'sgd', 'lr': lr, 'momentum': 0.9})


# EXP-008 (<= 20 epochs): SGD + momentum 0.9 at lr 1e-2 is best (0.582); Adam is flat within noise, best at lr 3e-4 (0.535).
SGD_LR, ADAM_LR = 1e-2, 3e-4
PHASE5_2_BASE = merge_cfg(PHASE5_BASE, optimizer={'lr': ADAM_LR})


def phase5_2():
    # Momentum. SGD at its best lr from 5.1 (momentum 0.9 without Nesterov is already in 5.1); Adam betas; RMSprop lr.
    b = PHASE5_2_BASE
    for m in [0.0, 0.5, 0.95, 0.99]:
        opt_run('EXP-009_sgd_m%s' % m, 'EXP-009', 'SGD lr %s momentum %s' % (lr_tag(SGD_LR), m), b,
                {'name': 'sgd', 'lr': SGD_LR, 'momentum': m})
    for m in [0.5, 0.9, 0.95, 0.99]:
        opt_run('EXP-009_sgd_m%s_nesterov' % m, 'EXP-009', 'SGD lr %s momentum %s Nesterov' % (lr_tag(SGD_LR), m), b,
                {'name': 'sgd', 'lr': SGD_LR, 'momentum': m, 'nesterov': True})
    for b1 in [0.8, 0.95]:
        opt_run('EXP-009_adam_b1_%s' % b1, 'EXP-009', 'Adam beta_1 %s' % b1, b, {'beta_1': b1})
    opt_run('EXP-009_adam_b2_0.99', 'EXP-009', 'Adam beta_2 0.99', b, {'beta_2': 0.99})
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        opt_run('EXP-009_rmsprop_lr%s' % lr_tag(lr), 'EXP-009', 'RMSprop lr %s' % lr_tag(lr), b,
                {'name': 'rmsprop', 'lr': lr, 'momentum': 0.0})


def phase5_2_seeds():
    # Single-seed differences in 5.2 are non-monotonic (noise ~2-3 pp?): repeat the two finalists with seeds 2 and 3.
    for seed in [2, 3]:
        opt_run('EXP-009_sgd_m0.9_seed%d' % seed, 'EXP-009', 'SGD lr 1e-2 momentum 0.9, seed %d' % seed, PHASE5_2_BASE,
                {'name': 'sgd', 'lr': SGD_LR, 'momentum': 0.9}, seed=seed)
        opt_run('EXP-009_adam_b1_0.95_seed%d' % seed, 'EXP-009', 'Adam beta_1 0.95, seed %d' % seed, PHASE5_2_BASE,
                {'beta_1': 0.95}, seed=seed)


# EXP-009: seed noise ~1-2 pp. SGD+m 0.9 (0.565 +- 0.018, 3 seeds) and Adam beta_1 0.95 (0.571 +- 0.009) tie;
# Adam is kept: lower seed variance, flat lr response in 5.1, and phase 5.5 is about Adam options.
PHASE5_3_BASE = merge_cfg(PHASE5_2_BASE, optimizer={'beta_1': 0.95})


def phase5_3():
    # Batch size, lr rescaled with the square-root rule (and the linear rule for two sizes).
    b = PHASE5_3_BASE
    lr0, bs0 = b['optimizer']['lr'], b.get('batch_size', DEFAULT_CFG['batch_size'])
    for bs in [32, 64, 128, 256]:
        lr = lr0 * (bs / bs0) ** 0.5
        opt_run('EXP-010_bs%d_sqrt' % bs, 'EXP-010', 'batch %d, lr %.2g (sqrt rule)' % (bs, lr), b, {'lr': lr}, batch_size=bs)
    for bs in [64, 256]:
        lr = lr0 * bs / bs0
        opt_run('EXP-010_bs%d_linear' % bs, 'EXP-010', 'batch %d, lr %.2g (linear rule)' % (bs, lr), b, {'lr': lr}, batch_size=bs)


# EXP-010: every batch size within seed noise (0.552-0.574); batch 256 with lr 1.2e-3 (sqrt rule) is 15x faster per epoch.
# Runs now take < 1 min, so the short exploration budget is dropped: back to 60 epochs (EarlyStopping 10).
PHASE5_4_BASE = merge_cfg(PHASE5_3_BASE, optimizer={'lr': 3e-4 * (256 / 16) ** 0.5}, batch_size=256, epochs=60)


def phase5_4():
    # Reference with the full budget first (only the epoch budget changes vs EXP-010_bs256_sqrt).
    run('EXP-010_bs256_sqrt_60ep', 'EXP-010', 5, 'batch 256, lr 0.0012, 60 epochs (new base)', PHASE5_4_BASE)
    # Learning-rate schedule (the reference is ReduceLROnPlateau, already in the base).
    b = PHASE5_4_BASE
    lr = b['optimizer']['lr']  # PHASE5_4_BASE is built with merge_cfg, so it always has 'optimizer'
    opt_run('EXP-011_cosine', 'EXP-011', 'cosine decay', b, schedule='cosine')
    opt_run('EXP-011_exponential', 'EXP-011', 'exponential decay 0.9/epoch', b, schedule='exponential', exp_decay=0.9)
    opt_run('EXP-011_warmup_cosine_2x', 'EXP-011', 'warmup 3 ep + cosine, lr x2', b, {'lr': 2 * lr}, schedule='cosine', warmup_epochs=3)
    opt_run('EXP-011_warmup_plateau_2x', 'EXP-011', 'warmup 3 ep + plateau, lr x2', b, {'lr': 2 * lr}, warmup_epochs=3)


def phase5_4_check():
    # Warmup + 2x lr won: separate the two factors, and use 3 seeds now that a run takes ~25 s.
    b = PHASE5_4_BASE
    lr = b['optimizer']['lr']
    opt_run('EXP-011_plateau_2x', 'EXP-011', 'plateau, lr x2 (no warmup)', b, {'lr': 2 * lr})
    for seed in [2, 3]:
        run('EXP-010_bs256_sqrt_60ep_seed%d' % seed, 'EXP-010', 5, 'batch 256, lr 0.0012, 60 epochs, seed %d' % seed,
            dict(PHASE5_4_BASE, seed=seed))
        opt_run('EXP-011_warmup_plateau_2x_seed%d' % seed, 'EXP-011', 'warmup 3 ep + plateau, lr x2, seed %d' % seed, b,
                {'lr': 2 * lr}, warmup_epochs=3, seed=seed)


# EXP-011 (3 seeds): no schedule beats ReduceLROnPlateau beyond noise (warmup + plateau x2: 0.584 +- 0.028 vs base
# 0.578 +- 0.007). Keep the base: plateau, lr 1.2e-3, no warmup.
PHASE5_5_BASE = PHASE5_4_BASE
SEEDS = [1, 2, 3]


def seeded(run_id, seed):
    return run_id if seed == 1 else '%s_seed%d' % (run_id, seed)


def phase5_5():
    # Other optimizer settings: clipnorm, AMSGrad, decoupled weight decay. 3 seeds each (runs take ~25 s);
    # the base with seeds 1-3 is EXP-010_bs256_sqrt_60ep[_seed2|_seed3].
    b = PHASE5_5_BASE
    for seed in SEEDS:
        opt_run(seeded('EXP-012_no_clipnorm', seed), 'EXP-012', 'no clipnorm', b, {'clipnorm': 0.0}, seed=seed)
        opt_run(seeded('EXP-012_no_amsgrad', seed), 'EXP-012', 'no AMSGrad', b, {'amsgrad': False}, seed=seed)
        for wd in [1e-4, 1e-3, 1e-2]:
            opt_run(seeded('EXP-012_adamw_wd%s' % lr_tag(wd), seed), 'EXP-012', 'AdamW weight decay %s' % lr_tag(wd), b,
                    {'name': 'adamw', 'weight_decay': wd}, seed=seed)


def phase5_check():
    # End of phase 5: is the new base (batch 256, 0.578 +- 0.007) really comparable to EXP-007 (0.591, one seed)?
    for seed in [2, 3]:
        run('EXP-007_leaky0.3_seed%d' % seed, 'EXP-007', 4, 'activation leaky0.3, seed %d' % seed,
            dict(PHASE4_BASE, activation='leaky_relu', leaky_alpha=0.3, seed=seed))



# EXP-012 (3 seeds): clipnorm, AMSGrad and AdamW weight decay change nothing beyond noise. Phase 5 result = PHASE5_5_BASE.
# Regularisation slows training down (dropout 0.2 peaked at epoch 60/60): cap raised to 150 epochs. The base is unchanged
# by this (its 3 seeds were stopped by EarlyStopping before epoch 45, and runs are deterministic).
PHASE6_BASE = dict(PHASE5_5_BASE, epochs=150)
PHASE6_VARIANTS = [('dropout0.1', {'dropout': 0.1}), ('dropout0.2', {'dropout': 0.2}), ('dropout0.3', {'dropout': 0.3}),
                   ('dropout0.5', {'dropout': 0.5}), ('l2_1e-5', {'l2': 1e-5}), ('l2_1e-4', {'l2': 1e-4}),
                   ('l2_1e-3', {'l2': 1e-3}), ('bn_before', {'bn': 'before'}), ('bn_after', {'bn': 'after'})]


def phase6():
    # Regularisation, one technique at a time, 3 seeds each. Base with seeds 1-3: EXP-010_bs256_sqrt_60ep[_seed2|_seed3].
    for seed in SEEDS:
        for tag, change in PHASE6_VARIANTS:
            run(seeded('EXP-013_%s' % tag, seed), 'EXP-013', 6, '%s' % tag, dict(PHASE6_BASE, seed=seed, **change))




def phase6_bn_lr():
    # BatchNorm tied with the base at the same lr; it usually tolerates a larger lr: try lr x3.
    for seed in SEEDS:
        for bn in ['before', 'after']:
            opt_run(seeded('EXP-013_bn_%s_lr3x' % bn, seed), 'EXP-013', 'bn_%s, lr x3' % bn, dict(PHASE6_BASE, bn=bn, seed=seed),
                    {'lr': 3 * PHASE6_BASE['optimizer']['lr']})


# EXP-013 (3 seeds): dropout and L2 lower train AND validation (underfitting side); BN at the same lr ties the base.
# BN before the activation with lr x3 (3.6e-3): 0.590 +- 0.009 vs 0.578 +- 0.007 -> adopted (marginal, t ~ 2).
# Reference for phase 7 (3 seeds): EXP-013_bn_before_lr3x[_seed2|_seed3].
PHASE7_BASE = merge_cfg(PHASE6_BASE, bn='before', optimizer={'lr': 3 * PHASE6_BASE['optimizer']['lr']})

PHASE7_VARIANTS = [('cw_inverse', {'class_weight': 'inverse'}), ('cw_sqrt', {'class_weight': 'sqrt'}),
                   ('oversample_balanced', {'oversample': 1.0}), ('oversample_sqrt', {'oversample': 0.5})]


def phase7():
    # Class imbalance, 3 seeds each. Base = phase-6 winner (PHASE7_BASE, 3 seeds already run in phase 6).
    for seed in SEEDS:
        for tag, change in PHASE7_VARIANTS:
            run(seeded('EXP-014_%s' % tag, seed), 'EXP-014', 7, tag, dict(PHASE7_BASE, seed=seed, **change))


# EXP-014 (3 seeds): every imbalance treatment adds ~3 pp of mean recall (0.619-0.625 vs 0.590 +- 0.009), tied among them.
# Square-root class weights kept: same mean recall as inverse weights (0.622 +- 0.005 vs 0.625 +- 0.010) without the
# accuracy drop (0.614 vs 0.588).
PHASE8_BASE = dict(PHASE7_BASE, class_weight='sqrt')

PHASE8_VARIANTS = [('flip', ['flip']), ('rot90', ['rot90']), ('d4', ['flip', 'rot90']), ('color', ['color']),
                   ('d4_color', ['flip', 'rot90', 'color'])]


def phase8():
    # Augmentation (train only), 3 seeds each. Augmentation needs more epochs: patience 15 / 6 for every run of the
    # phase, including a no-augmentation reference with the same patience (so only the augmentation changes).
    b = dict(PHASE8_BASE, es_patience=15, rlr_patience=6)
    for seed in SEEDS:
        run(seeded('EXP-015_none_patience15', seed), 'EXP-015', 8, 'no augmentation, patience 15/6', dict(b, seed=seed))
        for tag, ops in PHASE8_VARIANTS:
            run(seeded('EXP-015_%s' % tag, seed), 'EXP-015', 8, 'augmentation %s' % tag, dict(b, seed=seed, augment=ops))



# EXP-015 (3 seeds): D4 augmentation 0.656 +- 0.006 vs 0.615 +- 0.007 without; photometric jitter hurts.
FINAL_BASE = dict(PHASE8_BASE, es_patience=15, rlr_patience=6, augment=['flip', 'rot90'])
# Reference with seeds 1-3: EXP-015_d4[_seed2|_seed3].


def phase9_tune():
    # Fine-tuning around the combined model and re-checks of earlier choices that augmentation may have changed.
    lr = FINAL_BASE['optimizer']['lr']
    variants = [('lr_x0.5', merge_cfg(FINAL_BASE, optimizer={'lr': lr / 2})),
                ('lr_x2', merge_cfg(FINAL_BASE, optimizer={'lr': lr * 2})),
                ('dropout0.1', dict(FINAL_BASE, dropout=0.1)),
                ('hidden_512x4', dict(FINAL_BASE, hidden=[512] * 4)),
                ('res32', dict(FINAL_BASE, res=32))]
    for seed in SEEDS:
        for tag, cfg in variants:
            run(seeded('EXP-016_%s' % tag, seed), 'EXP-016', 9, tag, dict(cfg, seed=seed))


def phase9_batch16():
    # IDEA-011: small batch on the final configuration (lr rescaled with the square-root rule). ~10 min per run.
    lr = FINAL_BASE['optimizer']['lr'] * (16 / 256) ** 0.5
    for seed in SEEDS:
        run(seeded('EXP-016_batch16', seed), 'EXP-016', 9, 'batch 16, lr %.2g' % lr,
            dict(merge_cfg(FINAL_BASE, optimizer={'lr': lr}, batch_size=16), seed=seed))


def phase9_ablation():
    # Remove one ingredient at a time from the combined model (no D4 = EXP-015_none_patience15, already run).
    variants = [('no_class_weight', dict(FINAL_BASE, class_weight=None)),
                ('no_bn', merge_cfg(dict(FINAL_BASE, bn=None), optimizer={'lr': FINAL_BASE['optimizer']['lr'] / 3})),
                ('relu', dict(FINAL_BASE, activation='relu')),
                ('one_layer_256', dict(FINAL_BASE, hidden=[256])),
                ('adam_beta1_0.9', merge_cfg(FINAL_BASE, optimizer={'beta_1': 0.9}))]
    for seed in SEEDS:
        for tag, cfg in variants:
            run(seeded('EXP-017_%s' % tag, seed), 'EXP-017', 9, 'ablation: %s' % tag, dict(cfg, seed=seed))



# EXP-016: nothing improves on the combined model (lr x0.5/x2 equal; dropout, 512x4, 32x32, batch 16 worse).
# EXP-017 ablation: depth -8.0 pp, BN -5.3, class weights -5.0, D4 -4.1, beta_1 -1.5; ReLU instead of LeakyReLU ties
# (0.652 +- 0.004 vs 0.656 +- 0.006) with better accuracy and min recall -> the final model uses ReLU.
FINAL = dict(FINAL_BASE, activation='relu', es_patience=25, rlr_patience=8, epochs=300)


def phase9_final():
    # Final model trained longer (patience 25 / 8, up to 300 epochs), 3 seeds. Seed 1 is the one rebuilt in ffNN_example.ipynb.
    for seed in SEEDS:
        run(seeded('EXP-018_final', seed), 'EXP-018', 9, 'final model (ReLU), patience 25/8', dict(FINAL, seed=seed))



# ---------------------------------------------------------------- delivery 2: regularisation (DEC-009)
# Base: winner of phase 6 (EXP-013_bn_before_lr3x: BatchNorm before the activation, lr 3.6e-3), WITHOUT class weights
# (class imbalance moves to delivery 5). Augmentation needs more epochs: patience 15 / 6 for every run, including the
# reference without augmentation, so that only the augmentation changes.
D2_BASE = dict(PHASE7_BASE, es_patience=15, rlr_patience=6)


def d2_augmentation():
    for seed in SEEDS:
        run(seeded('EXP-019_none_patience15', seed), 'EXP-019', 'd2', 'no augmentation, patience 15/6 (BN, no class weights)',
            dict(D2_BASE, seed=seed))
        for tag, ops in PHASE8_VARIANTS:
            run(seeded('EXP-019_%s' % tag, seed), 'EXP-019', 'd2', 'augmentation %s (BN, no class weights)' % tag,
                dict(D2_BASE, seed=seed, augment=ops))



# EXP-019 (3 seeds): geometric augmentation regularises (train acc 0.82 -> ~0.69, val acc 0.615 -> ~0.63); rot90 has the
# best mean recall (0.617 +- 0.015), tied with D4 (0.606 +- 0.008) and flips (0.609); photometric jitter hurts. rot90 kept.
D2_AUG = dict(D2_BASE, augment=['rot90'])


def d2_combine():
    # Explicit penalties on top of BN + rot90 (does any help now that augmentation closed part of the gap?) and the
    # contribution of BatchNorm in this setting (without BN the lr goes back to the phase-5 value, 1.2e-3).
    variants = [('dropout0.1', dict(D2_AUG, dropout=0.1)), ('dropout0.2', dict(D2_AUG, dropout=0.2)),
                ('l2_1e-5', dict(D2_AUG, l2=1e-5)), ('l2_1e-4', dict(D2_AUG, l2=1e-4)),
                ('no_bn', merge_cfg(dict(D2_AUG, bn=None), optimizer={'lr': D2_AUG['optimizer']['lr'] / 3}))]
    for seed in SEEDS:
        for tag, cfg in variants:
            run(seeded('EXP-020_%s' % tag, seed), 'EXP-020', 'd2', 'rot90 + BN + %s' % tag if tag != 'no_bn' else 'rot90 without BN (lr 1.2e-3)',
                dict(cfg, seed=seed))



# EXP-020 (3 seeds): no penalty helps on top of BN + rot90 (dropout 0.1/0.2: 0.577/0.550; L2 1e-5/1e-4: 0.603/0.608 vs
# 0.617 +- 0.015); without BN 0.569 (-4.8 pp). Delivery-2 model = BN + rot90, trained longer as in EXP-018.
D2_FINAL = dict(D2_AUG, es_patience=25, rlr_patience=8, epochs=300)


def d2_final():
    for seed in SEEDS:
        run(seeded('EXP-021_final', seed), 'EXP-021', 'd2', 'delivery-2 model (BN + rot90), patience 25/8', dict(D2_FINAL, seed=seed))


# EXP-021 (3 seeds): 0.646 +- 0.015 mean recall, 0.643 +- 0.011 val acc; train-val gap 12.1 pp. Delivery-2 model.


def d2_label_smoothing():
    # Output regularisation not tried yet: soften the one-hot targets (Szegedy et al. 2016; Mueller et al. 2019).
    # Base EXP-021 (D2_FINAL); only epsilon changes.
    for seed in SEEDS:
        for eps in (0.05, 0.1, 0.2):
            run(seeded('EXP-022_ls%s' % eps, seed), 'EXP-022', 'd2', 'label smoothing %s on the delivery-2 model' % eps,
                dict(D2_FINAL, seed=seed, label_smoothing=eps))


# EXP-022 (3 seeds): label smoothing 0.05/0.1/0.2 -> 0.603/0.617/0.593 (val acc unchanged, Helipad loses most). Discarded.
# EXP-023: average of the 3 seeds of EXP-021 -> 0.652 (+0.6 pp, not conclusive).


D2_WIDTHS = (256, 512, 1024)
D2_REGS = [('none', {}), ('dropout0.2', {'dropout': 0.2}), ('dropout0.5', {'dropout': 0.5}), ('l2_1e-4', {'l2': 1e-4})]


def d2_capacity_reg():
    # Capacity x regularisation (DEC-013): Dropout and L2 were judged only on 4 x 256, a network chosen in delivery 1
    # partly because larger ones memorised. Dropout is meant for networks with capacity to spare (Srivastava et al.).
    # Two factors ON PURPOSE: the interaction is what is measured. Base EXP-021 (D2_FINAL); 4 hidden layers, only the
    # width and the regulariser change. w256_none = EXP-021_final (not re-run).
    for seed in SEEDS:
        for w in D2_WIDTHS:
            for tag, reg in D2_REGS:
                if w == 256 and tag == 'none':
                    continue
                run(seeded('EXP-024_w%d_%s' % (w, tag), seed), 'EXP-024', 'd2', 'width %d x 4, %s (BN + rot90, patience 25/8)' % (w, tag),
                    dict(D2_FINAL, seed=seed, hidden=[w] * 4, **reg))


# EXP-024 (3 seeds): the cost of Dropout shrinks with the width (0.2: -8.5 / -5.0 / -1.6 pp at 256 / 512 / 1024), but no
# cell beats EXP-021 (best: 512 + L2 1e-4, 0.645 +- 0.003; 512 alone 0.644). Kept: 4 x 256, no Dropout, no L2.


def d2_shift():
    # Random translations (IDEA-013): the invariance a dense network lacks most. Base EXP-021; only the shift is added.
    for seed in SEEDS:
        for k in (1, 2):
            run(seeded('EXP-025_shift%d' % k, seed), 'EXP-025', 'd2', 'rot90 + random shift <= %d px (delivery-2 model)' % k,
                dict(D2_FINAL, seed=seed, augment=['rot90', 'shift%d' % k]))


# EXP-025 (3 seeds): random shifts hurt (1 px: 0.585 +- 0.021, 2 px: 0.552 +- 0.018; train acc falls too). At 16 x 16
# one pixel is 6 % of the image. EXP-026: TTA over the 4 rotations, 0.643 vs 0.646 (no gain).


def d2_subpixel_shift():
    # Was it the size of the shift? 224 -> 32 (area) -> 2 x 2 mean = 224 -> 16 (area), so a 1-px shift on the 32 cache
    # followed by AveragePooling2D(2) is a half-pixel shift at the model's resolution. Control without shift first:
    # it must reproduce EXP-021 (up to the uint8 rounding of the caches).
    base = dict(D2_FINAL, res=32, pool=2)
    for seed in SEEDS:
        run(seeded('EXP-027_res32pool_control', seed), 'EXP-027', 'd2', 'res 32 + AveragePooling 2 (= 16), rot90 (control)',
            dict(base, seed=seed))
        run(seeded('EXP-027_res32pool_shift1', seed), 'EXP-027', 'd2', 'res 32 + AveragePooling 2, rot90 + shift 1 px at 32 (half a pixel at 16)',
            dict(base, seed=seed, augment=['rot90', 'shift1']))


# EXP-027 (3 seeds): half a pixel (32 + pool 2, shift 1 at 32): 0.620 +- 0.012 vs its control 0.631 +- 0.005. Shifts out.


def d2_patience_ablation():
    # EXP-021 changed two things at once vs EXP-019_rot90: EarlyStopping 15 -> 25 and ReduceLROnPlateau 6 -> 8 (and the
    # cap 150 -> 300, never reached). The histories point at the lr patience (with the noisy validation of rot90,
    # patience 6 cut the lr at epoch 14-27). One factor each:
    for seed in SEEDS:
        run(seeded('EXP-028_rlr8_es15', seed), 'EXP-028', 'd2', 'rot90, only ReduceLROnPlateau patience 6 -> 8 (ES 15)',
            dict(D2_FINAL, seed=seed, es_patience=15))
        run(seeded('EXP-028_rlr6_es25', seed), 'EXP-028', 'd2', 'rot90, only EarlyStopping patience 15 -> 25 (RLR 6)',
            dict(D2_FINAL, seed=seed, rlr_patience=6))


# ---------------------------------------------------------------- delivery 3: CNN from scratch (DEC-015..DEC-017)
# Base: the delivery-2 recipe (EXP-021: LeakyReLU 0.3, rot90, Adam beta_1 0.95 + AMSGrad + clipnorm 1, ReduceLROnPlateau 8 /
# EarlyStopping 25 on val_mean_recall, no class weights) with the dense layers replaced by the own CNN
# (architectures.custom_cnn): 3 blocks of 2 x [Conv3x3 -> BN -> LeakyReLU] with 32, 64 and 128 filters, each followed by
# 2x2 max pooling, then global average pooling and Dense(13). float32 and op determinism kept (CHG-028).
D3_BASE = dict(D2_FINAL, arch='cnn', hidden=[], bn=None, conv_blocks=[32, 64, 128], convs_per_block=2, kernel=3,
               downsample='max', head='gap', eval_batch_size=128)


def d3_conv():
    # D3-1: the convolution at the same 16 x 16 input as the ffNN. (a) same batch 256 and lr 3.6e-3 as EXP-021: only the
    # architecture changes. (b) batch 64 with the lr scaled by the square-root rule (1.8e-3, as in EXP-010): the batch that
    # the larger inputs need (memory), so that D3-2 knows what the batch change costs.
    for seed in SEEDS:
        run(seeded('EXP-029_cnn16_bs256', seed), 'EXP-029', 'd3', 'own CNN at 16x16, batch 256 (only the architecture changes)',
            dict(D3_BASE, seed=seed, res=16))
        run(seeded('EXP-029_cnn16_bs64', seed), 'EXP-029', 'd3', 'own CNN at 16x16, batch 64, lr 1.8e-3 (sqrt rule)',
            merge_cfg(D3_BASE, seed=seed, res=16, batch_size=64, optimizer={'lr': 1.8e-3}))


# EXP-029 (3 seeds): the CNN at the ffNN's 16 x 16 input reaches 0.741 +- 0.011 (batch 256) / 0.751 +- 0.007 (batch 64)
# mean recall vs 0.646 of the ffNN (EXP-021): +9.5 pp from the architecture alone, with fewer parameters. Batch 64 ties
# with 256, so it is kept for the larger inputs (memory).
D3_BS64 = merge_cfg(D3_BASE, batch_size=64, optimizer={'lr': 1.8e-3})


def d3_resolution():
    # D3-2: input resolution (DEC-016); 16 is EXP-029_cnn16_bs64. Seeds in the outer loop: partial results usable.
    for seed in SEEDS:
        for res in (32, 64, 112):
            run(seeded('EXP-030_r%d' % res, seed), 'EXP-030', 'd3', 'own CNN at %dx%d (batch 64)' % (res, res),
                dict(D3_BS64, seed=seed, res=res))


# EXP-030 (3 seeds, batch 64): 32 x 32 is best (0.770 +- 0.006); 64 and 112 are worse (0.745, 0.675) with LOWER train
# mean recall (0.784, 0.706) and a tiny gap: underfitting. With 3 blocks the receptive field is ~30 px, the whole image
# at 32 but a fragment at 112, so resolution and depth interact -> two-factor grid (DEC-013), brought forward.
D3_DEPTHS = {3: [32, 64, 128], 4: [32, 64, 128, 256], 5: [32, 64, 128, 256, 512]}


def d3_depth_res():
    # D3-3 (re-ordered, DEC-018): depth {3, 4, 5} blocks x input {32, 64, 112}, 3 seeds; 3 blocks = EXP-030 / EXP-029.
    # Each run in its own process (run_isolated): the 112-px cells are the largest runs so far.
    for seed in SEEDS:
        for res in (32, 64, 112):
            for depth in (4, 5):
                run_isolated(seeded('EXP-031_r%d_b%d' % (res, depth), seed), 'EXP-031', 'd3',
                             'own CNN, %d blocks, %dx%d (batch 64)' % (depth, res, res),
                             dict(D3_BS64, seed=seed, res=res, conv_blocks=D3_DEPTHS[depth]))


# EXP-031 (3 seeds): depth rescues the large inputs (112: 3 blocks 0.675 -> 5 blocks 0.788). 32/4 blocks (0.784 +- 0.010),
# 64/5 and 112/5 tie; 32 x 32 with 4 blocks is kept (6x cheaper than 112/5). 112/5 stays as the reference at 112.
D3_B4 = dict(D3_BS64, res=32, conv_blocks=D3_DEPTHS[4])


def d3_optim():
    # D3-4: learning rate (x0.5, x2) and batch (32, 128, lr by the square-root rule); base = EXP-031_r32_b4 (batch 64,
    # lr 1.8e-3). One factor at a time, 3 seeds.
    variants = [('lr0.9e-3', {'optimizer': {'lr': 0.9e-3}}), ('lr3.6e-3', {'optimizer': {'lr': 3.6e-3}}),
                ('bs32', {'batch_size': 32, 'optimizer': {'lr': 1.27e-3}}),
                ('bs128', {'batch_size': 128, 'optimizer': {'lr': 2.55e-3}})]
    for seed in SEEDS:
        for tag, ch in variants:
            run(seeded('EXP-032_%s' % tag, seed), 'EXP-032', 'd3', 'own CNN 32/4 blocks: %s' % tag,
                merge_cfg(D3_B4, seed=seed, **ch))


# EXP-032 (3 seeds): lr x0.5 / x2 and batch 32 / 128 all tie within noise (0.776-0.788). Batch 128 (lr 2.55e-3, square-root
# rule) is kept: same validation, 1.7x faster epochs.
D3_B4_128 = merge_cfg(D3_B4, batch_size=128, optimizer={'lr': 2.55e-3})


def d3_capacity():
    # D3-5: the rest of the capacity on EXP-032_bs128, one factor at a time, 3 seeds: width (x0.5 / x2), convolutions per
    # block (1 / 3), spatial reduction (average pooling / stride-2 convolution; Springenberg et al. 2015) and head
    # (Flatten instead of global average pooling; Lin et al. 2014).
    variants = [('w16', {'conv_blocks': [16, 32, 64, 128]}), ('w64', {'conv_blocks': [64, 128, 256, 512]}),
                ('convs1', {'convs_per_block': 1}), ('convs3', {'convs_per_block': 3}),
                ('avgpool', {'downsample': 'avg'}), ('stride', {'downsample': 'stride'}),
                ('flatten', {'head': 'flatten'})]
    for seed in SEEDS:
        for tag, ch in variants:
            run(seeded('EXP-033_%s' % tag, seed), 'EXP-033', 'd3', 'own CNN 32/4 blocks/batch 128: %s' % tag,
                dict(D3_B4_128, seed=seed, **ch))


# EXP-033 (3 seeds): the base architecture wins. Halving the width or 1 conv per block costs ~3 pp; doubling the width or
# 3 convs per block tie (-0.7); average pooling -1.5 (fits train less), stride-2 convolution -3.1, Flatten head -1.1 (gap
# 17.6 pp). Kept: 4 blocks x 2 [Conv3x3 -> BN -> LeakyReLU] 32-64-128-256, max pooling, GAP (EXP-032_bs128).


def d3_imbalance():
    # D3-6: class imbalance (update of DEC-009), one treatment at a time on EXP-032_bs128, 3 seeds: class weights (square
    # root, inverse), oversampling (alpha 0.5 and 1 = balanced) and logit adjustment (tau 1; Menon et al. 2021).
    variants = [('cw_sqrt', {'class_weight': 'sqrt'}), ('cw_inverse', {'class_weight': 'inverse'}),
                ('over0.5', {'oversample': 0.5}), ('over1.0', {'oversample': 1.0}),
                ('logitadj1', {'loss': 'logit_adjust', 'la_tau': 1.0})]
    for seed in SEEDS:
        for tag, ch in variants:
            run(seeded('EXP-034_%s' % tag, seed), 'EXP-034', 'd3', 'own CNN, imbalance: %s' % tag,
                dict(D3_B4_128, seed=seed, **ch))


# EXP-034 (3 seeds): logit adjustment tau 1 is the only treatment above the noise (0.803 +- 0.007, +1.8 pp; -1.3 pp of
# accuracy); oversampling 0.5 ties (+1.2, no accuracy loss); class weights +0.5 / +1.1. Kept: logit adjustment.
D3_LA = dict(D3_B4_128, loss='logit_adjust', la_tau=1.0)


def d3_reg_screen():
    # D3-7, step 1: screening of regularisers, ONE seed each, compared with seed 1 of the base (EXP-034_logitadj1: 0.810).
    # The base already has rot90; geometric/photometric/occlusion ops are added to it. Mixing: MixUp alpha 0.2, CutMix
    # alpha 1 (the values of the papers). Classic: Dropout 0.3 before the output, AdamW (decoupled weight decay 1e-4),
    # label smoothing 0.1 (discarded on the ffNN, EXP-022; retried because the model's bias is different).
    variants = [('d4', {'augment': ['rot90', 'flip']}), ('shift2', {'augment': ['rot90', 'shift2']}),
                ('rcrop', {'augment': ['rot90', 'rcrop']}), ('jitter', {'augment': ['rot90', 'jitter']}),
                ('erase', {'augment': ['rot90', 'erase']}), ('mixup', {'mix': 'mixup', 'mix_alpha': 0.2}),
                ('cutmix', {'mix': 'cutmix', 'mix_alpha': 1.0}), ('dropout0.3', {'head_dropout': 0.3}),
                ('adamw', {'optimizer': {'name': 'adamw', 'weight_decay': 1e-4}}), ('ls0.1', {'label_smoothing': 0.1})]
    for tag, ch in variants:
        run('EXP-035_%s' % tag, 'EXP-035', 'd3', 'regulariser screen (1 seed): %s' % tag, merge_cfg(D3_LA, seed=1, **ch))


# EXP-035 (1 seed vs 0.809 of the base's seed 1): nothing stands out; best AdamW +0.7, shift2 +0.6, erase +0.4; CutMix
# -1.6 (covers the small centred objects), label smoothing -0.9, colour jitter -0.5, MixUp and D4 ~0.
D3_REG = {'adamw': {'optimizer': {'name': 'adamw', 'weight_decay': 1e-4}}, 'shift2': {'augment': ['rot90', 'shift2']},
          'erase': {'augment': ['rot90', 'erase']}}


def d3_reg_confirm():
    # D3-7, step 2: the three best of the screen with 3 seeds (seed 1 = EXP-035_<tag>) and their combination (3 seeds),
    # because regularisers interact (DEC-013).
    combo = {'optimizer': {'name': 'adamw', 'weight_decay': 1e-4}, 'augment': ['rot90', 'shift2', 'erase']}
    for seed in SEEDS:
        for tag, ch in D3_REG.items():
            if seed > 1:
                run('EXP-035_%s_seed%d' % (tag, seed), 'EXP-035', 'd3', 'regulariser confirm: %s' % tag,
                    merge_cfg(D3_LA, seed=seed, **ch))
        run(seeded('EXP-036_combo', seed), 'EXP-036', 'd3', 'AdamW + shift2 + erase (on logit adjustment)',
            merge_cfg(D3_LA, seed=seed, **combo))


# EXP-035 (3 seeds): rotations + random shifts of +-2 px: 0.821 +- 0.007 (+1.8 pp), gap halved; AdamW +0.7 and erasing
# +0.2 within noise; the combination adds nothing (EXP-036, 0.820). FINAL OWN CNN (32 px): EXP-035_shift2.
D3_FINAL32 = merge_cfg(D3_LA, augment=['rot90', 'shift2'])
# At 112 px: 5 blocks (EXP-031: depth needed for the receptive field), shifts scaled to the same 6 % of the side (+-7 px),
# batch 64 for memory with the lr by the square-root rule (EXP-032: batch does not matter).
D3_FINAL112 = merge_cfg(D3_FINAL32, res=112, conv_blocks=D3_DEPTHS[5], augment=['rot90', 'shift7'], batch_size=64,
                        optimizer={'lr': 1.8e-3})
KNOWN = ('vgg16', 'resnet50', 'inception_v3')


def d3_cnn112():
    # The final own-CNN recipe at 112 px, 3 seeds: the own CNN's entry in the comparison with the known architectures.
    for seed in SEEDS:
        run_isolated(seeded('EXP-037_cnn112', seed), 'EXP-037', 'd3', 'final own-CNN recipe at 112 px, 5 blocks',
                     dict(D3_FINAL112, seed=seed))


def d3_known_screen():
    # Learning rate of each known architecture from scratch (DEC-015), same recipe as D3_FINAL112 (logit adjustment,
    # rotations + shifts, batch 64). Short budget only to rank the lrs: up to 40 epochs, patience 10 / 4, seed 1.
    for arch in KNOWN:
        for lr in (3e-4, 1e-3, 3e-3):
            run_isolated('EXP-038_%s_lr%g' % (arch, lr), 'EXP-038', 'd3', '%s from scratch, lr %g (screen, <= 40 ep)' % (arch, lr),
                         merge_cfg(D3_FINAL112, arch=arch, epochs=40, es_patience=10, rlr_patience=4, optimizer={'lr': lr}))


# EXP-037 (3 seeds): the own CNN at 112 px / 5 blocks: 0.814 +- 0.021, ties with 32 px (0.821) at 10x the cost per epoch.
# EXP-038 (seed 1, <= 40 epochs): VGG16 only trains at lr 1e-3 (0.761; 3e-4 and 3e-3 collapse to one class from epoch 1);
# ResNet50 and InceptionV3 are best at the lowest lr, 3e-4 (0.824, 0.781), the edge of the grid -> extend to 1e-4.


def d3_known_screen_low():
    for arch in ('resnet50', 'inception_v3'):
        run_isolated('EXP-038_%s_lr0.0001' % arch, 'EXP-038', 'd3', '%s from scratch, lr 0.0001 (screen, <= 40 ep)' % arch,
                     merge_cfg(D3_FINAL112, arch=arch, epochs=40, es_patience=10, rlr_patience=4, optimizer={'lr': 1e-4}))


# EXP-038, extended (seed 1, <= 40 epochs): lr 1e-4 is worse than 3e-4 for ResNet50 (0.799 vs 0.824) and InceptionV3
# (0.768 vs 0.781): 3e-4 is an interior optimum. Chosen: VGG16 1e-3, ResNet50 3e-4, InceptionV3 3e-4.
KNOWN_LR = {'vgg16': 1e-3, 'resnet50': 3e-4, 'inception_v3': 3e-4}


def d3_known():
    # D3-8: each known architecture from scratch with its lr and the FULL recipe of the own CNN at 112 (D3_FINAL112:
    # logit adjustment, rotations + shifts, patience 25 / 8, up to 300 epochs), 3 seeds. Seeds in the outer loop.
    for seed in SEEDS:
        for arch in KNOWN:
            run_isolated(seeded('EXP-039_%s' % arch, seed), 'EXP-039', 'd3', '%s from scratch, lr %g, full recipe' % (arch, KNOWN_LR[arch]),
                         merge_cfg(D3_FINAL112, arch=arch, seed=seed, optimizer={'lr': KNOWN_LR[arch]}))


def magerit_ref():
    # CHG-029: EXP-035_shift2 (seed 1, run on WSL / RTX 4050) repeated unchanged on Magerit, to measure the drift that
    # the new GPU and software (numpy 2.5, Python 3.13) introduce before comparing new runs with the old ones.
    run_isolated('CHG-029_ref_EXP-035_shift2', 'CHG-029', 'd3', 'EXP-035_shift2 (seed 1) repeated on Magerit',
                 merge_cfg(D3_LA, seed=1, **D3_REG['shift2']))


if __name__ == '__main__' and sys.argv[1:2] == ['--one']:   # one run in this process (run_isolated)
    import json
    run_experiment(*json.loads(sys.argv[2]))
    sys.exit(0)

if __name__ == '__main__':
    for name in sys.argv[1:]:
        print('==== %s ====' % name, flush=True)
        globals()[name]()
