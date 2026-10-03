"""run_experiment(): one call = one training run of a ffNN on the cached xview data, logged to experiments/.

Outputs per run (id = '<EXP-###>_<variant>'):
  experiments/experiments.csv          one summary row
  experiments/history/<id>.json        per-epoch history, config, per-class recall, confusion matrix
  experiments/models/<id>.keras        best checkpoint (not versioned)
"""
import copy
import csv
import json
import math
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers

from xview_data import CLASS_NAMES, NUM_CLASSES, ROOT, load_cache, load_train_annotations, split_indices

for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)
tf.config.experimental.enable_op_determinism()  # same seed -> same run, also on GPU (EXP-001)

EXP_DIR = os.path.join(ROOT, 'experiments')
CSV_PATH = os.path.join(EXP_DIR, 'experiments.csv')
HISTORY_DIR = os.path.join(EXP_DIR, 'history')
MODELS_DIR = os.path.join(EXP_DIR, 'models')
CSV_FIELDS = ['id', 'exp_id', 'phase', 'change', 'params', 'actual_epochs', 'best_epoch', 'train_acc', 'train_mean_recall',
              'val_acc', 'val_loss', 'mean_recall', 'min_recall', 'time_s', 's_per_epoch', 'notes', 'config']

DEFAULT_CFG = {
    # data
    'res': 224, 'stratify': True, 'seed': 1,
    # model
    'norm': 'raw',            # 'raw' (pixels in [0, 255]) | 'scale' (Rescaling 1/255) | 'standardize' (per-channel train mean/std)
    'input_relu': False,      # the original Activation('relu') right after Flatten (ISS-001)
    'hidden': [],             # widths of the hidden Dense layers
    'activation': 'relu',     # relu | leaky_relu | elu | gelu | selu | tanh
    'leaky_alpha': 0.1,
    'init': None,             # None -> he_normal (ReLU family), lecun_normal (selu), glorot_uniform (tanh)
    'dropout': 0.0, 'l2': 0.0,
    'bn': None,               # None | 'before' (Dense -> BN -> act) | 'after' (Dense -> act -> BN)
    # optimisation
    'optimizer': {'name': 'adam', 'lr': 1e-3, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8, 'amsgrad': True,
                  'clipnorm': 1.0, 'momentum': 0.0, 'nesterov': False, 'weight_decay': 0.0},
    'batch_size': 16, 'epochs': 20,
    'monitor': 'val_loss',    # val_loss | val_accuracy | val_mean_recall: checkpoint, EarlyStopping and ReduceLROnPlateau
    'es_patience': 10, 'rlr_patience': 4, 'rlr_factor': 0.1,
    'schedule': 'plateau',    # plateau | cosine | exponential | constant
    'exp_decay': 0.9, 'warmup_epochs': 0,
    # imbalance and augmentation
    'class_weight': None,     # None | 'inverse' | 'sqrt'
    'oversample': None,       # None | alpha: sample class c with prob proportional to n_c**(1 - alpha) (1 = balanced)
    'augment': [],            # subset of ['flip', 'rot90', 'color']
}

_DATA = {}


def merge_cfg(base, **changes):
    cfg = copy.deepcopy(base)
    for k, v in changes.items():
        if k == 'optimizer':
            cfg['optimizer'].update(v)
        else:
            cfg[k] = v
    return cfg


def get_data(res, stratify=True):
    """Cached arrays and split. res < 224 is loaded into RAM; 224 (2.8 GB) stays memory-mapped."""
    key = (res, stratify)
    if key not in _DATA:
        x = load_cache('train', res, mmap=(res == 224))
        _, y = load_train_annotations()
        tr, va = split_indices(y, stratify=stratify)
        _DATA[key] = {'x': x, 'y': y, 'train': np.asarray(tr), 'valid': np.asarray(va)}
    return _DATA[key]


def channel_stats(data):
    """Per-channel mean and variance over the training split only (no validation information)."""
    if 'stats' not in data:
        s, s2, n = np.zeros(3), np.zeros(3), 0
        idx = np.sort(data['train'])
        for i in range(0, len(idx), 1024):
            b = np.asarray(data['x'][idx[i:i + 1024]], dtype=np.float64).reshape(-1, 3)
            s += b.sum(0); s2 += (b ** 2).sum(0); n += len(b)
        mean = s / n
        data['stats'] = (mean, s2 / n - mean ** 2)
    return data['stats']


def augment_batch(x, rng, ops):
    """Label-preserving transforms for overhead imagery (no canonical 'up'): D4 group + mild photometric jitter."""
    if 'rot90' in ops or 'flip' in ops:
        for i in range(len(x)):
            if 'rot90' in ops:
                x[i] = np.rot90(x[i], k=rng.integers(4))
            if 'flip' in ops:
                if rng.random() < 0.5:
                    x[i] = x[i, :, ::-1]
                if rng.random() < 0.5:
                    x[i] = x[i, ::-1, :]
    if 'color' in ops:
        c = rng.uniform(0.8, 1.2, size=(len(x), 1, 1, 1)).astype(np.float32)      # contrast
        b = rng.uniform(-20, 20, size=(len(x), 1, 1, 1)).astype(np.float32)       # brightness (uint8 units)
        m = x.mean(axis=(1, 2, 3), keepdims=True)
        x = np.clip((x - m) * c + m + b, 0, 255)
    return x


class CachedSequence(keras.utils.PyDataset):
    """Batches from the uint8 cache, converted to float32 per batch. Returns (x, one_hot, sample_weight)."""

    def __init__(self, x, y, idx, batch_size, shuffle, seed, augment=(), sample_w=None, oversample=None):
        super().__init__()
        self.x, self.y, self.idx, self.bs = x, y, np.asarray(idx), batch_size
        self.shuffle, self.augment, self.sample_w = shuffle, list(augment), sample_w
        self.rng = np.random.default_rng(seed)
        self.p = None
        if oversample is not None:
            counts = np.bincount(y[self.idx], minlength=NUM_CLASSES).astype(float)
            w = counts[y[self.idx]] ** (-float(oversample))   # per-sample prob ∝ n_c^-alpha -> class prob ∝ n_c^(1-alpha)
            self.p = w / w.sum()
        self.order = self.idx
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idx) / self.bs)

    def on_epoch_end(self):
        if self.p is not None:
            self.order = self.rng.choice(self.idx, size=len(self.idx), replace=True, p=self.p)
        elif self.shuffle:
            self.order = self.rng.permutation(self.idx)

    def __getitem__(self, i):
        b = np.sort(self.order[i * self.bs:(i + 1) * self.bs])  # sorted reads are faster on the memmap
        xb = np.asarray(self.x[b], dtype=np.float32)
        if self.augment:
            xb = augment_batch(xb, self.rng, self.augment)
        yb = keras.utils.to_categorical(self.y[b], NUM_CLASSES).astype(np.float32)
        wb = self.sample_w[self.y[b]].astype(np.float32) if self.sample_w is not None else np.ones(len(b), np.float32)
        return xb, yb, wb


def predict_probs(model, x, idx, batch_size=256):
    """Batched prediction (replaces the one-image-at-a-time model.predict of the notebook)."""
    idx = np.asarray(idx)
    out = []
    for i in range(0, len(idx), batch_size):
        out.append(model.predict_on_batch(np.asarray(x[idx[i:i + batch_size]], dtype=np.float32)))
    return np.concatenate(out)


def inactive_units(model, x, idx):
    """Fraction of units of the first hidden Dense layer whose pre-activation is <= 0 for every validation image
    (for ReLU: dead units, which get no gradient). None if the model has no hidden layer."""
    dense = [l for l in model.layers if isinstance(l, layers.Dense)]
    if len(dense) < 2:
        return None
    sub = keras.Model(model.inputs, dense[0].output)
    pre = predict_probs(sub, x, np.sort(idx))
    return float((pre.max(0) <= 0).mean())


def classification_metrics(y_true, probs):
    y_pred = probs.argmax(1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-7, 1.0))))
    return {'acc': float(np.mean(y_pred == y_true)), 'mean_recall': float(recall.mean()), 'min_recall': float(recall.min()),
            'recall': recall.tolist(), 'cm': cm.tolist(), 'loss': loss}


def class_weights(y_train, mode):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    w = len(y_train) / (NUM_CLASSES * counts)          # inverse frequency: n_total / (n_classes * n_c)
    if mode == 'sqrt':
        w = np.sqrt(w)
    elif mode != 'inverse':
        raise ValueError(mode)
    return w / np.sum(w * counts / len(y_train))      # mean weight over the training samples = 1 (same loss scale)


def build_model(cfg, data):
    r = cfg['res']
    act = cfg['activation']
    init = cfg['init'] or {'selu': 'lecun_normal', 'tanh': 'glorot_uniform'}.get(act, 'he_normal')
    reg = regularizers.l2(cfg['l2']) if cfg['l2'] else None
    inp = keras.Input(shape=(r, r, 3))
    x = inp
    if cfg['norm'] == 'scale':
        x = layers.Rescaling(1.0 / 255)(x)
    elif cfg['norm'] == 'standardize':
        mean, var = channel_stats(data)
        x = layers.Normalization(mean=mean.tolist(), variance=var.tolist())(x)
    x = layers.Flatten()(x)
    if cfg['input_relu']:
        x = layers.Activation('relu')(x)
    for h in cfg['hidden']:
        x = layers.Dense(h, kernel_initializer=init, kernel_regularizer=reg, use_bias=cfg['bn'] != 'before')(x)
        if cfg['bn'] == 'before':
            x = layers.BatchNormalization()(x)
        if act == 'leaky_relu':
            x = layers.LeakyReLU(negative_slope=cfg['leaky_alpha'])(x)
        else:
            x = layers.Activation(act)(x)
        if cfg['bn'] == 'after':
            x = layers.BatchNormalization()(x)
        if cfg['dropout']:
            x = (layers.AlphaDropout if act == 'selu' else layers.Dropout)(cfg['dropout'])(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=reg)(x)
    return keras.Model(inp, out)


def build_optimizer(o):
    clip = o.get('clipnorm') or None
    name = o['name']
    if name == 'adam':
        return keras.optimizers.Adam(o['lr'], beta_1=o['beta_1'], beta_2=o['beta_2'], epsilon=o['epsilon'],
                                     amsgrad=o['amsgrad'], clipnorm=clip)
    if name == 'adamw':
        return keras.optimizers.AdamW(o['lr'], weight_decay=o['weight_decay'], beta_1=o['beta_1'], beta_2=o['beta_2'],
                                      epsilon=o['epsilon'], amsgrad=o['amsgrad'], clipnorm=clip)
    if name == 'sgd':
        return keras.optimizers.SGD(o['lr'], momentum=o['momentum'], nesterov=o['nesterov'], clipnorm=clip)
    if name == 'rmsprop':
        return keras.optimizers.RMSprop(o['lr'], momentum=o['momentum'], clipnorm=clip)
    raise ValueError(name)


class EpochMetrics(keras.callbacks.Callback):
    """Validation in one batched pass: adds val_loss, val_accuracy, val_mean_recall, learning_rate and epoch_time
    to the logs. Must be the first callback, so that ModelCheckpoint / EarlyStopping / ReduceLROnPlateau see them."""

    def __init__(self, x, y, idx):
        super().__init__()
        self.x, self.y, self.idx = x, y, idx

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        m = classification_metrics(self.y[self.idx], predict_probs(self.model, self.x, self.idx))
        logs['val_loss'], logs['val_accuracy'], logs['val_mean_recall'] = m['loss'], m['acc'], m['mean_recall']
        logs['learning_rate'] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        logs['epoch_time'] = time.time() - self.t0


class EpochLR(keras.callbacks.Callback):
    """Per-epoch learning rate: linear warmup, then cosine / exponential decay (or untouched for plateau/constant)."""

    def __init__(self, cfg):
        super().__init__()
        self.base, self.cfg = cfg['optimizer']['lr'], cfg

    def on_epoch_begin(self, epoch, logs=None):
        c, w = self.cfg, self.cfg['warmup_epochs']
        if epoch < w:
            lr = self.base * (epoch + 1) / (w + 1)
        elif c['schedule'] == 'cosine':
            t = (epoch - w) / max(1, c['epochs'] - w)
            lr = self.base * 0.5 * (1 + math.cos(math.pi * t))
        elif c['schedule'] == 'exponential':
            lr = self.base * c['exp_decay'] ** (epoch - w)
        elif epoch == w and w > 0:
            lr = self.base
        else:
            return
        self.model.optimizer.learning_rate.assign(lr)


def run_experiment(run_id, exp_id, phase, change, cfg, notes='', verbose=0):
    """Train one configuration, select the best epoch on validation (cfg['monitor']), log everything."""
    cfg = merge_cfg(DEFAULT_CFG, **cfg)
    for d in (HISTORY_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)
    keras.backend.clear_session()
    keras.utils.set_random_seed(cfg['seed'])  # python, numpy and TF seeds
    data = get_data(cfg['res'], cfg['stratify'])
    x, y, tr, va = data['x'], data['y'], data['train'], data['valid']

    model = build_model(cfg, data)
    model.compile(optimizer=build_optimizer(cfg['optimizer']), loss='categorical_crossentropy', metrics=['accuracy'])
    sw = class_weights(y[tr], cfg['class_weight']) if cfg['class_weight'] else None
    train_seq = CachedSequence(x, y, tr, cfg['batch_size'], True, cfg['seed'], cfg['augment'], sw, cfg['oversample'])

    mon = cfg['monitor']
    mode = 'min' if mon == 'val_loss' else 'max'
    model_path = os.path.join(MODELS_DIR, run_id + '.keras')
    callbacks = [EpochMetrics(x, y, va), EpochLR(cfg),
                 keras.callbacks.ModelCheckpoint(model_path, monitor=mon, mode=mode, save_best_only=True),
                 keras.callbacks.EarlyStopping(mon, mode=mode, patience=cfg['es_patience']),
                 keras.callbacks.TerminateOnNaN()]
    if cfg['schedule'] == 'plateau':
        callbacks.insert(3, keras.callbacks.ReduceLROnPlateau(mon, mode=mode, factor=cfg['rlr_factor'],
                                                              patience=cfg['rlr_patience']))
    t0 = time.time()
    h = model.fit(train_seq, epochs=cfg['epochs'], callbacks=callbacks, verbose=verbose)
    elapsed = time.time() - t0
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    vals = np.array(hist[mon])
    best = int(np.nanargmin(vals) if mode == 'min' else np.nanargmax(vals)) if np.isfinite(vals).any() else len(vals) - 1

    if os.path.exists(model_path):
        model.load_weights(model_path)
    val = classification_metrics(y[va], predict_probs(model, x, va))
    trn = classification_metrics(y[tr], predict_probs(model, x, tr))
    val['inactive_units'] = inactive_units(model, x, va)

    row = {'id': run_id, 'exp_id': exp_id, 'phase': phase, 'change': change, 'params': model.count_params(),
           'actual_epochs': len(vals), 'best_epoch': best + 1, 'train_acc': round(trn['acc'], 4),
           'train_mean_recall': round(trn['mean_recall'], 4), 'val_acc': round(val['acc'], 4),
           'val_loss': round(val['loss'], 4), 'mean_recall': round(val['mean_recall'], 4),
           'min_recall': round(val['min_recall'], 4), 'time_s': round(elapsed, 1),
           's_per_epoch': round(float(np.mean(hist['epoch_time'])), 2), 'notes': notes,
           'config': json.dumps(cfg, sort_keys=True)}
    with open(os.path.join(HISTORY_DIR, run_id + '.json'), 'w') as ofs:
        json.dump({'id': run_id, 'exp_id': exp_id, 'phase': phase, 'change': change, 'config': cfg, 'history': hist,
                   'best_epoch': best + 1, 'val': val, 'train': {k: trn[k] for k in ('acc', 'mean_recall', 'recall', 'loss')},
                   'class_names': CLASS_NAMES}, ofs, indent=1)
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline='') as ifs:
            rows = [r for r in csv.DictReader(ifs) if r['id'] != run_id]   # re-running an id replaces its row
    rows.append(row)
    with open(CSV_PATH, 'w', newline='') as ofs:
        w = csv.DictWriter(ofs, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print('%-28s %-40s params=%-9d ep=%2d/%2d train_acc=%.3f val_acc=%.3f mean_recall=%.3f min_rec=%.3f %.0fs'
          % (run_id, change[:40], row['params'], best + 1, len(vals), trn['acc'], val['acc'], val['mean_recall'],
             val['min_recall'], elapsed), flush=True)
    return row
