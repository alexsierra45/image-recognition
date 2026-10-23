"""run_experiment(): one call = one training run (ffNN or CNN: cfg['arch']) on the cached xview data, logged to experiments/.

Outputs per run (id = '<EXP-###>_<variant>'):
  experiments/experiments.csv          one summary row
  experiments/history/<id>.json        per-epoch history, config, per-class recall, confusion matrix
  experiments/models/<id>.keras        best checkpoint (not versioned)
"""
import contextlib
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

import architectures

from xview_data import CLASS_NAMES, NUM_CLASSES, ROOT, load_cache, load_train_annotations, split_indices

for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:  # TF already initialised (e.g. imported from a notebook after using the GPU): keep its setting
        pass
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
    'arch': 'mlp',            # 'mlp' (deliveries 1-2) | 'cnn' (own CNN) | 'vgg16' | 'vgg16_bn' | 'resnet50' | 'inception_v3' | 'mobilenet_v2'
    'norm': 'raw',            # 'raw' (pixels in [0, 255]) | 'scale' (Rescaling 1/255) | 'standardize' (per-channel train mean/std)
    'input_relu': False,      # the original Activation('relu') right after Flatten (ISS-001)
    'pool': 1,                # >1: AveragePooling2D(pool) right after the input scaling (res 32 + pool 2 = area to 16: EXP-027)
    'hidden': [],             # widths of the hidden Dense layers
    'activation': 'relu',     # relu | leaky_relu | elu | gelu | selu | tanh
    'leaky_alpha': 0.1,
    'init': None,             # None -> he_normal (ReLU family), lecun_normal (selu), glorot_uniform (tanh)
    'dropout': 0.0, 'l2': 0.0,
    'label_smoothing': 0.0,   # targets (1 - eps) one-hot + eps / K (EXP-022); 0 -> the plain cross-entropy string loss
    'bn': None,               # None | 'before' (Dense -> BN -> act) | 'after' (Dense -> act -> BN)
    # own CNN (arch 'cnn', architectures.custom_cnn) and head of every CNN
    'conv_blocks': [32, 64, 128], 'convs_per_block': 2, 'kernel': 3,
    'downsample': 'max',      # 'max' | 'avg' | 'stride' (last conv of each block with stride 2)
    'head': 'gap',            # 'gap' (global average pooling) | 'flatten'
    'head_dense': 0,          # units of a hidden Dense layer in the head (0 = none)
    'head_dropout': 0.0,      # Dropout right before the output layer (CNN only)
    'loss': 'ce',             # 'ce' | 'logit_adjust' (logits + la_tau * log prior during training: Menon et al. 2021)
    'la_tau': 1.0,
    'precision': 'float32',   # 'float32' | 'mixed_float16' (softmax kept in float32)
    'eval_batch_size': 256,   # batch of the per-epoch validation pass and of the final predictions
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
    'augment': [],            # subset of ['flip', 'rot90', 'color', 'shiftK', 'rcrop', 'jitter', 'erase'] (see augment_batch)
    'mix': None,              # None | 'mixup' | 'cutmix': mixes images and labels within the batch
    'mix_alpha': 0.2,         # Beta(alpha, alpha) of the mixing proportion
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
    shift = max([int(o[5:]) for o in ops if o.startswith('shift')] or [0])
    if shift:   # random translation by up to `shift` pixels in each axis, reflecting at the border (EXP-025)
        h, w = x.shape[1:3]
        pad = np.pad(x, ((0, 0), (shift, shift), (shift, shift), (0, 0)), mode='reflect')
        for i in range(len(x)):
            dy, dx = rng.integers(-shift, shift + 1, size=2)
            x[i] = pad[i, shift + dy:shift + dy + h, shift + dx:shift + dx + w]
    if 'color' in ops:
        c = rng.uniform(0.8, 1.2, size=(len(x), 1, 1, 1)).astype(np.float32)      # contrast
        b = rng.uniform(-20, 20, size=(len(x), 1, 1, 1)).astype(np.float32)       # brightness (uint8 units)
        m = x.mean(axis=(1, 2, 3), keepdims=True)
        x = np.clip((x - m) * c + m + b, 0, 255)
    # delivery 3 (D3-6); each op only draws random numbers when present, so the older ops are unchanged
    if 'rcrop' in ops:   # random crop of 80-100 % of the side, resized back (scale jitter + translation)
        n, h = len(x), x.shape[1]
        side = rng.uniform(0.8, 1.0, size=n)
        y0 = rng.uniform(0, 1, size=n) * (1 - side)
        x0 = rng.uniform(0, 1, size=n) * (1 - side)
        boxes = np.stack([y0, x0, y0 + side, x0 + side], axis=1).astype(np.float32)
        x = tf.image.crop_and_resize(x, boxes, np.arange(n, dtype=np.int32), (h, h), method='bilinear').numpy()
    if 'jitter' in ops:  # mild photometric jitter: brightness, contrast, saturation and per-channel gain
        n = len(x)
        b = rng.uniform(-10, 10, size=(n, 1, 1, 1)).astype(np.float32)
        c = rng.uniform(0.9, 1.1, size=(n, 1, 1, 1)).astype(np.float32)
        s = rng.uniform(0.9, 1.1, size=(n, 1, 1, 1)).astype(np.float32)
        g = rng.uniform(0.95, 1.05, size=(n, 1, 1, 3)).astype(np.float32)
        gray = x.mean(axis=3, keepdims=True)
        x = gray + (x - gray) * s
        m = x.mean(axis=(1, 2, 3), keepdims=True)
        x = np.clip(((x - m) * c + m + b) * g, 0, 255)
    if 'erase' in ops:   # Random Erasing (Zhong et al. 2020): p 0.5, 2-10 % of the area, aspect 0.3-3.3, random values
        h, w = x.shape[1:3]
        for i in range(len(x)):
            if rng.random() < 0.5:
                area = rng.uniform(0.02, 0.10) * h * w
                ratio = np.exp(rng.uniform(np.log(0.3), np.log(1 / 0.3)))
                eh = min(h, max(1, int(round(np.sqrt(area * ratio)))))
                ew = min(w, max(1, int(round(np.sqrt(area / ratio)))))
                top, left = rng.integers(0, h - eh + 1), rng.integers(0, w - ew + 1)
                x[i, top:top + eh, left:left + ew] = rng.uniform(0, 255, size=(eh, ew, x.shape[3]))
    return x


def mix_batch(x, y, w, rng, kind, alpha):
    """MixUp (Zhang et al. 2018) or CutMix (Yun et al. 2019) within the batch: images, one-hot labels and sample
    weights are mixed with the same proportion lam ~ Beta(alpha, alpha) (one lam per batch)."""
    lam = float(rng.beta(alpha, alpha))
    perm = rng.permutation(len(x))
    if kind == 'mixup':
        x = lam * x + (1 - lam) * x[perm]
    else:   # cutmix: paste a box of area (1 - lam) from the permuted batch; lam = exact kept area
        h, wd = x.shape[1:3]
        ch, cw = int(round(h * np.sqrt(1 - lam))), int(round(wd * np.sqrt(1 - lam)))
        cy, cx = rng.integers(h), rng.integers(wd)
        y0, y1 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch - ch // 2, 0, h)
        x0, x1 = np.clip(cx - cw // 2, 0, wd), np.clip(cx + cw - cw // 2, 0, wd)
        x = x.copy()
        x[:, y0:y1, x0:x1] = x[perm, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * wd)
    return x, lam * y + (1 - lam) * y[perm], lam * w + (1 - lam) * w[perm]


class CachedSequence(keras.utils.PyDataset):
    """Batches from the uint8 cache, converted to float32 per batch. Returns (x, one_hot, sample_weight)."""

    def __init__(self, x, y, idx, batch_size, shuffle, seed, augment=(), sample_w=None, oversample=None, mix=None,
                 mix_alpha=0.2):
        super().__init__()
        self.mix, self.mix_alpha = mix, mix_alpha
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
        if self.mix:
            xb, yb, wb = mix_batch(xb, yb, wb, self.rng, self.mix, self.mix_alpha)
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


@keras.saving.register_keras_serializable(package='xview')
class LogitAdjustedCE(keras.losses.Loss):
    """Logit-adjusted cross-entropy (Menon et al. 2021) on a softmax output: CE(softmax(log p + tau * log prior)),
    with log p the log-probabilities of the model. Only training changes; at prediction the model outputs softmax(z).
    It targets the balanced error, i.e. the mean recall."""

    def __init__(self, log_prior, tau=1.0, label_smoothing=0.0, name='logit_adjusted_ce', **kw):
        super().__init__(name=name, **kw)
        self.log_prior = [float(v) for v in log_prior]
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        logp = keras.ops.log(keras.ops.clip(keras.ops.cast(y_pred, 'float32'), 1e-7, 1.0))
        z = logp + self.tau * keras.ops.convert_to_tensor(self.log_prior, dtype='float32')
        if self.label_smoothing:   # targets (1 - eps) one-hot + eps / K, as in CategoricalCrossentropy
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / len(self.log_prior)
        return -keras.ops.sum(y_true * keras.ops.log_softmax(z, axis=-1), axis=-1)

    def get_config(self):
        return {**super().get_config(), 'log_prior': self.log_prior, 'tau': self.tau, 'label_smoothing': self.label_smoothing}


def build_model(cfg, data):
    """Dispatch on cfg['arch']: the dense network of deliveries 1-2 (unchanged) or a CNN (delivery 3)."""
    if cfg['arch'] == 'mlp':
        return _build_mlp(cfg, data)
    r = cfg['res']
    inp = keras.Input(shape=(r, r, 3))
    x = inp
    if cfg['norm'] == 'scale':
        x = layers.Rescaling(1.0 / 255)(x)
    elif cfg['norm'] == 'standardize':
        mean, var = channel_stats(data)
        x = layers.Normalization(mean=mean.tolist(), variance=var.tolist())(x)
    if cfg['pool'] > 1:
        x = layers.AveragePooling2D(cfg['pool'])(x)
    if cfg['arch'] == 'cnn':
        x = architectures.custom_cnn(x, cfg)
    elif cfg['arch'] == 'vgg16_bn':
        x = architectures.vgg16_bn(x, cfg)
    else:
        x = architectures.known(cfg['arch'], x, r // cfg['pool'] if cfg['pool'] > 1 else r)
    if cfg['head_dropout']:
        x = layers.Dropout(cfg['head_dropout'])(x)
    reg = regularizers.l2(cfg['l2']) if cfg['l2'] else None
    out = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=reg, dtype='float32')(x)
    return keras.Model(inp, out)


def _build_mlp(cfg, data):
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
    if cfg['pool'] > 1:
        x = layers.AveragePooling2D(cfg['pool'])(x)
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

    def __init__(self, x, y, idx, batch_size=256):
        super().__init__()
        self.x, self.y, self.idx, self.bs = x, y, idx, batch_size

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        m = classification_metrics(self.y[self.idx], predict_probs(self.model, self.x, self.idx, self.bs))
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


@contextlib.contextmanager
def csv_lock(timeout=600):
    """Exclusive lock on experiments.csv across processes and nodes: mkdir is atomic on POSIX file systems (BeeGFS too),
    unlike flock, which BeeGFS only honours cluster-wide if the server enables it. A lock older than `timeout` is stale."""
    lock = CSV_PATH + '.lock'
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:   # the holder may release it between mkdir and stat
                if time.time() - os.path.getmtime(lock) > timeout:
                    os.rmdir(lock)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.2)
    try:
        yield
    finally:
        os.rmdir(lock)


def run_experiment(run_id, exp_id, phase, change, cfg, notes='', verbose=0):
    """Train one configuration, select the best epoch on validation (cfg['monitor']), log everything."""
    cfg = merge_cfg(DEFAULT_CFG, **cfg)
    for d in (HISTORY_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy(cfg['precision'])
    keras.utils.set_random_seed(cfg['seed'])  # python, numpy and TF seeds
    data = get_data(cfg['res'], cfg['stratify'])
    x, y, tr, va = data['x'], data['y'], data['train'], data['valid']

    model = build_model(cfg, data)
    if cfg['loss'] == 'logit_adjust':
        prior = np.bincount(y[tr], minlength=NUM_CLASSES) / len(tr)
        loss = LogitAdjustedCE(np.log(prior), cfg['la_tau'], cfg['label_smoothing'])
    elif cfg['label_smoothing']:
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=cfg['label_smoothing'])
    else:
        loss = 'categorical_crossentropy'
    model.compile(optimizer=build_optimizer(cfg['optimizer']), loss=loss, metrics=['accuracy'])
    sw = class_weights(y[tr], cfg['class_weight']) if cfg['class_weight'] else None
    train_seq = CachedSequence(x, y, tr, cfg['batch_size'], True, cfg['seed'], cfg['augment'], sw, cfg['oversample'],
                               cfg['mix'], cfg['mix_alpha'])

    mon = cfg['monitor']
    mode = 'min' if mon == 'val_loss' else 'max'
    model_path = os.path.join(MODELS_DIR, run_id + '.keras')
    callbacks = [EpochMetrics(x, y, va, cfg['eval_batch_size']), EpochLR(cfg),
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
    val = classification_metrics(y[va], predict_probs(model, x, va, cfg['eval_batch_size']))
    trn = classification_metrics(y[tr], predict_probs(model, x, tr, cfg['eval_batch_size']))
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
    with csv_lock():   # ISS-016: several SLURM jobs may finish at once; the read-modify-write must not interleave
        rows = []
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, newline='') as ifs:
                rows = [r for r in csv.DictReader(ifs) if r['id'] != run_id]   # re-running an id replaces its row
        rows.append(row)
        tmp = CSV_PATH + '.tmp'
        with open(tmp, 'w', newline='') as ofs:
            w = csv.DictWriter(ofs, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, CSV_PATH)
    print('%-28s %-40s params=%-9d ep=%2d/%2d train_acc=%.3f val_acc=%.3f mean_recall=%.3f min_rec=%.3f %.0fs'
          % (run_id, change[:40], row['params'], best + 1, len(vals), trn['acc'], val['acc'], val['mean_recall'],
             val['min_recall'], elapsed), flush=True)
    return row
