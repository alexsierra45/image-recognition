"""Data loading for xview_recognition: annotations, stratified split and a uint8 .npy cache per resolution.

Reading the 224x224 TIFs with rasterio dominated the training time (ISS-005). Each image is read once,
stored as uint8 in cache/<split>_<res>.npy and memory-mapped afterwards (IDEA-009 / CHG-007).
"""
import json
import os
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'xview_recognition')
CACHE_DIR = os.path.join(ROOT, 'cache')

CATEGORIES = {0: 'Cargo plane', 1: 'Small car', 2: 'Bus', 3: 'Truck', 4: 'Motorboat', 5: 'Fishing vessel', 6: 'Dump truck',
              7: 'Excavator', 8: 'Building', 9: 'Helipad', 10: 'Storage tank', 11: 'Shipping container', 12: 'Pylon'}
CLASS_NAMES = list(CATEGORIES.values())
NUM_CLASSES = len(CLASS_NAMES)


def load_geoimage(filename):
    """Same reader as the notebook (cell 4): HxWxC array with the raster dtype (uint8)."""
    import rasterio
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
    with rasterio.open(os.path.join(DATA_DIR, filename), 'r') as src:
        img = np.zeros((src.height, src.width, src.count), dtype=src.profile['dtype'])
        for band in range(src.count):
            img[:, :, band] = src.read(band + 1)
    return img


def load_train_annotations():
    """(filenames, labels) in the order of xview_ann_train.json, as the notebook builds `anns` (cell 7)."""
    with open(os.path.join(DATA_DIR, 'xview_ann_train.json')) as ifs:
        json_data = json.load(ifs)
    filenames, labels = [], []
    for json_img, json_ann in zip(json_data['images'].values(), json_data['annotations'].values()):
        filenames.append(json_img['filename'])
        labels.append(CLASS_NAMES.index(json_ann['category_id']))
    return filenames, np.array(labels, dtype=np.int64)


def list_test_files():
    """Sorted test TIF names. Only *.tif: copies made from Windows carry `:Zone.Identifier` files (ISS-006)."""
    return sorted(f for f in os.listdir(os.path.join(DATA_DIR, 'xview_test')) if f.lower().endswith('.tif'))


def split_indices(labels, test_size=0.1, random_state=1, stratify=True):
    """Train/valid indices. Stratified by class since EXP-001 (IDEA-003)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True,
                            stratify=labels if stratify else None)


def _resize(batch, res):
    """Area resize (average over the covered pixels: antialiased downsampling), back to uint8."""
    if res == batch.shape[1]:
        return batch
    import tensorflow as tf
    with tf.device('/CPU:0'):
        out = tf.image.resize(tf.convert_to_tensor(batch, tf.float32), (res, res), method='area')
    return np.clip(np.round(out.numpy()), 0, 255).astype(np.uint8)


def _build_cache(path, filenames, res, chunk=512):
    os.makedirs(CACHE_DIR, exist_ok=True)
    src_224 = os.path.join(CACHE_DIR, os.path.basename(path).replace('_%d.npy' % res, '_224.npy'))
    tmp = path + '.tmp.npy'
    arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(filenames), res, res, 3))
    base = np.load(src_224, mmap_mode='r') if res != 224 and os.path.exists(src_224) else None
    for start in range(0, len(filenames), chunk):
        stop = min(start + chunk, len(filenames))
        if base is not None:
            batch = np.asarray(base[start:stop])
        else:
            batch = np.stack([load_geoimage(f) for f in filenames[start:stop]])
        arr[start:stop] = _resize(batch, res)
    arr.flush()
    del arr
    os.replace(tmp, path)


def load_cache(split, res=224, mmap=True):
    """uint8 array (N, res, res, 3) for split in {'train', 'test'}; built on first use.

    'train' follows the order of load_train_annotations(); 'test' follows list_test_files().
    """
    path = os.path.join(CACHE_DIR, '%s_%d.npy' % (split, res))
    if not os.path.exists(path):
        if res != 224:
            load_cache(split, 224)  # resize from the 224 cache instead of re-reading the TIFs
        if split == 'train':
            filenames = load_train_annotations()[0]
        else:
            filenames = ['xview_test/' + f for f in list_test_files()]
        _build_cache(path, filenames, res)
    return np.load(path, mmap_mode='r' if mmap else None)
