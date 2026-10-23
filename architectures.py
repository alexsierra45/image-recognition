"""Convolutional architectures for delivery 3 (DEC-015..DEC-017), used by experiment_utils.build_model.

- custom_cnn: our own CNN, built from cfg keys (conv_blocks, convs_per_block, kernel, downsample, head, head_dense).
- known: popular architectures from keras.applications, trained from scratch (weights=None), with the same head
  for all of them (global average pooling -> the Dense(13) added by build_model). They are the architectures that
  delivery 4 will use with ImageNet weights, so "from scratch vs pre-trained" is a direct comparison (DEC-015).
"""
import keras
from keras import layers, regularizers

KNOWN = {
    'vgg16': keras.applications.VGG16,              # Simonyan & Zisserman (2015)
    'resnet50': keras.applications.ResNet50,        # He et al. (2016)
    'inception_v3': keras.applications.InceptionV3,  # Szegedy et al. (2016), GoogLeNet family; needs >= 75 px
    'mobilenet_v2': keras.applications.MobileNetV2,  # Sandler et al. (2018), lightweight reference
}


def _activation(x, cfg):
    if cfg['activation'] == 'leaky_relu':
        return layers.LeakyReLU(negative_slope=cfg['leaky_alpha'])(x)
    return layers.Activation(cfg['activation'])(x)


def custom_cnn(x, cfg):
    """Blocks of [Conv(k x k, 'same') -> BatchNorm -> activation] x convs_per_block, each block followed by a 2x
    spatial reduction (max pooling, average pooling or a stride-2 last convolution), then the head: global average
    pooling (no parameters) or Flatten, optionally followed by one hidden Dense layer."""
    reg = regularizers.l2(cfg['l2']) if cfg['l2'] else None
    size = cfg['res'] // cfg['pool'] if cfg['pool'] > 1 else cfg['res']
    for w in cfg['conv_blocks']:
        for j in range(cfg['convs_per_block']):
            last = j == cfg['convs_per_block'] - 1
            stride = 2 if (cfg['downsample'] == 'stride' and last and size >= 2) else 1
            x = layers.Conv2D(w, cfg['kernel'], strides=stride, padding='same', use_bias=False,
                              kernel_initializer='he_normal', kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = _activation(x, cfg)
        if size >= 2:   # no reduction once the feature map is 1 x 1 (e.g. 5 blocks on a 16 x 16 input)
            if cfg['downsample'] == 'max':
                x = layers.MaxPooling2D(2)(x)
            elif cfg['downsample'] == 'avg':
                x = layers.AveragePooling2D(2)(x)
            size //= 2
    if cfg['head'] == 'gap':
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.Flatten()(x)
    if cfg['head_dense']:
        x = layers.Dense(cfg['head_dense'], use_bias=False, kernel_initializer='he_normal', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = _activation(x, cfg)
    return x


VGG16_BLOCKS = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]   # (filters, convs) of configuration D, Simonyan & Zisserman


def vgg16_bn(x, cfg):
    """VGG16 (the 13 convolutions of keras.applications.VGG16: same widths, 3x3 'same', ReLU, default glorot init, 2x2 max
    pooling after each block) with BatchNorm after every convolution, the "vgg16_bn" variant (ISS-014: plain VGG16 only
    trains from scratch in a narrow lr window). Conv bias dropped, as BN's beta replaces it. Global average pooling head."""
    for filters, convs in VGG16_BLOCKS:
        for _ in range(convs):
            x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
    return layers.GlobalAveragePooling2D()(x)


def known(name, x, res):
    """keras.applications backbone without its ImageNet classifier, from scratch, with global average pooling."""
    base = KNOWN[name](include_top=False, weights=None, input_shape=(res, res, 3), pooling='avg')
    return base(x)
