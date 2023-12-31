from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range

from ..utils import _raise, backend_channels_last

from ..utils.tf import keras_import, K
Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, Cropping2D, Cropping3D, Concatenate, Add, Dropout, Activation, BatchNormalization = \
    keras_import('layers', 'Conv2D', 'MaxPooling2D', 'UpSampling2D', 'Conv3D', 'MaxPooling3D', 'UpSampling3D', 'Cropping2D', 'Cropping3D', 'Concatenate', 'Add', 'Dropout', 'Activation', 'BatchNormalization')
istrain = True  # None  # False  #
droptrain = False  # True  # 显示设置dropout的training参数
from keras import backend as K
# from keras.layers import UpSampling2D


def conv_block2(n_filter, n1, n2,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            if droptrain:
                s = Dropout(dropout)(s, training=istrain)
            else:
                s = Dropout(dropout)(s)
        return s

    return _func



def conv_block3(n_filter, n1, n2, n3,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            if droptrain:
                s = Dropout(dropout)(s, training=istrain)
            else:
                s = Dropout(dropout)(s)
        return s

    return _func



def unet_block(n_depth=2, n_filter_base=16, kernel_size=(3,3), n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None,
               pool=(2,2),
               kernel_init="glorot_uniform",
               prefix=''):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    conv_block = conv_block2  if n_dim == 2 else conv_block3
    pooling    = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   init=kernel_init,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(n_filter_base * 2 ** n_depth, *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)

        layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
                           dropout=dropout,
                           activation=activation,
                           init=kernel_init,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

        return layer

    return _func


def unet_blockdropout(n_depth=2, n_filter_base=16, kernel_size=(3,3), n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.7, dropout0=0.0,
               last_activation=None,
               pool=(2,2),
               kernel_init="glorot_uniform",
               prefix='', dropouttype='_dropout'):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    conv_block = conv_block2  if n_dim == 2 else conv_block3
    pooling    = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    def _func(input):
        skip_layers = []
        layer = input
    
        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout0,
                                   activation=activation,
                                   init=kernel_init,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)
    
        # middle
        import numpy as np
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(np.int32(np.ceil((n_filter_base * 2 ** n_depth) // dropout)), *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)
    
        layer = conv_block(np.int32(np.ceil((n_filter_base * 2 ** max(0, n_depth - 1)) // dropout)), *kernel_size,
                           dropout=dropout,
                           activation=activation,
                           init=kernel_init,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)
    
        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout0,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)
        
            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout0,
                               init=kernel_init,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)
    
        return layer


    def _funcv2(input):
        dropout = 0.99
        skip_layers = []
        layer = input

        # down ...
        for i in range(n_conv_per_depth):  # 2
            layer = conv_block(n_filter_base * 2 ** 0, *kernel_size,
                               dropout=dropout0,
                               activation=activation,
                               init=kernel_init,
                               batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (0, i)))(layer)
        skip_layers.append(layer)
        layer = pooling(pool, name=_name("max_%s" % 0))(layer)
        
        for n in range(1, n_depth):  # 2-1=1
            for i in range(n_conv_per_depth):  # 2
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   init=kernel_init,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        import numpy as np
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(np.int32(np.ceil((n_filter_base * 2 ** n_depth)//dropout)), *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)

        layer = conv_block(np.int32(np.ceil((n_filter_base * 2 ** max(0, n_depth - 1))//dropout)), *kernel_size,
                           dropout=dropout,
                           activation=activation,
                           init=kernel_init,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)

        # ...and up with skip layers
        for n in reversed(range(1, n_depth)):  # 1
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):  # 1
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)


        layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[0]])
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(n_filter_base * 2 ** 0, *kernel_size,
                               dropout=dropout0,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (0, i)))(layer)

        layer = conv_block(n_filter_base * 2 ** max(0, 0 - 1), *kernel_size,
                           dropout=dropout0,
                           init=kernel_init,
                           activation=activation if 0 > 0 else last_activation,
                           batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (0, n_conv_per_depth)))(layer)

        return layer


    if (dropouttype == '_dropout') or (dropouttype == '_dropout_GAN'):
        return _func
    elif (dropouttype == '_dropout_v2') or (dropouttype == '_dropout_v2_GAN'):
        return _funcv2


def resnet_block(n_filter, kernel_size=(3, 3), pool=(1, 1), n_conv_per_block=2,
                 batch_norm=False, kernel_initializer='he_normal', activation='relu'):

    n_conv_per_block >= 2 or _raise(ValueError('required: n_conv_per_block >= 2'))
    len(pool) == len(kernel_size) or _raise(ValueError('kernel and pool sizes must match.'))
    n_dim = len(kernel_size)
    n_dim in (2,3) or _raise(ValueError('resnet_block only 2d or 3d.'))

    conv_layer = Conv2D if n_dim == 2 else Conv3D
    conv_kwargs = dict (
        padding            = 'same',
        use_bias           = not batch_norm,
        kernel_initializer = kernel_initializer,
    )
    channel_axis = -1 if backend_channels_last() else 1

    def f(inp):
        x = conv_layer(n_filter, kernel_size, strides=pool, **conv_kwargs)(inp)
        if batch_norm:
            x = BatchNormalization(axis=channel_axis)(x)
        x = Activation(activation)(x)

        for _ in range(n_conv_per_block-2):
            x = conv_layer(n_filter, kernel_size, **conv_kwargs)(x)
            if batch_norm:
                x = BatchNormalization(axis=channel_axis)(x)
            x = Activation(activation)(x)

        x = conv_layer(n_filter, kernel_size, **conv_kwargs)(x)
        if batch_norm:
            x = BatchNormalization(axis=channel_axis)(x)

        if any(p!=1 for p in pool) or n_filter != K.int_shape(inp)[-1]:
            inp = conv_layer(n_filter, (1,)*n_dim, strides=pool, **conv_kwargs)(inp)

        x = Add()([inp, x])
        x = Activation(activation)(x)
        return x

    return f
