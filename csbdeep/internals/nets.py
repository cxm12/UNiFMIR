from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip

from ..utils.tf import keras_import, keras
Input = keras_import('layers', 'Input')
Model = keras_import('models', 'Model')
# from tensorflow.keras.layers import Dropout
from .blocks import unet_block, unet_blockdropout, conv_block2, conv_block3, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, Cropping2D, Cropping3D, Concatenate, Add, Dropout, Activation, BatchNormalization
import re
Lambda = keras_import('layers', 'Lambda')

from ..utils import _raise, backend_channels_last
import numpy as np
from keras import regularizers

# keras.backend.set_learning_phase(1)
# training = K.learning_phase()
# print('training = K.learning_phase()', training)
istrain = True  # None  # False  #
droptrain = False  # True  # 显示设置dropout的training参数


def common_disnet(n_dim=2, kern_size=3):
    def _build_this(input_shape):
        return custom_disnet(input_shape, (kern_size,)*n_dim)
    return _build_this


def custom_disnet(input_shape, kernel_size=(3,3,3), dropout=0.0):
    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))
    n_dim = len(kernel_size)
    conv_block = conv_block2 if n_dim == 2 else conv_block3

    input = Input(input_shape, name="input")

    # k3n64s1
    f1 = conv_block(64, kernel_size, dropout=dropout, activation="lrelu",
                    batch_norm=False, init="glorot_uniform", name="conv1")(input)

    # k3n64s2
    f2 = Conv3D(64, kernel_size, stride=(2, 2), padding="same", kernel_initializer="glorot_uniform", name="conv2")(f1)
    f2bn = BatchNormalization()(f2)
    f2 = Activation("lrelu")(f2bn)

    # k3n128s1
    f3 = conv_block(128, kernel_size, dropout=dropout, activation="lrelu",
                    batch_norm=True, init="glorot_uniform", name="conv3")(f2)

    # k3n128s2
    f4 = Conv3D(128, kernel_size, stride=(2, 2), padding="same", kernel_initializer="glorot_uniform", name="conv4")(f3)
    f4bn = BatchNormalization()(f4)
    f4 = Activation("lrelu")(f4bn)

    # k3n256s1
    f5 = conv_block(256, kernel_size, dropout=dropout, activation="lrelu",
                    batch_norm=True, init="glorot_uniform", name="conv5")(f4)

    # k3n256s2
    f6 = Conv3D(256, kernel_size, stride=(2, 2), padding="same", kernel_initializer="glorot_uniform", name="conv6")(f5)
    f6bn = BatchNormalization()(f6)
    f6 = Activation("lrelu")(f6bn)

    # k3n512s1
    f7 = conv_block(512, kernel_size, dropout=dropout, activation="lrelu",
                    batch_norm=True, init="glorot_uniform", name="conv7")(f6)

    # k3n512s2
    f8 = Conv3D(512, kernel_size, stride=(2, 2), padding="same", kernel_initializer="glorot_uniform", name="conv8")(f7)
    f8bn = BatchNormalization()(f8)
    f8 = Activation("lrelu")(f8bn)
    
    dense = keras.layers.Dense(1024, "lrelu")(f8)
    final = keras.layers.Dense(1, "sigmoid")(dense)
    
    return Model(inputs=input, outputs=final)


def common_unet(n_dim=2, n_depth=1, kern_size=3, n_first=16, n_channel_out=1, residual=True, prob_out=False,
                last_activation='linear', scale=1):
    """Construct a common CARE neural net based on U-Net [1]_ and residual learning [2]_ to be used for image restoration/enhancement.

    Parameters
    ----------
    n_dim : int
        number of image dimensions (2 or 3)
    n_depth : int
        number of resolution levels of U-Net architecture
    kern_size : int
        size of convolution filter in all image dimensions
    n_first : int
        number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
    n_channel_out : int
        number of channels of the predicted output image
    residual : bool
        if True, model will internally predict the residual w.r.t. the input (typically better)
        requires number of input and output image channels to be equal
    prob_out : bool
        standard regression (False) or probabilistic prediction (True)
        if True, model will predict two values for each input pixel (mean and positive scale value)
    last_activation : str
        name of activation function for the final output layer

    Returns
    -------
    function
        Function to construct the network, which takes as argument the shape of the input image

    Example
    -------
    >>> model = common_unet(2, 1,3,16, 1, True, False)(input_shape)

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
    .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016
    """
    def _build_this(input_shape):
        return custom_unet(input_shape, last_activation, n_depth, n_first, (kern_size,)*n_dim,
                    pool_size=(2,)*n_dim, n_channel_out=n_channel_out, residual=residual, prob_out=prob_out, scale=scale)
    return _build_this


def custom_unet(input_shape,
                last_activation,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3,3,3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.0,
                pool_size=(2,2,2),
                n_channel_out=1,
                residual=False,
                prob_out=False,
                eps_scale=1e-3, scale=1):
    """ TODO """

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

    channel_axis = -1 if backend_channels_last() else 1

    n_dim = len(kernel_size)
    conv = Conv2D if n_dim==2 else Conv3D

    input = Input(input_shape, name="input")
    unet = unet_block(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)

    if scale != 1:
        upsampling = UpSampling2D if n_dim == 2 else UpSampling3D
        unet = upsampling(scale)(unet)
        bic = upsampling(scale)(input)

        final = conv(n_channel_out, (1,) * n_dim, activation='linear')(unet)
        if residual:
            if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
                raise ValueError("number of input and output channels must be the same for a residual net.")
            final = Add()([final, bic])
        final = Activation(activation=last_activation)(final)
    else:
        final = conv(n_channel_out, (1,) * n_dim, activation='linear')(unet)
        if residual:
            if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
                raise ValueError("number of input and output channels must be the same for a residual net.")
            final = Add()([final, input])
        final = Activation(activation=last_activation)(final)

    if prob_out:
        scale = conv(n_channel_out, (1,)*n_dim, activation='softplus')(unet)
        scale = Lambda(lambda x: x+np.float32(eps_scale))(scale)
        final = Concatenate(axis=channel_axis)([final, scale])

    return Model(inputs=input, outputs=final)


def common_unetdropout(n_dim=2, n_depth=1, kern_size=3, n_first=16, n_channel_out=1, residual=True, prob_out=False, last_activation='linear',modeltype='', scale=1):
    """Construct a common CARE neural net based on U-Net [1]_ and residual learning [2]_ to be used for image restoration/enhancement.

    Parameters
    ----------
    n_dim : int
        number of image dimensions (2 or 3)
    n_depth : int
        number of resolution levels of U-Net architecture
    kern_size : int
        size of convolution filter in all image dimensions
    n_first : int
        number of convolution filters for first U-Net resolution level (value is doubled after each downsampling operation)
    n_channel_out : int
        number of channels of the predicted output image
    residual : bool
        if True, model will internally predict the residual w.r.t. the input (typically better)
        requires number of input and output image channels to be equal
    prob_out : bool
        standard regression (False) or probabilistic prediction (True)
        if True, model will predict two values for each input pixel (mean and positive scale value)
    last_activation : str
        name of activation function for the final output layer

    Returns
    -------
    function
        Function to construct the network, which takes as argument the shape of the input image

    Example
    -------
    >>> model = common_unet(2, 1,3,16, 1, True, False)(input_shape)

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
    .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016
    """
    def _build_this(input_shape):
        return custom_unetdropout(input_shape, last_activation, n_depth, n_first, (kern_size,)*n_dim,
                    pool_size=(2,)*n_dim, n_channel_out=n_channel_out, residual=residual,
                        prob_out=prob_out, dropouttype=modeltype, scale=scale)
    return _build_this


def custom_unetdropout(input_shape,
                last_activation,
                n_depth=2,
                n_filter_base=16,
                kernel_size=(3,3,3),
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.7,
                pool_size=(2,2,2),
                n_channel_out=1,
                residual=False,
                prob_out=False,
                eps_scale=1e-3,
                dropouttype='_dropout', scale=1):
    """ TODO """

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

    channel_axis = -1 if backend_channels_last() else 1

    n_dim = len(kernel_size)
    conv = Conv2D if n_dim==2 else Conv3D

    ## define model :
    input = Input(input_shape, name="input")
    
    unet = unet_blockdropout(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size, dropouttype=dropouttype)(input)

    if scale != 1:
        upsampling = UpSampling2D if n_dim == 2 else UpSampling3D
        unet = upsampling(scale)(unet)
        bic = upsampling(scale)(input)

        final = conv(n_channel_out, (1,) * n_dim, activation='linear')(unet)
        if residual:
            if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
                raise ValueError("number of input and output channels must be the same for a residual net.")
            final = Add()([final, bic])
        final = Activation(activation=last_activation)(final)
    else:
        final = conv(n_channel_out, (1,) * n_dim, activation='linear')(unet)
        if residual:
            if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
                raise ValueError("number of input and output channels must be the same for a residual net.")
            final = Add()([final, input])
        final = Activation(activation=last_activation)(final)

    ## 输出概率分布：
    if prob_out:
        scale = conv(n_channel_out, (1,)*n_dim, activation='softplus')(unet)
        scale = Lambda(lambda x: x+np.float32(eps_scale))(scale)
        final = Concatenate(axis=channel_axis)([final, scale])

    return Model(inputs=input, outputs=final)


###  ------------------ FeedBack SISR --------------------- ###
def common_fbdropout(n_dim=2, kern_size=3, n_first=16, n_channel_out=1, residual=True,
                       last_activation='linear', isdropout=True, step=3, scale=1):
    def _build_this(input_shape):
        return custom_fbdropout(input_shape, last_activation, n_first, (kern_size,) * n_dim,
            n_channel_out=n_channel_out, residual=residual, isdropout=isdropout, step=step, scale=scale)
    
    return _build_this


def custom_fbdropout(input_shape,
                       last_activation='linear',
                       n_filter_base=16,
                       kernel_size=(3, 3, 3),
                       dropout=0.7,
                       n_channel_out=1,
                       residual=True, isdropout=True, step=3, scale=1):
    
    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")
    
    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))
    
    n_dim = len(kernel_size)
    conv = Conv2D if n_dim == 2 else Conv3D
    pooling = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D
    channel_axis = -1 if backend_channels_last() else 1
    border_mode = "same"
    init = "glorot_uniform"

    ## define model :
    input = Input(input_shape, name="input")
    
    f = conv(n_filter_base//4, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='conv0')(input)
    # sub_layer = Lambda(lambda x: tf.space_to_depth(x, 2))
    # f = sub_layer(inputs=f)
    fin = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='convin')(f)

    out = []
    for i in range(step):
        if i == 0:
            f0 = Concatenate(axis=channel_axis)([fin, fin])
        f1 = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='conv1_%d'%i)(f0)
        up1 = f1  # upsampling(2)(f1)  #
        fup1 = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='conu1_%d'%i)(up1)
        dn1 = fup1  # pooling(2, name="max_%d" % i)(fup1)  #
        fdn1 = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='convd1_%d'%i)(dn1)

        up2 = Concatenate(axis=channel_axis)([fdn1, f1])  # upsampling(2)(Concatenate(axis=channel_axis)([fdn1, f1]))
        fup2 = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='conu2_%d'%i)(up2)
        dn2 = Concatenate(axis=channel_axis)([fup2, fup1])  # pooling(2, name="max2_%d" % i)(Concatenate(axis=channel_axis)([fup2, fup1]))
        fdn2 = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='cond2_%d'%i)(dn2)

        fcat = Concatenate(axis=channel_axis)([fdn2, fdn1])
        fcat = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='con2_%d'%i)(fcat)  # np.int32(n_filter_base//dropout)
        if isdropout:
            if droptrain:
                fcat = Dropout(dropout)(fcat, training=istrain)
            else:
                fcat = Dropout(dropout)(fcat)
            
        fout = Add()([fcat, f1])

        f0 = Concatenate(axis=channel_axis)([fin, fout])
        out.append(fout)
    
    if step == 1:
        fstep = out[0]
    else:
        fstep = Concatenate(axis=channel_axis)(out)

    if isdropout:
        fd = conv(np.int32(np.ceil((n_filter_base * 4)//0.5)), (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='concat')(fstep)
        if droptrain:
            fd = Dropout(dropout)(fd, training=istrain)
        else:
            fd = Dropout(dropout)(fd)
    else:
        fd = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='concat')(fstep)
        # fd = conv(n_filter_base * 4, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu', name='concat')(fstep)
    # sub_layer = Lambda(lambda x: tf.depth_to_space(x, 2))
    # fd = sub_layer(inputs=fd)
    
    if scale != 1:
        fd = upsampling(scale)(fd)
        bic = upsampling(scale)(input)
        final = conv(n_channel_out, (1,) * n_dim, padding=border_mode, kernel_initializer=init, activation='linear',
                     name='conout')(fd)
        if residual:
            if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
                raise ValueError("number of input and output channels must be the same for a residual net.")
            final = Add()([final, bic])
    else:
        final = conv(n_channel_out, (1,) * n_dim, padding=border_mode, kernel_initializer=init, activation='linear',
                     name='conout')(fd)
        if residual:
            if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
                raise ValueError("number of input and output channels must be the same for a residual net.")
            final = Add()([final, input])
            # final = Activation(activation=last_activation)(final)
    
    return Model(inputs=input, outputs=final)


###  ------------------ SRCNN SISR --------------------- ###
def common_srcnndropout(n_dim=2, kern_size=3, n_channel_out=1, last_activation='linear', dropouttype='_dropout', scale=1):
    def _build_this(input_shape):
        return custom_srcnndropout(input_shape, last_activation, (kern_size,) * n_dim,
                                   n_channel_out=n_channel_out, dropouttype=dropouttype, scale=scale)
    
    return _build_this


def custom_srcnndropout(input_shape,
                        last_activation='linear',
                        kernel_size=(3, 3, 3),
                        dropout=0.7,
                        n_channel_out=1, dropouttype='_dropout', scale=1):
    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")
    
    keras.backend.set_learning_phase(1)
    

    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))
    
    n_dim = len(kernel_size)
    conv = Conv2D if n_dim == 2 else Conv3D
    border_mode = "same"
    init = "glorot_uniform"
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    ## define model :
    input = Input(input_shape, name="input")
    
    f = conv(64, (9,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu',
             name='conv0')(input)
    
    f2 = conv(32, (1,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu',
              name='convin')(f)
    
    
    if 'dropout' in dropouttype:
        if droptrain:
            f2 = Dropout(dropout)(f2, training=istrain)
        else:
            f2 = Dropout(dropout)(f2)
    
    if scale != 1:
        f2 = upsampling(scale)(f2)
        final = conv(n_channel_out, (5,) * n_dim, padding=border_mode, kernel_initializer=init, activation='linear',
                     name='conout')(f2)
    else:
        final = conv(n_channel_out, (5,) * n_dim, padding=border_mode, kernel_initializer=init, activation='linear',
                     name='conout')(f2)
            
    return Model(inputs=input, outputs=final)


###  ------------------ SRCNN SISR --------------------- ###
def act_func(x, delta_T = 0.08, sat_I = 7.268, ns_T = 0.572):
    y = x * (1 - delta_T * tf.exp(-x / sat_I) - ns_T)
    return y


def srcnn(n_channel_out=1, activation="relu"):
    def _build_this(input_shape):
        kernel_size = (3, 3, 3)
        keras.backend.set_learning_phase(1)
    
        all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))
    
        n_dim = len(kernel_size)
        conv = Conv2D if n_dim == 2 else Conv3D
        border_mode = "same"
        init = "glorot_uniform"
    
        ## define model :
        input = Input(input_shape, name="input")

        bic = UpSampling2D(size=(2, 2), data_format=None, interpolation='bicubic')(input)  # tf.keras.layers.
        # bic = tensorlayer.prepro.zoom(input, zoom_range=(0.9, 1.1), is_random=False, row_index=0,
        #                               col_index=1, channel_index=2, fill_mode='nearest', cval=0.0, order=1)
        input = bic
        
        if activation == 'relu':
            f = conv(64, (9,) * n_dim, padding=border_mode, kernel_initializer=init, activation=activation,
                     name='conv0')(input)
    
            f2 = conv(32, (1,) * n_dim, padding=border_mode, kernel_initializer=init, activation=activation,
                      name='convin')(f)
        elif activation == 'swish':
            f = conv(64, (9,) * n_dim, padding=border_mode, kernel_initializer=init, activation=None,
                     name='conv0')(input)
            f = tf.nn.swish(f)
            f2 = conv(32, (1,) * n_dim, padding=border_mode, kernel_initializer=init, activation=None,
                      name='convin')(f)
            f2 = tf.nn.swish(f2)
        elif activation == 'our1':
            f = conv(64, (9,) * n_dim, padding=border_mode, kernel_initializer=init, activation=None,
                     name='conv0')(input)
            f = act_func(f, delta_T=0.3, sat_I=8.116, ns_T=0.532)
            f2 = conv(32, (1,) * n_dim, padding=border_mode, kernel_initializer=init, activation=None,
                      name='convin')(f)
            f2 = act_func(f2, delta_T=0.3, sat_I=8.116, ns_T=0.532)
        elif activation == 'our2':
            f = conv(64, (9,) * n_dim, padding=border_mode, kernel_initializer=init, activation=None,
                     name='conv0')(input)
            f = act_func(f, delta_T=0.43, sat_I=9.620, ns_T=0.470)
            f2 = conv(32, (1,) * n_dim, padding=border_mode, kernel_initializer=init, activation=None,
                      name='convin')(f)
            f2 = act_func(f2, delta_T=0.43, sat_I=9.620, ns_T=0.470)

        
        final = conv(n_channel_out, (5,) * n_dim, padding=border_mode, kernel_initializer=init, activation='linear',
                     name='conout')(f2)
    
        return Model(inputs=input, outputs=final)
    return _build_this


###  ------------------ EDSR SISR --------------------- ###
def common_edsrdropout(n_dim=2, kern_size=3, n_channel_out=1,
                        last_activation='linear', dropouttype='_dropout'):
    def _build_this(input_shape):
        return custom_edsrdropout(input_shape, last_activation, (kern_size,) * n_dim,
                                   n_channel_out=n_channel_out, dropouttype=dropouttype)
    
    return _build_this


def custom_edsrdropout(input_shape,
                        last_activation='linear',
                        kernel_size=(3, 3, 3),
                        dropout=0.7,
                        n_channel_out=1, dropouttype='_dropout',
                       n_filter_base=32,  # n_filter_base=256,
                       n_resblocks=8):  # n_resblocks=32):
    
    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")
    
    all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))
    
    n_dim = len(kernel_size)
    conv = Conv2D if n_dim == 2 else Conv3D
    border_mode = "same"
    init = "glorot_uniform"
    
    ## define model :
    input = Input(input_shape, name="input")
    
    def res_block(input_tensor, nf, res_scale=1.0, name='resblock'):
        x = conv(nf, (3,) * n_dim, padding=border_mode, kernel_initializer=init, activation='relu',
                   activity_regularizer=regularizers.l1(10e-10), name=name+'convin1')(input_tensor)
        x = conv(nf, (3,) * n_dim, padding=border_mode, kernel_initializer=init,
                 activity_regularizer=regularizers.l1(10e-10), name=name+'convin2')(x)
        x = Lambda(lambda x: x * res_scale)(x)
        x = Add()([x, input_tensor])
        return x

    x = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init,
             activity_regularizer=regularizers.l1(10e-10), name='conv1')(input)
    conv1 = x
    if n_filter_base == 256:
        res_scale = 0.1
    else:
        res_scale = 1.0
    for i in range(n_resblocks//2-1): x = res_block(x, n_filter_base, res_scale, name='resblock%d' % i)
    if 'dropout' in dropouttype:
        if droptrain:
            x = Dropout(dropout)(x, training=istrain)
        else:
            x = Dropout(dropout)(x)
    
    x = res_block(x, n_filter_base, res_scale, name='resblock%d' % ((n_resblocks // 2)-1))
    if 'dropout' in dropouttype:
        if droptrain:
            x = Dropout(dropout)(x, training=istrain)
        else:
            x = Dropout(dropout)(x)
    x = res_block(x, n_filter_base, res_scale, name='resblock%d' % ((n_resblocks // 2)))
    if 'dropout' in dropouttype:
        if droptrain:
            x = Dropout(dropout)(x, training=istrain)
        else:
            x = Dropout(dropout)(x)
        
    for i in range((n_resblocks//2)-1): x = res_block(x, n_filter_base, res_scale, name='resblock%d' % (i+(n_resblocks//2)+1))
   
    
    x = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init,
             activity_regularizer=regularizers.l1(10e-10), name='conv2')(x)
    x = Add()([x, conv1])

    x = conv(n_filter_base, (3,) * n_dim, padding=border_mode, kernel_initializer=init,
             activity_regularizer=regularizers.l1(10e-10), name='conv3')(x)
    
    final = conv(n_channel_out, (1,) * n_dim, padding=border_mode, kernel_initializer=init, activation='linear',
              activity_regularizer=regularizers.l1(10e-10), name='conv4')(x)

    return Model(inputs=input, outputs=final)


modelname = re.compile("^(?P<model>resunet|unet)(?P<n_dim>\d)(?P<prob_out>p)?_(?P<n_depth>\d+)_(?P<kern_size>\d+)_(?P<n_first>\d+)(_(?P<n_channel_out>\d+)out)?(_(?P<last_activation>.+)-last)?$")
def common_unet_by_name(model):
    r"""Shorthand notation for equivalent use of :func:`common_unet`.

    Parameters
    ----------
    model : str
        define model to be created via string, which is parsed as a regular expression:
        `^(?P<model>resunet|unet)(?P<n_dim>\d)(?P<prob_out>p)?_(?P<n_depth>\d+)_(?P<kern_size>\d+)_(?P<n_first>\d+)(_(?P<n_channel_out>\d+)out)?(_(?P<last_activation>.+)-last)?$`

    Returns
    -------
    function
        Calls :func:`common_unet` with the respective parameters.

    Raises
    ------
    ValueError
        If argument `model` is not a valid string according to the regular expression.

    Example
    -------
    >>> model = common_unet_by_name('resunet2_1_3_16_1out')(input_shape)
    >>> # equivalent to: model = common_unet(2, 1,3,16, 1, True, False)(input_shape)

    Todo
    ----
    Backslashes in docstring for regexp not rendered correctly.

    """
    m = modelname.fullmatch(model)
    if m is None:
        raise ValueError("model name '%s' unknown, must follow pattern '%s'" % (model, modelname.pattern))
    # from pprint import pprint
    # pprint(m.groupdict())
    options = {k:int(m.group(k)) for k in ['n_depth','n_first','kern_size']}
    options['prob_out'] = m.group('prob_out') is not None
    options['residual'] = {'unet': False, 'resunet': True}[m.group('model')]
    options['n_dim'] = int(m.group('n_dim'))
    options['n_channel_out'] = 1 if m.group('n_channel_out') is None else int(m.group('n_channel_out'))
    if m.group('last_activation') is not None:
        options['last_activation'] = m.group('last_activation')

    return common_unet(**options)



def receptive_field_unet(n_depth, kern_size, pool_size=2, n_dim=2, img_size=1024):
    """Receptive field for U-Net model (pre/post for each dimension)."""
    x = np.zeros((1,)+(img_size,)*n_dim+(1,))
    mid = tuple([s//2 for s in x.shape[1:-1]])
    x[(slice(None),) + mid + (slice(None),)] = 1
    model = custom_unet (
        x.shape[1:],
        n_depth=n_depth, kernel_size=[kern_size]*n_dim, pool_size=[pool_size]*n_dim,
        n_filter_base=8, activation='linear', last_activation='linear',
    )
    y  = model.predict(x)[0,...,0]
    y0 = model.predict(0*x)[0,...,0]
    ind = np.where(np.abs(y-y0)>0)
    return [(m-np.min(i),np.max(i)-m) for (m,i) in zip(mid,ind)]
