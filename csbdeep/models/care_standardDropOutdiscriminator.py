# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division

import warnings
from .config import Config
from .base_model import BaseModel, suppress_without_basedir

from ..utils import _raise, axes_check_and_normalize, axes_dict
from ..utils.tf import IS_TF_1, keras_import, keras
from ..data import PercentileNormalizer
from ..data import PadAndCropResizer
from ..internals.predict import predict_tiled, tile_overlap, Progress, total_n_tiles
from ..internals import nets, train

from ..utils.tf import keras, callbacks

Model = keras_import('models', 'Model')
Input, Dense, BatchNormalization, Conv2D, Conv3D, MaxPooling3D, UpSampling3D, Lambda, Multiply, LeakyReLU = \
    keras_import('layers', 'Input', 'Dense', 'BatchNormalization', 'Conv2D', 'Conv3D', 'MaxPooling3D', 'UpSampling3D', 'Lambda', 'Multiply', 'LeakyReLU')

from csbdeep.internals.dataloader_mcx import DataLoader
TerminateOnNaN = keras_import('callbacks', 'TerminateOnNaN')
import datetime
import os
from tifffile import imread, imsave
from csbdeep.func_mcx import *
from csbdeep.utils import normalize


# https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/
class SnapshotEnsemble(callbacks.Callback):
    
    def __init__(self, n_epochs, n_cycles, lrate_max):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()
    
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = n_epochs // n_cycles
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max / 2 * (np.cos(cos_inner) + 1)
    
    def on_epoch_begin(self, epoch, logs={}):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        print(f'epoch {epoch + 1}, lr {lr}')
        K.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)
    
    def on_epoch_end(self, epoch, logs={}):
        epochs_per_cycle = self.epochs // self.cycles
        # if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
        # filename = f"snapshot_model_{int((epoch + 1) / epochs_per_cycle)}.h5"
        # self.model.save(filename)
        # print(f'>saved snapshot {filename}, epoch {epoch}')


class CAREDropOutDis(BaseModel):
    
    def __init__(self, config, name=None, basedir='.', modeltype='', testset='', epoch=200, istrain=True):
        """See class docstring."""
        self.modeltype = modeltype
        self.testset = testset
        self.istrain = istrain
        super(CAREDropOutDis, self).__init__(config=config, name=name, basedir=basedir)
        if istrain:
            self.discriminator = self.build_discriminator()  # _buildDis() # self.keras_model = self._build()
        self.epochs = epoch

        self.savepath = str(self.logdir)
        self.savepathd = str(self.logdir) + '/Dis/'
        print(self.savepath, '\n', self.logdir)
        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(self.savepathd, exist_ok=True)

        trainstart = self.finetune_and_load_weights(prefer='_ep')
        if trainstart:
            self.startepoch = 0
            print('!!! Train From Start !!!')


    def finetune_and_load_weights(self, prefer=''):
        from itertools import chain
        # get all weight files and sort by modification time descending (newest first)
        weights_ext   = ('*.h5', '*.hdf5')
        weights_files = chain(*(self.logdir.glob(ext) for ext in weights_ext))  # glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合
        weights_files = reversed(sorted(weights_files, key=lambda f: f.stat().st_mtime))
        weights_files = list(weights_files)
        if len(weights_files) == 0:
            warnings.warn("Couldn't find any network weights (%s) to load." % ', '.join(weights_ext))
            return True
        weights_preferred = list(filter(lambda f: prefer in f.name, weights_files))
        weights_chosen = weights_preferred[0] if len(weights_preferred)>0 else weights_files[0]
        self.load_weights(weights_chosen.name)
        print("Loading G network weights from '%s/%s'." % (self.logdir, weights_chosen.name))
        self.startepoch = np.int(weights_chosen.name[weights_chosen.name.find('_ep')+3:-3])
        if self.istrain:
            print('******** Load D from ', self.savepathd+weights_chosen.name, '**********')
            self.discriminator.load_weights(self.savepathd+weights_chosen.name, by_name=True)
        return False
        

    def _build(self):
        if '_FB' in self.modeltype:
            if 'dropout' in self.modeltype:
                print('!!! model FeedBack_dropout !!!')
                return nets.common_fbdropout(
                    n_dim=self.config.n_dim,
                    n_channel_out=self.config.n_channel_out,
                    residual=self.config.unet_residual,
                    kern_size=self.config.unet_kern_size,
                    n_first=self.config.unet_n_first,
                    last_activation=self.config.unet_last_activation, isdropout=True, step=3
                )(self.config.unet_input_shape)
            else:
                print('!!! model FeedBack !!!')
                return nets.common_fbdropout(
                    n_dim=self.config.n_dim,
                    n_channel_out=self.config.n_channel_out,
                    residual=self.config.unet_residual,
                    kern_size=self.config.unet_kern_size,
                    n_first=self.config.unet_n_first,
                    last_activation=self.config.unet_last_activation, isdropout=False, step=3
                )(self.config.unet_input_shape)
        # if '_FB' in self.modeltype:
        #     if 'dropout' in self.modeltype:
        #         print('!!! model FeedBack_dropout !!!')
        #         return nets.common_fbdropout(
        #             n_dim=self.config.n_dim,
        #             n_channel_out=self.config.n_channel_out,
        #             residual=self.config.unet_residual,
        #             kern_size=self.config.unet_kern_size,
        #             n_first=self.config.unet_n_first,
        #             last_activation=self.config.unet_last_activation, isdropout=True, step=3
        #         )(self.config.unet_input_shape)
        #     else:
        #         print('!!! model FeedBack !!!')
        #         return nets.common_fbdropout(
        #             n_dim=self.config.n_dim,
        #             n_channel_out=self.config.n_channel_out,
        #             residual=self.config.unet_residual,
        #             kern_size=self.config.unet_kern_size,
        #             n_first=self.config.unet_n_first,
        #             last_activation=self.config.unet_last_activation, isdropout=False
        #         )(self.config.unet_input_shape)
        elif 'srcnn' in self.modeltype:
            if 'dropout' in self.modeltype:
                print('!!! model SRCNN_Dropout !!!')
                return nets.common_srcnndropout(
                    n_dim=self.config.n_dim,
                    n_channel_out=self.config.n_channel_out,
                    kern_size=self.config.unet_kern_size,
                    last_activation=self.config.unet_last_activation,
                    dropouttype='_dropout'
                )(self.config.unet_input_shape)
            else:
                print('!!! model SRCNN !!!')
                return nets.common_srcnndropout(
                    n_dim=self.config.n_dim,
                    n_channel_out=self.config.n_channel_out,
                    kern_size=self.config.unet_kern_size,
                    last_activation=self.config.unet_last_activation,
                    dropouttype=''
                )(self.config.unet_input_shape)
        elif 'edsr' in self.modeltype:
            if 'dropout' in self.modeltype:
                print('!!! model EDSR_Dropout !!!')
                return nets.common_edsrdropout(
                    n_dim=self.config.n_dim,
                    n_channel_out=self.config.n_channel_out,
                    kern_size=self.config.unet_kern_size,
                    last_activation=self.config.unet_last_activation,
                    dropouttype='_dropout'
                )(self.config.unet_input_shape)
            else:
                print('!!! model EDSR !!!')
                return nets.common_edsrdropout(
                    n_dim=self.config.n_dim,
                    n_channel_out=self.config.n_channel_out,
                    kern_size=self.config.unet_kern_size,
                    last_activation=self.config.unet_last_activation,
                    dropouttype=''
                )(self.config.unet_input_shape)
        elif 'dropout' in self.modeltype:
            return nets.common_unetdropout(
                n_dim=self.config.n_dim,
                n_channel_out=self.config.n_channel_out,
                prob_out=self.config.probabilistic,
                residual=self.config.unet_residual,
                n_depth=self.config.unet_n_depth,
                kern_size=self.config.unet_kern_size,
                n_first=self.config.unet_n_first,
                last_activation=self.config.unet_last_activation,
                modeltype='_dropout',  # self.modeltype  # _dropout_v2
            )(self.config.unet_input_shape)
        
        else:
            return nets.common_unet(
                n_dim=self.config.n_dim,
                n_channel_out=self.config.n_channel_out,
                prob_out=self.config.probabilistic,
                residual=self.config.unet_residual,
                n_depth=self.config.unet_n_depth,
                kern_size=self.config.unet_kern_size,
                n_first=self.config.unet_n_first,
                last_activation=self.config.unet_last_activation,
            )(self.config.unet_input_shape)


    def _buildDis(self):
        return nets.common_disnet(
            n_dim           = self.config.n_dim,
            kern_size       = self.config.unet_kern_size,
        )(self.config.unet_input_shape)


    def build_discriminator(self):
        n_dim = 3
        conv = Conv2D if n_dim == 2 else Conv3D
        self.df = 64

        def d_block(layer_input, filters, strides=(1,1,1), bn=True):
            """Discriminator layer"""
            # final = conv(n_channel_out, (1,) * n_dim, activation='linear')(unet)
            d = conv(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
    
        # Input img
        d0 = Input(shape=self.config.unet_input_shape)
    
        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=(1,2,2))
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=(1,2,2))
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=(1,2,2))
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=(1,2,2))
    
        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
    
        return Model(d0, validity)
    
    
    def calculate_learning_rate(self, epoch):
        if epoch <= 100:
            return self.config.train_learning_rate
        else:
            decay_ep = 20
            decayrate = 0.5 ** (epoch-100)//decay_ep
            return self.config.train_learning_rate * decayrate


    def prepare_for_training(self, optimizer=None):
        if optimizer is None:
            Adam = keras_import('optimizers', 'Adam')
            self.optimizer = Adam(lr=self.config.train_learning_rate)
    
        # Build and compile the discriminator
        self.discriminator.compile(loss='mse',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
    
        # High res. and low res. images
        img_hr = Input(shape=self.config.unet_input_shape, name='HR')
        img_lr = Input(shape=self.config.unet_input_shape, name='LR')
    
        # Generate high res. version from low res.
        fake_hr = self.keras_model(img_lr)
    
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
    
        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)
        # print('validity.shape', validity.shape)  # (?, ?, ?, ?, 1)
    
        self.combined = Model([img_lr, img_hr], [validity, fake_hr])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],  # [1e-1, 1],  #
                              optimizer=self.optimizer)
        
        # self.callbacksDis = [TerminateOnNaN()]
        # self.callbacks = [TerminateOnNaN()]
        # if self.config.train_reduce_lr is not None:
        #     # # 训练过程中缩小学习率： # 余弦退火缩小学习率：
        #     reduce_lr = SnapshotEnsemble(n_epochs=self.config.train_epochs, n_cycles=self.config.train_epochs / 3,
        #                                  lrate_max=self.config.train_learning_rate)
        #     self.callbacks.insert(0, reduce_lr)
        #
        #     ReduceLROnPlateau = keras_import('callbacks', 'ReduceLROnPlateau')
        #     rlrop_params = self.config.train_reduce_lr
        #     if 'verbose' not in rlrop_params:
        #         rlrop_params['verbose'] = True
        #     # TF2: add as first callback to put 'lr' in the logs for TensorBoard
        #     self.callbacksDis.insert(0, ReduceLROnPlateau(**rlrop_params))
        
        self._model_prepared = True


    def prepare_for_trainingor(self, optimizer=None, **kwargs):
        if optimizer is None:
            Adam = keras_import('optimizers', 'Adam')
            optimizer = Adam(lr=self.config.train_learning_rate)
    
        # Build and compile the discriminator
        self.discriminator = self._buildDis()  # self.build_discriminator()
    
        ## define loss + model.compile(optimizer=optimizer, loss='mae'):
        self.callbacks = train.prepare_modelgenerator(self.keras_model, self.keras_modelDis, optimizer,
                                                      self.config.train_loss, **kwargs)
        self.callbacksDis = train.prepare_modeldiscriminator(self.keras_model, optimizer, 'disloss', **kwargs)
    
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()
        
            if self.config.train_tensorboard:
                if IS_TF_1:
                    from ..utils.tf import CARETensorBoard
                    self.callbacks.append(
                        CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3,
                                        write_images=True, prob_out=self.config.probabilistic))
                else:
                    from tensorflow.keras.callbacks import TensorBoard
                    self.callbacks.append(
                        TensorBoard(log_dir=str(self.logdir / 'logs'), write_graph=False, profile_batch=0))
    
        if self.config.train_reduce_lr is not None:
            # # 训练过程中缩小学习率：
            # 余弦退火缩小学习率：
            reduce_lr = SnapshotEnsemble(n_epochs=self.config.train_epochs, n_cycles=self.config.train_epochs / 3,
                                         lrate_max=self.config.train_learning_rate)
            self.callbacks.insert(0, reduce_lr)
        
            ReduceLROnPlateau = keras_import('callbacks', 'ReduceLROnPlateau')
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            self.callbacksDis.insert(0, ReduceLROnPlateau(**rlrop_params))
    
        self._model_prepared = True

    
    def trainsrgan(self, X, Y, validation_data):
        start_time = datetime.datetime.now()
        ((isinstance(validation_data, (list, tuple)) and len(validation_data) == 2)
         or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        h = X.shape[2]
        # print('X.shape', X.shape)  # (17005, 16, 64, 64, 1)
        patch = int(h / 2 ** 4)
        self.disc_patch = (X.shape[1], patch, patch, 1)
        
        n_train = len(X)
        # axes = axes_check_and_normalize('S' + self.config.axes, X.ndim)
        # ax = axes_dict(axes)
        
        if not self._model_prepared:
            self.prepare_for_training()

        steps_per_epoch = n_train // self.config.train_batch_size  # 1  #
        # training_data = train.DataWrapper(X, Y, self.config.train_batch_size, length=epochs * steps_per_epoch)
        self.data_loader = DataLoader()
        maxps = 0
        maxss = 0
        maxep = 0
        for epoch in range(self.startepoch, self.epochs):
            ## set learning rate
            lr = K.get_value(self.optimizer.lr)  # 获取当前学习率
            current_learning_rate = self.calculate_learning_rate(epoch)
            K.set_value(self.optimizer.lr, current_learning_rate)  # set new lr
    
            for iteration in range(steps_per_epoch):
                
                # ----------------------
                #  Train Discriminator
                # ----------------------
    
                # Sample images and their conditioning counterparts
                
                # print(iter(training_data))
                # [imgs_lr, imgs_hr] = iter(training_data)
                imgs_hr, imgs_lr = self.data_loader.load_data(X, Y, start=iteration, len=n_train, batch_size=self.config.train_batch_size)
    
                # From low res. image generate high res. version
                fake_hr = self.keras_model.predict(imgs_lr)
                
                valid = np.ones((self.config.train_batch_size,) + self.disc_patch)
                fake = np.zeros((self.config.train_batch_size,) + self.disc_patch)
                # print('valid.shape', valid.shape)
    
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # loss; acc
    
                # ------------------
                #  Train Generator
                # ------------------
    
                # Sample images and their conditioning counterparts
                imgs_hr, imgs_lr = self.data_loader.load_data(X, Y, start=iteration, len=n_train, batch_size=self.config.train_batch_size)
                # [imgs_lr, imgs_hr] = iter(training_data)
    
                # The generators want the discriminators to label the generated images as real
                valid = np.ones((self.config.train_batch_size,) + self.disc_patch)
    
                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, imgs_hr])  # loss; model2_loss; model1_loss
                
                if iteration % 200 == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    print("EPOCH%d , ite%d(/%d) time: %s" % (epoch, iteration, steps_per_epoch, elapsed_time), '; Dloss/acc=', d_loss, '; Gloss=', g_loss)

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("EP_%d time: %s; learning rate=%f" % (epoch, elapsed_time, lr))
            # print("%d time: %s; Dloss %f; Gloss %f" % (epoch, elapsed_time, d_loss, g_loss))
            if epoch % 5 == 0:
                self.export_TF(self.modeltype, self.testset, epoch)
                meanps, meanss = self.Valid_on_testset()
                if maxps < meanps:
                    maxps = meanps
                    maxss = meanss
                    maxep = epoch
                    print('Current model', maxep, 'Turns to be Max PSNR; max psnr=', maxps, maxss)
                else:
                    print('Current model', epoch, ' psnr=', meanps, meanss)
                    print('Not Change Max PSNR model', maxep, 'max psnr=', maxps)
        print('Trained END \n model', maxep, ' Has Max PSNR/SSIM = ', maxps, maxss)


    def Valid_on_testset(self):
        spath = self.savepath + '/Valid/'
        os.makedirs(spath, exist_ok=True)
        print('*******************!!! Start VAlidation on Testset !!!*************************')
        if '_dropout' in self.modeltype:
            testnum = 10  # 3  #
        else:
            testnum = 1
        testset = 'Denoising_Planaria'
        level = 'condition_1'

        testdatapath = '../../DataSet/' + testset + '/test_data/' + level + '/'
        testGTpath = '../../DataSet/' + testset + '/test_data/GT/'
    
        meanpsnr = 0
        meanssim = 0
    
        imnum = 4  # len(filename)  #
        namelst = ['EXP278_Smed_fixed_RedDot1_sub_5_N7_m0004', 'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0010',
                   'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0007', 'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0013']
        for i in range(imnum):
            name = namelst[i]
            # print(name)
            y = imread(testGTpath + name + '.tif')
            x = imread(testdatapath + name + '.tif')
        
            if testset == 'Denoising_Planaria':
                [z, h, w] = x.shape
                axes = 'ZYX'
                
            if '_dropout' in self.modeltype:
                # 将模式转为训练模式，在测试时才能使用dropout!!!
                keras.backend.set_learning_phase(1)
            # # CARE model
            # if '_GAN' in modeltype:
            #     model = CAREDropOutDis(config=None, name='my_model%s/' % (modeltype) + testset, basedir='models',
            #                            modeltype=modeltype, testset=testset)
            
        
            # ## Apply CARE network to raw image
            resultlst = []
            x = x[(z - 10):(z - 5), (h // 4):(h - h // 4), (w // 4):(w - w // 4)]
            y = y[(z - 10):(z - 5), (h // 4):(h - h // 4), (w // 4):(w - w // 4)]
            for ti in range(testnum):
                restoredraw = self.predict(x, axes, n_tiles=(1, 4, 4))
                restored = normalize(restoredraw, 0.1, 99.9) * 255
                resultlst.append(restored)
        
            array = np.array(resultlst)  # [testnum, 95, 256, 256]
            mean = np.mean(array, axis=0)
            
            for z in range(mean.shape[0]):
                savecolorim(spath + name + '-dZ%d' % z + '.png', np.uint8(mean[z, :, :]))
                dif = np.abs(normalize(y[z, :, :], 0, 100) * 255 - restored[z, :, :])
                imsave(spath + name + '-dZ%d' % z + 'Dif.png', dif)
                
            restored = np.swapaxes(restored, 0, 2)
            y = np.swapaxes(y, 0, 2)
            mean = np.swapaxes(mean, 0, 2)
            psm, ssmm = compute_psnr_and_ssim(mean, normalize(y, 0.1, 99.9) * 255)
            # print('Image - ', name, '- PSNR/SSIM = ', ps, ssm, '- PSNR/SSIM of mean test = ', psm, ssmm)
            meanpsnr += psm
            meanssim += ssmm
        print('meanssim, meanpsnr', meanssim / imnum, meanpsnr / imnum)
        print('******************* !!! End VAlidation on Testset !!! *************************')
        return meanpsnr / imnum, meanssim / imnum


    @suppress_without_basedir(warn=True)
    def export_TF(self, modeltype, testset, ep):
        """Export neural network via :func:`csbdeep.utils.tf.export_SavedModel`.

        Parameters
        ----------
        fname : str or None
            Path of the created SavedModel archive (will end with ".zip").
            If ``None``, "<model-directory>/TF_SavedModel.zip" will be used.

        """
        fn = self.savepath + 'TF_SavedModel_ep' + str(ep) + '.h5'
        fnd = self.savepathd + 'TF_SavedModel_ep' + str(ep) + '.h5'
        
        self.keras_model.save(fn)
        self.discriminator.save(fnd)
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % fn)
        # fname = './models/my_model%s/%s/TF_SavedModel%d.zip' % (modeltype, testset, ep)
        # fnamed = './models/my_model%s/%s/Dis/TF_SavedModel%d_dis.zip' % (modeltype, testset, ep)
        # export_SavedModel(self.discriminator, str(fnamed), meta=meta)
        # export_SavedModel(self.keras_model, str(fname), meta=meta)
        # print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fname.resolve()))
        # self._training_finished()  # save weight_best/last.h5


    def predict(self, img, axes, normalizer=PercentileNormalizer(),
                resizer=PadAndCropResizer(), n_tiles=None):
        """Apply neural network to raw image to predict restored image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image
        axes : str
            Axes of the input ``img``.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            Normalization of input image before prediction and (potentially) transformation back after prediction.
        resizer : :class:`csbdeep.data.Resizer` or None
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that can then be processed independently and re-assembled to yield the restored image.
            This parameter denotes a tuple of the number of tiles for every image axis.
            Note that if the number of tiles is too low, it is adaptively increased until
            OOM errors are avoided, albeit at the expense of runtime.
            A value of ``None`` denotes that no tiling should initially be used.

        Returns
        -------
        :class:`numpy.ndarray`
            Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
            the predicted per-pixel Laplace distributions (i.e., the expected restored image).
            Axes semantics are the same as in the input image. Only if the output is multi-channel and
            the input image didn't have a channel axis, then output channels are appended at the end.

        """
        
        ## Load model :base_model._find_and_load_weights() ==
        # self.fname = './models/my_model%s/%s/TF_SavedModel.h5' % (self.modeltype, self.testset)
        # ep = 0
        # fn = self.fname[:self.fname.rfind('.h5')] + 'ep_' + str(ep) + '.h5'
        # self.keras_model.load_weights(fn)
        # print('Load h5 model from %s' % fn)
        
        return self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)[0]
    

    def _predict_mean_and_scale(self, img, axes, normalizer, resizer, n_tiles=None):
        """Apply neural network to raw image to predict restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        """
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        # axes = axes_check_and_normalize(axes,img.ndim)

        # different kinds of axes
        # -> typical case: net_axes_in = net_axes_out, img_axes_in = img_axes_out
        img_axes_in = axes_check_and_normalize(axes, img.ndim)
        net_axes_in = self.config.axes
        net_axes_out = axes_check_and_normalize(self._axes_out)
        set(net_axes_out).issubset(set(net_axes_in)) or _raise(ValueError("different kinds of output than input axes"))
        net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
        img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)
        # print(' -> '.join((img_axes_in, net_axes_in, net_axes_out, img_axes_out)))
        tiling_axes = net_axes_out.replace('C','') # axes eligible for tiling

        _permute_axes = self._make_permute_axes(img_axes_in, net_axes_in, net_axes_out, img_axes_out)
        # _permute_axes: (img_axes_in -> net_axes_in), undo: (net_axes_out -> img_axes_out)
        x = _permute_axes(img)
        # x has net_axes_in semantics
        x_tiling_axis = tuple(axes_dict(net_axes_in)[a] for a in tiling_axes) # numerical axis ids for x

        channel_in = axes_dict(net_axes_in)['C']
        channel_out = axes_dict(net_axes_out)['C']
        net_axes_in_div_by = self._axes_div_by(net_axes_in)
        net_axes_in_overlaps = self._axes_tile_overlap(net_axes_in)
        self.config.n_channel_in == x.shape[channel_in] or _raise(ValueError())

        # TODO: refactor tiling stuff to make code more readable

        def _total_n_tiles(n_tiles):
            n_block_overlaps = [int(np.ceil(1.* tile_overlap / block_size)) for tile_overlap, block_size in zip(net_axes_in_overlaps, net_axes_in_div_by)]
            return total_n_tiles(x,n_tiles=n_tiles,block_sizes=net_axes_in_div_by,n_block_overlaps=n_block_overlaps,guarantee='size')

        _permute_axes_n_tiles = self._make_permute_axes(img_axes_in, net_axes_in)
        # _permute_axes_n_tiles: (img_axes_in <-> net_axes_in) to convert n_tiles between img and net axes
        def _permute_n_tiles(n,undo=False):
            # hack: move tiling axis around in the same way as the image was permuted by creating an array
            return _permute_axes_n_tiles(np.empty(n,np.bool),undo=undo).shape

        # to support old api: set scalar n_tiles value for the largest tiling axis
        if np.isscalar(n_tiles) and int(n_tiles)==n_tiles and 1<=n_tiles:
            largest_tiling_axis = [i for i in np.argsort(x.shape) if i in x_tiling_axis][-1]
            _n_tiles = [n_tiles if i==largest_tiling_axis else 1 for i in range(x.ndim)]
            n_tiles = _permute_n_tiles(_n_tiles,undo=True)
            warnings.warn("n_tiles should be a tuple with an entry for each image axis")
            print("Changing n_tiles to %s" % str(n_tiles))

        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)

        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))
        n_tiles = tuple(map(int,n_tiles))
        n_tiles = _permute_n_tiles(n_tiles)
        (all(n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
            _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))
        # n_tiles_limited = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
        # if any(np.array(n_tiles) != np.array(n_tiles_limited)):
        #     print("Limiting n_tiles to %s" % str(_permute_n_tiles(n_tiles_limited,undo=True)))
        # n_tiles = n_tiles_limited
        n_tiles = list(n_tiles)


        # normalize & resize
        x = normalizer.before(x, net_axes_in)
        x = resizer.before(x, net_axes_in, net_axes_in_div_by)

        done = False
        progress = Progress(_total_n_tiles(n_tiles),1)
        c = 0
        while not done:
            try:
                # raise tf.errors.ResourceExhaustedError(None,None,None) # tmp
                x = predict_tiled(self.keras_model,x,axes_in=net_axes_in,axes_out=net_axes_out,
                                  n_tiles=n_tiles,block_sizes=net_axes_in_div_by,tile_overlaps=net_axes_in_overlaps,pbar=progress)
                # x has net_axes_out semantics
                done = True
                progress.close()
            except tf.errors.ResourceExhaustedError:
                # TODO: how to test this code?
                # n_tiles_prev = list(n_tiles) # make a copy
                tile_sizes_approx = np.array(x.shape) / np.array(n_tiles)
                t = [i for i in np.argsort(tile_sizes_approx) if i in x_tiling_axis][-1]
                n_tiles[t] *= 2
                # n_tiles = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
                # if all(np.array(n_tiles) == np.array(n_tiles_prev)):
                    # raise MemoryError("Tile limit exceeded. Memory occupied by another process (notebook)?")
                if c >= 8:
                    raise MemoryError("Giving up increasing number of tiles. Memory occupied by another process (notebook)?")
                print('Out of memory, retrying with n_tiles = %s' % str(_permute_n_tiles(n_tiles,undo=True)))
                progress.total = _total_n_tiles(n_tiles)
                c += 1

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)
        x.shape[channel_out] == n_channel_predicted or _raise(ValueError())

        x = resizer.after(x, net_axes_out)

        mean, scale = self._mean_and_scale_from_prediction(x,axis=channel_out)
        # mean and scale have net_axes_out semantics

        if normalizer.do_after and self.config.n_channel_in==self.config.n_channel_out:
            mean, scale = normalizer.after(mean, scale, net_axes_out)

        mean, scale = _permute_axes(mean,undo=True), _permute_axes(scale,undo=True)
        # mean and scale have img_axes_out semantics

        return mean, scale


    def _mean_and_scale_from_prediction(self,x,axis=-1):
        # separate mean and scale
        if self.config.probabilistic:
            _n = self.config.n_channel_out
            assert x.shape[axis] == 2*_n
            slices = [slice(None) for _ in x.shape]
            slices[axis] = slice(None,_n)
            mean = x[tuple(slices)]
            slices[axis] = slice(_n,None)
            scale = x[tuple(slices)]
        else:
            mean, scale = x, None
        return mean, scale

    
    def _axes_div_by(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        # default: must be divisible by power of 2 to allow down/up-sampling steps in unet
        pool_div_by = 2**self.config.unet_n_depth
        return tuple((pool_div_by if a in 'XYZT' else 1) for a in query_axes)

    def _axes_tile_overlap(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        overlap = tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size)
        return tuple((overlap if a in 'XYZT' else 0) for a in query_axes)

    @property
    def _config_class(self):
        return Config
