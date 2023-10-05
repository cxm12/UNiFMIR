# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division

import warnings
from csbdeep.internals.probability import ProbabilisticPrediction
from .config import Config
from .base_model import BaseModel, suppress_without_basedir

from ..utils import _raise, axes_check_and_normalize, axes_dict
from ..utils.six import Path
from ..utils.tf import export_SavedModel, IS_TF_1, CARETensorBoardImage, tf, K, keras, Callback
from ..version import __version__ as package_version
from ..data import PercentileNormalizer, PadAndCropResizer
from ..internals.predict import predict_tiled, tile_overlap, Progress, total_n_tiles
from ..internals import nets, train
from csbdeep.utils import normalize
from csbdeep.func_mcx import np, savecolorim, savecolorim1, compute_psnr_and_ssim
import os
import cv2
# if IS_TF_1:
#     import tensorflow as tf
# else:
#     import tensorflow.compat.v1 as tf
#     # tf.disable_v2_behavior()
from tifffile import imread, imsave


class CosineAnnealing(Callback):
    """Cosine annealing according to DECOUPLED WEIGHT DECAY REGULARIZATION.
    余弦退火
    # Arguments
        eta_max: float, eta_max in eq(5).
        eta_min: float, eta_min in eq(5).
        total_iteration: int, Ti in eq(5).
        iteration: int, T_cur in eq(5).
        verbose: 0 or 1.
    """
    
    def __init__(self, eta_max=1, eta_min=0, total_iteration=0, iteration=0, verbose=0, **kwargs):
        super(CosineAnnealing, self).__init__()
        
        global lr_list
        
        lr_list = []
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.total_iteration = total_iteration
        self.iteration = iteration
    
    def on_train_begin(self, logs=None):
        self.lr = K.get_value(self.model.optimizer.lr)
    
    def on_train_end(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.lr)
    
    def on_batch_end(self, epoch, logs=None):
        self.iteration += 1
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * 0.5 * (
                    1 + np.cos(np.pi * self.iteration / self.total_iteration))
        new_lr = self.lr * eta_t
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealing '
                  'learning rate to %s.' % (epoch + 1, new_lr))
        lr_list.append(logs['lr'])


# https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/
class SnapshotEnsemble(Callback):
    
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


class CAREDropOut(BaseModel):
    """ CARE UNet + DropOut """

    def __init__(self, config, name=None, basedir='.', modeltype='', finetune=False, scale=1):
        """See class docstring."""
        self.modeltype = modeltype
        self.scale = scale
        super(CAREDropOut, self).__init__(config=config, name=name, basedir=basedir)

        if finetune:
            # name is None or (isinstance(name, string_types) and len(name) > 0) or _raise(
            #     ValueError("No valid name: '%s'" % str(name)))
            # basedir is None or isinstance(basedir, (string_types, Path)) or _raise(
            #     ValueError("No valid basedir: '%s'" % str(basedir)))
            # self.name = name
            # self.basedir = Path(basedir) if basedir is not None else None
            # self.logdir = self.basedir / self.name
            self._find_and_load_weights()

    def _build(self):
        if self.scale == 1:
            return nets.common_unetdropout(
                n_dim=self.config.n_dim,
                n_channel_out=self.config.n_channel_out,
                prob_out=self.config.probabilistic,
                residual=self.config.unet_residual,
                n_depth=self.config.unet_n_depth,
                kern_size=self.config.unet_kern_size,
                n_first=self.config.unet_n_first,
                last_activation=self.config.unet_last_activation,
                modeltype=self.modeltype
            )(self.config.unet_input_shape)
        else:
            print('Super-Resolution: scale = 2')
            return nets.common_unetdropout(
                n_dim=self.config.n_dim,
                n_channel_out=self.config.n_channel_out,
                prob_out=self.config.probabilistic,
                residual=self.config.unet_residual,
                n_depth=self.config.unet_n_depth,
                kern_size=self.config.unet_kern_size,
                n_first=self.config.unet_n_first,
                last_activation=self.config.unet_last_activation,
                modeltype=self.modeltype, scale=self.scale
            )(self.config.unet_input_shape)

    def prepare_for_training(self, optimizer=None, **kwargs):
        """Prepare for neural network training.

        Calls :func:`csbdeep.internals.train.prepare_model` and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.
        kwargs : dict
            Additional arguments for :func:`csbdeep.internals.train.prepare_model`.

        """
        if optimizer is None:
            # optimizer = tf.keras.optimizers.Adam(lr=self.config.train_learning_rate)
            optimizer = keras.optimizers.Adam(lr=self.config.train_learning_rate)
            # Adam = keras_import('optimizers', 'Adam')
            # optimizer = Adam(lr=self.config.train_learning_rate)
            
        # 定义loss + model.compile(optimizer=optimizer, loss='mae'):
        # return list [TerminateOnNaN(), ParameterDecayCallback]
        self.callbacks = train.prepare_model(self.keras_model, optimizer, self.config.train_loss, **kwargs)

        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                if IS_TF_1:
                    from ..utils.tf import CARETensorBoard
                    self.callbacks.append(CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=self.config.probabilistic))
                else:
                    from tensorflow.keras.callbacks import TensorBoard
                    self.callbacks.append(TensorBoard(log_dir=str(self.logdir/'logs'), write_graph=False, profile_batch=0))

        if self.config.train_reduce_lr is not None:
            # # 训练过程中缩小学习率：
            # ReduceLROnPlateau = keras_import('callbacks', 'ReduceLROnPlateau')
            # rlrop_params = self.config.train_reduce_lr
            # if 'verbose' not in rlrop_params:
            #     rlrop_params['verbose'] = True
            # # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            # self.callbacks.insert(0, ReduceLROnPlateau(**rlrop_params))

            # 余弦退火缩小学习率：
            # reduce_lr = CosineAnnealing(eta_max=1, eta_min=0, total_iteration=Ti * (2000 // 16), iteration=0, verbose=0)
            reduce_lr = SnapshotEnsemble(n_epochs=self.config.train_epochs, n_cycles=self.config.train_epochs/3, lrate_max=self.config.train_learning_rate)
            self.callbacks.insert(0, reduce_lr)

        self._model_prepared = True

    def train(self, X, Y, validation_data, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        Y : :class:`numpy.ndarray`
            Array of target images.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        ((isinstance(validation_data, (list, tuple)) and len(validation_data) == 2)
         or _raise(ValueError('validation_data must be a pair of numpy arrays')))
    
        n_train, n_val = len(X), len(validation_data[0])
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100 * frac_val))
        axes = axes_check_and_normalize('S' + self.config.axes, X.ndim)
        ax = axes_dict(axes)
    
        for a, div_by in zip(axes, self._axes_div_by(axes)):
            n = X.shape[ax[a]]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axis %s"
                    " (which has incompatible size %d)" % (div_by, a, n)
                )
    
        if epochs is None:
            epochs = self.config.train_epochs  # 100
        if steps_per_epoch is None:
            steps_per_epoch = n_train // self.config.train_batch_size
            # steps_per_epoch = self.config.train_steps_per_epoch  # 400
    
        if not self._model_prepared:
            self.prepare_for_training()
    
        if (self.config.train_tensorboard and self.basedir is not None and
                not IS_TF_1 and not any(isinstance(cb, CARETensorBoardImage) for cb in self.callbacks)):
            self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=validation_data,
                                                       log_dir=str(self.logdir / 'logs' / 'images'),
                                                       n_images=3, prob_out=self.config.probabilistic))
    
        training_data = train.DataWrapper(X, Y, self.config.train_batch_size, length=epochs * steps_per_epoch)
        
        if epochs != 0:
            fit = self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit
            history = fit(iter(training_data), validation_data=validation_data,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      callbacks=self.callbacks, verbose=1)
            self._training_finished()
            return history

    @suppress_without_basedir(warn=True)
    def export_TF(self, fname=None):
        """Export neural network via :func:`csbdeep.utils.tf.export_SavedModel`.

        Parameters
        ----------
        fname : str or None
            Path of the created SavedModel archive (will end with ".zip").
            If ``None``, "<model-directory>/TF_SavedModel.zip" will be used.

        """
        if fname is None:
            fname = self.logdir / 'TF_SavedModel_dropout.zip'
        else:
            fname = Path(fname)

        meta = {
            'type':          self.__class__.__name__,
            'version':       package_version,
            'probabilistic': self.config.probabilistic,
            'axes':          self.config.axes,
            'axes_div_by':   self._axes_div_by(self.config.axes),
            'tile_overlap':  self._axes_tile_overlap(self.config.axes),
        }
        export_SavedModel(self.keras_model, str(fname), meta=meta)
        print("\nModel exported in TensorFlow's SavedModel format:\n%s" % str(fname.resolve()))

    def predict(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=None):
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
        # self._find_and_load_weights()
        # print('Load model for Predict from up ^^')
        return self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)[0]

    def predict_probabilistic(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=None):
        """Apply neural network to raw image to predict probability distribution for restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        :class:`csbdeep.internals.probability.ProbabilisticPrediction`
            Returns the probability distribution of the restored image.

        Raises
        ------
        ValueError
            If this is not a probabilistic model.

        """
        self.config.probabilistic or _raise(ValueError('This is not a probabilistic model.'))
        mean, scale = self._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)
        return ProbabilisticPrediction(mean, scale)

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
        # axes = axes_check_and_normalize(axes, img.ndim)

        # different kinds of axes
        # -> typical case: net_axes_in = net_axes_out, img_axes_in = img_axes_out
        img_axes_in = axes_check_and_normalize(axes, img.ndim)
        net_axes_in = self.config.axes
        net_axes_out = axes_check_and_normalize(self._axes_out)
        set(net_axes_out).issubset(set(net_axes_in)) or _raise(ValueError("different kinds of output than input axes"))
        net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
        img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)
        # print(' -> '.join((img_axes_in, net_axes_in, net_axes_out, img_axes_out)))
        tiling_axes = net_axes_out.replace('C', '')  # axes eligible for tiling

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

        all(np.isscalar(t) and 1 <= t and int(t)==t for t in n_tiles) or _raise(
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

    def _mean_and_scale_from_prediction(self, x, axis=-1):
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

    def Valid_on_testsetdenoise3D(self, level='condition_1'):
        psnrlst = []
        ssimlst = []
        mselst = []
        C123 = []
        patchsize = 600
        meanvar = 0
        meandf = 0
        PCCLst = []
        meanvarlst = []
        print('*******************!!! Start VAlidation on Testset !!!*************************')
        testnum = 10  # 2  #
        tile = (1, 4, 4)
    
        testset = 'Denoising_Planaria'
        testGTpath = '../../DataSet/' + testset + '/test_data/GT/'
        savepathzyx = './models/epoch200/my_model%s/%s/result/%s/AllT%d/MeanZYX/' % (self.modeltype, testset, level, testnum)
        os.makedirs(savepathzyx, exist_ok=True)
        testdatapath = '../../DataSet/' + testset + '/test_data/' + level + '/'

        meanpsnr = 0
        meanssim = 0
        filename = os.listdir(testdatapath)
        imnum = 1  # len(filename)  #
        for i in range(imnum):
            name = 'EXP280_Smed_live_RedDot1_slide_mnt_N3_stk1'  # filename[i][:-4]  # 'EXP278_Smed_fixed_RedDot1_sub_5_N7_m0007'  # namelst[i]
            y = imread(testGTpath + name + '.tif')  #
            x = imread(testdatapath + name + '.tif')
            ###*!!!!!!!! 千万不要裁剪边缘！！！！！ [:, 200:-200, 200:-200]  # # ynorm = normalize(y, 0.1, 99.9, clip=True) * 255  #
            if testset == 'Denoising_Planaria':
                [z, h, w] = x.shape
                axes = 'ZYX'
            if 'dropout' in self.modeltype:  # 将模式转为训练模式，在测试时才能使用dropout!!!
                keras.backend.set_learning_phase(1)
            resultlst = []
            resultlstv = []
            for ti in range(testnum):
                restored = self.predict(x, axes, n_tiles=tile)
                resultlst.append(restored)
                restorednorm = normalize(restored, 0.1, 99.9, clip=True)
                resultlstv.append(restorednorm)
                # print(f'x = {x.mean()}, restored = {restored.mean()}, y = {y.mean()}, restorednorm = {restorednorm.mean()}')
            mean = np.mean(np.array(resultlst), axis=0)
            imsave(savepathzyx + name + '-Mean_ZYX.tif', mean)

            psm, ssmm = compute_psnr_and_ssim(normalize(mean, 0.1, 99.9, clip=True) * 255,
                                              normalize(y, 0.1, 99.9, clip=True) * 255)
            rmse = np.mean(np.square(mean - y), dtype=np.float64)
            meanpsnr += psm
            meanssim += ssmm
            psnrlst.append(psm)
            ssimlst.append(ssmm)
            mselst.append(rmse)
    
            Var0 = np.var(np.array(resultlstv), axis=0)
            pcc0 = self.pearson_distance(Var0.ravel(), mean.ravel())
            imsave(savepathzyx + name + '-noNormVar.tif', Var0)
            v = np.mean(Var0)
            vmax = np.max(Var0)
            meanvarlst.append(v)
            print(f'Image - {name}- PSNR/SSIM{psm, ssmm} /PCC/MSE {pcc0, rmse} STD/STDMax{v, vmax}')
            # exit()
    
            y = y[:, 200:-200, 200:-200]
            for randc in range(80, z - 5):
                randh = 0
                randw = 0
                patchdflst = []
                patchlst = []
                for im in resultlst:
                    im = im[:, 200:-200, 200:-200]
                    patchlst.append(im[randc, randh:randh + patchsize, randw:randw + patchsize])
                    arrdf = np.abs(
                        normalize(im, 0.1, 99.9, clip=True)[randc, randh:randh + patchsize, randw:randw + patchsize]
                        - normalize(y, 0.1, 99.9, clip=True)[randc, randh:randh + patchsize, randw:randw + patchsize])
                    patchdflst.append(arrdf)
        
                Var = np.var(np.array(patchdflst), axis=0)
                Meandf = np.mean(np.array(patchdflst), axis=0)  # mean of difference
                pcc = self.pearson_distance(Var.ravel(), Meandf.ravel())
                PCCLst.append(pcc)
                Mean = np.mean(np.array(patchlst), axis=0)
                NormVar = normalize(Var, 0.1, 99.9, clip=True)
        
                # imsave(savepathzyx + name + '-Meandf_z%d.png' % randc, np.expand_dims(Meandf[200:-200, 200:-200], -1))
                # savecolorim(savepathzyx + name + '-MeandfC_z%d.png' % randc, Meandf)
                # imsave(savepathzyx + name + '-Meandf_z%d.tif' % randc, np.expand_dims(Meandf[200:-200, 200:-200], -1))
                savecolorim(savepathzyx + name + '-MeandfnoNormC_z%d.png' % randc,
                            np.clip(Meandf[200:-200, 200:-200] * 255, 0, 255), norm=False)
                savecolorim(savepathzyx + name + '-MeanC_z%d.png' % randc, Mean[200:-200, 200:-200])
                savecolorim1(savepathzyx + name + '-NormVarC_z%d.png' % randc, NormVar[200:-200, 200:-200])
        
                # meanvar += np.mean(Var)
                # meandf += np.mean(Meandf)
                del patchdflst, patchlst
            del resultlst
        # 将数组写入文件
        file = open(savepathzyx + "Psnrssimpccmse20.txt", 'w')
        file.write(
            'PSNR \n' + str(psnrlst) + '\n SSIM \n' + str(ssimlst) + '\n MSE \n' + str(mselst) + '\n PCC \n' + str(
                PCCLst) + '\n STD \n' + str(meanvarlst))
        file.close()
        print('meanssim, meanpsnr', meanssim / imnum, meanpsnr / imnum, np.mean(np.array(PCCLst)),
              np.mean(np.array(mselst)), np.mean(np.array(meanvarlst)))
        C123.append(
            testset + self.modeltype + f'PSNR{meanpsnr / imnum}, SSIM{meanssim / imnum}, PCC{np.mean(np.array(PCCLst))},'
                                       f'MSE{np.mean(np.array(mselst))}, Vae{np.mean(np.array(meanvarlst))}')
        print('******************* !!! End VAlidation on Testset !!! *************************')
        return meanpsnr / imnum, meanssim / imnum, psnrlst, ssimlst, np.mean(np.array(PCCLst)),\
               np.mean(np.array(mselst)), np.mean(np.array(meanvarlst)), C123

    def Valid_on_testsetSR(self, validation_data, testset):
        self.testset = testset
        xlst = validation_data[0]
        ylst = validation_data[1]
        print('*******************!!! Start VAlidation on Testset !!!*************************')
        testnum = 10  #
        imnum = 100
        Mdfmaplst = []
        Varmaplst = []
        PCCLst = []
        psnrlst = []
        ssimlst = []
        mselst = []
        meanvarlst = []
        meanpsnr = 0
        meanssim = 0
        meanvar = 0
        meandf = 0
        meanKLdiv = 0
        savepath = './models/epoch200/my_model%s/%s/result/AllT%d/' % (self.modeltype, self.testset, testnum)
        os.makedirs(savepath, exist_ok=True)
        for i in range(imnum):  # 4, 5):  # 6,  5
            y = ylst[i]
            x = xlst[i]
            axes = 'YXC'
            name = 'im%d_LR' % (i + 1)
        
            # ## Apply CARE network to raw image
            resultlst = []
            resultlstv = []
            for t in range(testnum):
                restored = self.predict(x, axes)
                resultlst.append(restored)
                restorednorm = normalize(restored, 0.1, 99.9, clip=True)
                resultlstv.append(restorednorm)
                # print(f'{t}: {x.mean()}, {restored.mean()}, {y.mean()}')
                
            Mean = np.squeeze(np.mean(np.array(resultlst), 0))
            y = np.squeeze(y)
            Meandf = np.abs(Mean - y)
            Var = np.var(np.array(resultlstv), axis=0)  # * 255 * 100 Compute the variance along the specified axis.
            NormVar = np.squeeze(normalize(Var, 0, 100))
           
            # ## 保存结果
            # cv2.imwrite(savepath + name + '-Meandf.png', np.expand_dims(Meandf, -1))
            # savecolorim(savepath + name + '-MeandfC.png', Meandf)
            savecolorim(savepath + name + '-MeandfnoNormC.png', np.clip(Meandf * 255, 0, 255), norm=False)
            savecolorim1(savepath + name + '-NormVarC.png', NormVar)
            savecolorim(savepath + name + '-MeanC.png', Mean)
            psm, ssmm = compute_psnr_and_ssim(normalize(Mean, 0.1, 99.9, clip=True) * 255, normalize(y, 0.1, 99.9, clip=True) * 255)
            pcc = self.pearson_distance(Var.ravel(), Meandf.ravel())
            PCCLst.append(pcc)
            rmsep = np.mean(np.square(Mean - y), dtype=np.float64)
            v = np.mean(Var)
            vmax = np.max(Var)
            meanvarlst.append(v)
            print(f'Image - {name}- PSNR/SSIM{psm, ssmm} /PCC/MSE {pcc, rmsep} STD/STDMax{v, vmax}')
            Mdfmaplst.append(Meandf)
            Varmaplst.append(Var)
            mselst.append(rmsep)
            meanpsnr += psm
            meanssim += ssmm
            psnrlst.append(psm)
            ssimlst.append(ssmm)
            meanvar += np.mean(Var)
            meandf += np.mean(Meandf)
            meanKLdiv += 0
            del resultlst, resultlstv
        file = open(savepath + "Psnrssimpccmse100.txt", 'w')
        file.write('\n PSNR \n' + str(psnrlst) + '\n SSIM \n' + str(ssimlst) + '\n MSE \n' + str(
            mselst) + '\n PCC \n' + str(PCCLst) + '\n STD \n' + str(meanvarlst))
        file.close()
        print('%d image, Mean Var/ Mean DF of Testset %s is' % (imnum, self.testset), meanvar / imnum, meandf / imnum)
        print(np.mean(np.array(psnrlst)), np.mean(np.array(ssimlst)), np.mean(np.array(mselst)),
              np.mean(np.array(PCCLst)))
        print(psnrlst, '\n', ssimlst)
        print('******************* !!! End VAlidation on Testset !!! *************************')
        return meanpsnr / imnum, meanssim / imnum

    # def _limit_tiling(self,img_shape,n_tiles,block_sizes):
    #     img_shape, n_tiles, block_sizes = np.array(img_shape), np.array(n_tiles), np.array(block_sizes)
    #     n_tiles_limit = np.ceil(img_shape / block_sizes) # each tile must be at least one block in size
    #     return [int(t) for t in np.minimum(n_tiles,n_tiles_limit)]

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
