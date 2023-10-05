import random
import imageio
import glob
import torch.utils.data as data
from PIL import Image
import torch
import sys
import os
import numpy as np
from tifffile import imread, imsave
from scipy.ndimage.interpolation import zoom
sys.path.append('..')
from csbdeep.utils import normalize, axes_dict, axes_check_and_normalize, backend_channels_last, move_channel_for_backend


datamin, datamax = 0, 100  #



def load_training_data(file, validation_split=0, axes=None, n_images=None, verbose=False):
    """Load training data from file in ``.npz`` format.

    The data file is expected to have the keys:

    - ``X``    : Array of training input images.
    - ``Y``    : Array of corresponding target images.
    - ``axes`` : Axes of the training images.


    Parameters
    ----------
    file : str
        File name
    validation_split : float
        Fraction of images to use as validation set during training.
    axes: str, optional
        Must be provided in case the loaded data does not contain ``axes`` information.
    n_images : int, optional
        Can be used to limit the number of images loaded from data.
    verbose : bool, optional
        Can be used to display information about the loaded images.

    Returns
    -------
    tuple( tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), str )
        Returns two tuples (`X_train`, `Y_train`), (`X_val`, `Y_val`) of training and validation sets
        and the axes of the input images.
        The tuple of validation data will be ``None`` if ``validation_split = 0``.

    """
    
    f = np.load(file)
    X, Y = f['X'], f['Y']
    print(Y.ndim, Y.shape)
    if axes is None:
        axes = f['axes']
    axes = axes_check_and_normalize(axes)
    
    assert X.ndim == Y.ndim
    assert len(axes) == X.ndim
    assert 'C' in axes
    if n_images is None:
        n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
    assert 0 <= validation_split < 1
    
    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']
    
    if validation_split > 0:
        n_val = int(round(n_images * validation_split))
        n_train = n_images - n_val
        assert 0 < n_val and 0 < n_train
        X_t, Y_t = X[-n_val:], Y[-n_val:]
        X, Y = X[:n_train], Y[:n_train]
        assert X.shape[0] == n_train and X_t.shape[0] == n_val
        X_t = move_channel_for_backend(X_t, channel=channel)
        Y_t = move_channel_for_backend(Y_t, channel=channel)
    
    X = move_channel_for_backend(X, channel=channel)
    Y = move_channel_for_backend(Y, channel=channel)
    
    axes = axes.replace('C', '')  # remove channel
    if backend_channels_last():
        axes = axes + 'C'
    else:
        axes = axes[:1] + 'C' + axes[1:]
    
    data_val = (X_t, Y_t) if validation_split > 0 else None
    
    if verbose:
        ax = axes_dict(axes)
        n_train, n_val = len(X), len(X_t) if validation_split > 0 else 0
        image_size = tuple(X.shape[ax[a]] for a in axes if a in 'TZYX')
        n_dim = len(image_size)
        n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]
        
        print('number of training images:\t', n_train)
        print('number of validation images:\t', n_val)
        print('image size (%dD):\t\t' % n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in, '/', n_channel_out)
    
    return (X, Y), data_val, axes


def np2Tensor(*args):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return tensor
    
    return [_np2Tensor(a) for a in args]


def loadData(traindatapath, axes='SCYX', validation_split=0.05):
    print('Load data npz')
    if validation_split > 0:
        (X, Y), (X_val, Y_val), axes = load_training_data(traindatapath, validation_split=validation_split, axes=axes, verbose=True)
    else:
        (X, Y), _, axes = load_training_data(traindatapath, validation_split=validation_split, axes=axes, verbose=True)
        X_val, Y_val = 0, 1
    print(X.shape, Y.shape)
    return X, Y, X_val, Y_val


class SR(data.Dataset):
    def __init__(self, percent=1.0, scale=2, name='CCPs', train=True, benchmark=False, patch_size=64, test_only=False,
                 rootdatapath='', length=-1):
        self.length = length
        self.patch_size = patch_size
        self.rgb_range = 1
        self.name = name
        self.train = train
        self.test_only = test_only
        self.benchmark = benchmark
        self.dir_data = rootdatapath+'train/%s/my_training_data.npz' % name
        self.dir_demo = rootdatapath+'test/%s/LR/' % name
        
        self.input_large = (self.dir_demo != '')
        self.scale = scale
        if train:
            X, Y, X_val, Y_val = self.loadData()
            print('np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
            
            list_hr, list_lr = Y, X
            list_hr = list_hr[:int(percent * len(list_hr))]
            list_lr = list_lr[:int(percent * len(list_lr))]
        else:
            self.filenames = glob.glob(self.dir_demo + '*.tif')
            list_hr, list_lr, name = self._scan()            
            self.name = name
            if not self.test_only:
                list_hr, list_lr, name = list_hr[:5], list_lr[:5], name[:5]

        self.images_hr, self.images_lr = list_hr, list_lr

    def loadData(self):
        patch_size = self.patch_size
        X, Y, X_val, Y_val = loadData(self.dir_data)

        N, height, width, c = X.shape
        X1 = []
        Y1 = []
        for n in range(len(X)):
            for i in range(0, width, patch_size):
                for j in range(0, height, patch_size):
                    if j + patch_size >= height and i + patch_size >= width:
                        X1.append(X[n][height-patch_size:, width - patch_size:, :])
                        Y1.append(Y[n][height * 2 - patch_size * 2:, width * 2 - patch_size * 2:, :])
                    elif j + patch_size >= height:
                        X1.append(X[n][height - patch_size:, i:i + patch_size, :])
                        Y1.append(Y[n][height * 2 - patch_size * 2:, i * 2:i * 2 + patch_size * 2, :])
                    elif i + patch_size >= width:
                        X1.append(X[n][j:j + patch_size, width - patch_size:, :])
                        Y1.append(Y[n][j * 2:j * 2 + patch_size * 2:, width * 2 - patch_size * 2:, :])
                    else:
                        X1.append(X[n][j:j + patch_size, i:i + patch_size, :])
                        Y1.append(Y[n][j * 2:j * 2 + patch_size * 2, i * 2:i * 2 + patch_size * 2, :])
            
        return X1, Y1, X_val, Y_val
    
    def _scan(self):
        list_hr, list_lr, nm = [], [], []
        for fi in self.filenames:
            hr = np.array(Image.open(fi.replace('LR', 'GT')))
            lr = np.array(Image.open(fi))
            nm.append(fi[len(self.dir_demo):])
            list_hr.append(np.expand_dims(hr, -1))
            list_lr.append(np.expand_dims(lr, -1))
        return list_hr, list_lr, nm

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        if self.train:
            lr, hr, filename = self.images_lr[idx], self.images_hr[idx], ''
        else:
            lr, hr, filename = self.images_lr[idx], self.images_hr[idx], self.name[idx]
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.rgb_range
        pair = (lr, hr)
        pair_t = np2Tensor(*pair)
        
        return pair_t[0], pair_t[1], filename

    def __len__(self):
        print('len(self.images_hr)', len(self.images_hr))
        if self.train:
            if self.length < 0:
                return len(self.images_hr)
            else:
                return self.length
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            if self.length < 0:
                return idx % len(self.images_hr)
            else:
                return idx % self.length + self.length * random.randint(0, len(self.images_hr) // self.length - 1)
        else:
            return idx


class Flourescenedenoise(data.Dataset):
    def __init__(self, name='Denoising_Planaria', istrain=True, c=1, patch_size=64, test_only=False,
                 rootdatapath='', length=-1):

        self.length = length
        self.patch_size = patch_size
        self.rgb_range = 1
        self.datamin, self.datamax = 0, 100
        self.istrain = istrain
        self.test_only = test_only

        self.denoisegt = name
        self.datapath = rootdatapath + '%s/' % self.denoisegt

        if istrain:
            self._scandenoisenpy()
        else:
            self._scandenoisetif(c)
            if self.test_only:
                self.nm_lrdenoise = self.nm_lrdenoise[:5]
        
        self.lenth = len(self.nm_lrdenoise)
        
        if istrain:
            print('++ ++ ++ ++ ++ ++ self.length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ self.length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
    
    def _scandenoisenpy(self):
        hr = []
        lr = []
        patch_size = self.patch_size
        
        if 'Denoising' in self.denoisegt:
            X, Y, X_val, Y_val = loadData(self.datapath + 'train_data/data_label.npz', axes='SCZYX')
        else:
            X, Y, X_val, Y_val = loadData(self.datapath + 'train_data/data_label.npz')

        print('Dataset:', self.denoisegt, 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
        print('X.shape, Y.shape, X_val.shape, Y_val.shape = ', X.shape, Y.shape, X_val.shape, Y_val.shape)
        height, width = X.shape[-3:-1]
        X = np.reshape(X, [-1, height, width, 1])
        Y = np.reshape(Y, [-1, height, width, 1])
        assert len(X) == len(Y)
        if not 'Denoising' in self.denoisegt:
            X1 = []
            Y1 = []
            for n in range(len(X)):
                for i in range(0, width, patch_size):
                    for j in range(0, height, patch_size):
                        X1.append(X[n][j:j + patch_size, i:i + patch_size, :])
                        Y1.append(Y[n][j:j + patch_size, i:i + patch_size, :])
            hr.extend(Y1)
            lr.extend(X1)
        else:
            hr.extend(Y)
            lr.extend(X)
            
        self.nm_hrdenoise, self.nm_lrdenoise = hr, lr
        assert len(hr) == len(lr)
    
    def _scandenoisetif(self, c=1):
        lr = []
        if ('Planaria' in self.denoisegt) or ('Tribolium' in self.denoisegt):
            lr.extend(sorted(glob.glob(self.datapath + 'test_data/condition_%d/*.tif' % c)))
            self.hrpath = self.datapath + 'test_data/GT/'
            
        lr.sort()
        self.nm_lrdenoise = lr
    
    def __getitem__(self, idx):
        idx = self._get_index(idx)
        if self.istrain:
            lr, hr, filename = self._load_file_denoise_npy(idx + 5//2)
        else:
            lr, hr, filename, d = self._load_file_denoise(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.rgb_range)).float()
        return lr, hr, filename
    
    def __len__(self):
        if self.istrain:
            if self.length < 0:
                return self.lenth - 2 * (5//2)
            else:
                return self.length
        else:
            return self.lenth

    def _get_index(self, idx):
        if self.istrain:
            if self.length < 0:
                return idx % self.lenth
            else:
                return idx % self.length + self.length * random.randint(0, (self.lenth - 2 * (5//2)) // self.length - 1)
        else:
            return idx % self.lenth
            
    def _load_file_denoise(self, idn):
        filename, fmt = os.path.splitext(os.path.basename(self.nm_lrdenoise[idn]))
        rgb = np.float32(imread(self.hrpath + filename + fmt))
        rgblr = np.float32(imread(self.nm_lrdenoise[idn]))

        return rgblr, rgb, filename, self.denoisegt

    def _load_file_denoise_npy(self, idx):
        lr = []
        hr = []
        idn = (idx) % self.lenth
        hr.extend(self.nm_hrdenoise[idn:idn + 1])
        lr.extend(self.nm_lrdenoise[idn - 5 // 2:idn + 5 // 2 + 1])
        rgb = np.concatenate(hr, -1)
        rgblr = np.squeeze(np.concatenate(lr, -1))
        rgb = np.transpose(np.float32(rgb), (2, 0, 1))
        rgblr = np.transpose(np.float32(rgblr), (2, 0, 1))
        
        return rgblr, rgb, ''


class Flouresceneiso(data.Dataset):
    def __init__(self, name='Isotropic_Liver', istrain=True, patch_size=64, test_only=False,
                rootdatapath='', length=-1):
        self.length = length
        self.patch_size = patch_size
        self.rgb_range = 1
        self.datamin, self.datamax = 0, 100
        self.istrain = istrain
        self.test_only = test_only

        self.iso = name  #
        self.datapath = rootdatapath + '%s/train_data/' % self.iso
        self.dir_lr = rootdatapath + '%s/test_data/' % self.iso
        if istrain:
            self._scanisonpy()
            self.lenth = len(self.nm_lriso)
        else:
            self._scaniso()
            self.lenth = len(self.nm_lr)
                    
        if istrain:
            print('++ ++ ++ ++ ++ ++ self.length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ self.length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        
    def _scanisonpy(self):
        hr = []
        lr = []
        patch_size = self.patch_size
        
        X, Y, _, _ = loadData(self.datapath + 'data_label.npz', axes='SCYX', validation_split=0.0)
        
        print('Dataset:', self.iso, 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
        print('X.shape, Y.shape = ', X.shape, Y.shape)
        height, width = X.shape[1:3]
        assert len(X) == len(Y)

        if patch_size < height:
            X1 = []
            Y1 = []
            for n in range(len(X)):
                for i in range(0, width, patch_size):
                    for j in range(0, height, patch_size):
                        X1.append(X[n][j:j + patch_size, i:i + patch_size, :])
                        Y1.append(Y[n][j:j + patch_size, i:i + patch_size, :])
            hr.extend(Y1)
            lr.extend(X1)
        else:
            hr.extend(Y)
            lr.extend(X)

        self.nm_hriso, self.nm_lriso = hr, lr

    def _scaniso(self):
        hr = []
        lr = []
        if self.iso == 'Isotropic_Liver':
            hr.append(self.dir_lr + 'input_subsample_1_groundtruth.tif')
            lr.append(self.dir_lr + 'input_subsample_8.tif')
        else:
            filenames = os.listdir(self.dir_lr)
            for fi in range(len(filenames)):
                name = filenames[fi][:-4]
                lr.append(self.dir_lr + name + '.tif')
            
        self.nm_hr, self.nm_lr = hr, lr
        
    def __getitem__(self, idx):
        idx = self._get_index(idx)
        if self.istrain:
            lr, hr, filename = self._load_file_iso_npy(idx)
        else:
            lr, hr, filename = self._load_file_isotest(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.rgb_range)).float()
        
        return lr, hr, filename
    
    def __len__(self):
        if self.istrain:
            if self.length < 0:
                return self.lenth
            else:
                return self.length
        else:
            return self.lenth

    def _get_index(self, idx):
        if self.istrain:
            if self.length < 0:
                return idx % self.lenth
            else:
                return idx % self.length + self.length * random.randint(0, self.lenth // self.length - 1)
        else:
            return idx % self.lenth
            
    def _load_file_iso_npy(self, idx):
        lr = []
        hr = []
        idn = (idx) % self.lenth
        hr.append(self.nm_hriso[idn])
        lr.append(self.nm_lriso[idn])

        rgb = np.concatenate(hr, -1)
        rgblr = np.concatenate(lr, -1)
        rgb = np.transpose(np.float32(rgb), (2, 0, 1))
        rgblr = np.transpose(np.float32(rgblr), (2, 0, 1))
    
        return rgblr, rgb, ''

    def _load_file_isotest(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))
        
        rgblr = np.float32(imread(self.nm_lr[idx]))
        
        if 'Isotropic_Liver' in self.nm_lr[idx]:
            hrp = self.nm_lr[idx].replace('_8.tif', '_1_groundtruth.tif')
            rgb = np.float32(imread(hrp))
            
            return rgblr, rgb, filename
        elif 'Retina' in self.nm_lr[idx]:
            rgblr = np.transpose(zoom(rgblr, (10.2, 1, 1, 1), order=1), [0, 2, 3, 1])
        
        return rgblr, rgblr, filename


class Flouresceneproj(data.Dataset):
    def __init__(self, patch_size=64, name='Projection_Flywing', istrain=True, condition=0, test_only=False,
                 rootdatapath='', length=-1):
        self.length = length
        self.patch_size = patch_size
        self.rgb_range = 1
        self.istrain = istrain
        self.test_only = test_only
        self.iso = [name]
        self.rootdatapath = rootdatapath + '%s/' % self.iso[0]
        if istrain:
            self._scannpy()
        else:
            self._scan(condition)
            if self.test_only:
                self.nm_lr = self.nm_lr[:5]
                
        self.lenth = len(self.nm_lr)

        if istrain:
            print('++ ++ ++ ++ ++ ++ self.length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ self.length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')

    def load_training_data(self, file, axes=None, n_images=None, verbose=False):
        print('Begin np.load(file)', file)
        f = np.load(file)
        X, Y = f['X'], f['Y']
        Y = np.expand_dims(Y, 2)
        print(Y.ndim, Y.shape)

        if axes is None:
            axes = f['axes']  #
        axes = axes_check_and_normalize(axes)
    
        assert X.ndim == Y.ndim
        assert len(axes) == X.ndim
        assert 'C' in axes
        if n_images is None:
            n_images = X.shape[0]
        assert X.shape[0] == Y.shape[0]
        assert 0 < n_images <= X.shape[0]
    
        X, Y = X[:n_images], Y[:n_images]
        channel = axes_dict(axes)['C']
        
        X = move_channel_for_backend(X, channel=channel)
        Y = move_channel_for_backend(Y, channel=channel)
    
        axes = axes.replace('C', '')  # remove channel
        if backend_channels_last():
            axes = axes + 'C'
        else:
            axes = axes[:1] + 'C' + axes[1:]
    
        if verbose:
            ax = axes_dict(axes)
            n_train, n_val = len(X), 0
            image_size = tuple(X.shape[ax[a]] for a in axes if a in 'TZYX')
            n_dim = len(image_size)
            n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]
        
            print('number of training images:\t', n_train)
            print('number of validation images:\t', n_val)
            print('image size (%dD):\t\t' % n_dim, image_size)
            print('axes:\t\t\t\t', axes)
            print('channels in / out:\t\t', n_channel_in, '/', n_channel_out)
    
        return X, Y

    def _scannpy(self):
        patch_size = self.patch_size
        
        mytraindata = 1
        datapath = self.rootdatapath + 'train_data/my_training_data.npz'
        datapath2 = self.rootdatapath + 'train_data/data_label.npz'

        if mytraindata == 1:
            X, Y, _, _ = loadData(datapath, axes=None, validation_split=0.0)
        elif mytraindata == 2:
            X1, Y1, _, _ = loadData(datapath, axes=None, validation_split=0.0)
            X1l = []
            Y1l = []
            for n in range(len(X1)):
                for i in range(0, 128, 64):
                    for j in range(0, 128, 64):
                        X1l.append(X1[n][:, j:j + 64, i:i + 64, :])
                        Y1l.append(Y1[n][:, j:j + 64, i:i + 64, :])
            X1 = np.array(X1l)
            Y1 = np.array(Y1l)

            X2, Y2 = self.load_training_data(datapath2, axes='SCZYX', verbose=True)
            X = np.concatenate([X1, X2], 0)
            Y = np.concatenate([Y1, Y2], 0)
        else:
            X, Y = self.load_training_data(datapath2, axes='SCZYX', verbose=True)

        print('Dataset:', self.iso[0], 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
        print('X.shape, Y.shape = ', X.shape, Y.shape)
        height, width = X.shape[2:4]
        assert len(X) == len(Y)

        if patch_size < height:
            X1 = []
            Y1 = []
            for n in range(len(X)):
                for i in range(0, width, patch_size):
                    for j in range(0, height, patch_size):
                        X1.append(X[n][:, j:j + patch_size, i:i + patch_size, :])
                        Y1.append(Y[n][:, j:j + patch_size, i:i + patch_size, :])
        else:
            Y1 = Y
            X1 = X
        self.nm_hr, self.nm_lr = Y1, X1
    
    def _scan(self, condition):
        hr = []
        lr = []
        self.dir_lr = self.rootdatapath + 'test_data/'

        lr.extend(glob.glob(self.dir_lr + 'Input/C%d/*.tif' % condition))
        hr.extend(glob.glob(self.dir_lr + 'GT/C%d/*.tif' % condition))
        
        hr.sort()
        lr.sort()

        self.nm_hr, self.nm_lr = hr, lr
    
    def __getitem__(self, idx):
        idx = self._get_index(idx)
        
        if self.istrain:
            lr, hr, filename = self._load_file_npy(idx)
        else:
            lr, hr, filename = self._load_file_test(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.rgb_range)).float()
        
        return lr, hr, filename
    
    def __len__(self):
        if self.istrain:
            if self.length < 0:
                return self.lenth
            else:
                return self.length
        else:
            return self.lenth

    def _get_index(self, idx):
        if self.istrain:
            if self.length < 0:
                return idx % self.lenth
            else:
                return idx % self.length + self.length * random.randint(0, self.lenth // self.length - 1)
        else:
            return idx % self.lenth
        
    def _load_file_npy(self, idn):
        hr = self.nm_hr[idn]
        lr = self.nm_lr[idn]
        
        rgb = np.float32(np.squeeze(hr, -1))
        rgblr = np.float32(np.squeeze(lr))
        
        return rgblr, rgb, ''
    
    def _load_file_test(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))
        
        rgblr = np.float32(imread(self.nm_lr[idx]))
        rgb = np.expand_dims(np.float32(imread(self.nm_hr[idx])), 0)
        
        return rgblr, rgb, filename


class FlouresceneVCD:
    def __init__(self, patch_size=64, istrain=True, test_only=False, subtestset='to_predict',
                 rootdatapath='', length=-1):
        self.length = length
        self.path = rootdatapath
        self.istrain = istrain
        self.test_only = test_only
        self.patch_size = patch_size
        self.rgb_range = 1
        self.lf2d_base_size = patch_size // 11
        self.n_slices = 61
        self.n_num = 11
        self.shuffle = True
        if test_only:
            self.nm_lr2d = sorted(glob.glob(self.path + '%s/*.tif' % subtestset))
            self.nm_hr3d = sorted(glob.glob(self.path + 'results/VCD_tubulin/*.tif'))
        else:
            self.nm_hr3d = sorted(glob.glob(self.path + 'vcd-example-data/data/train/WF/*.tif'))
            self.nm_lr2d = sorted(glob.glob(self.path + 'vcd-example-data/data/train/LF/*.tif'))
            
        assert len(self.nm_hr3d) == len(self.nm_lr2d)

        self.lenth = len(self.nm_lr2d)

        if istrain:
            print('++ ++ ++ ++ ++ ++ self.length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ self.length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')

    def _load_dataset(self, idx):
        def rearrange3d_fn(image):
            """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
            """
        
            image = np.squeeze(image)  # remove channels dimension
            depth, height, width = image.shape
            image_re = np.zeros([height, width, depth])
            for d in range(depth):
                image_re[:, :, d] = image[d, :, :]
            return image_re
    
        def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
            """
            Extract different views from a single LF projection

            Params:
                -lf2d: numpy.array, 2-D light field projection in shape of [height, width, channels=1]
                -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                        'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
                -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                             False : shrink size of extracted views to (lf2d.shape / Nnum);
            Returns:
                ndarray [height, width, channels=n_num^2] if mode is 'toChannel'
                        or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
            """
            n = n_num
            h, w, c = lf2d.shape
            if padding:
                if mode == 'toDepth':
                    lf_extra = np.zeros([n * n, h, w, c])  # [depth, h, w, c]
                
                    d = 0
                    for i in range(n):
                        for j in range(n):
                            lf_extra[d, i: h: n, j: w: n, :] = lf2d[i: h: n, j: w: n, :]
                            d += 1
                elif mode == 'toChannel':
                    lf2d = np.squeeze(lf2d)
                    lf_extra = np.zeros([h, w, n * n])
                    
                    d = 0
                    for i in range(n):
                        for j in range(n):
                            lf_extra[i: h: n, j: w: n, d] = lf2d[i: h: n, j: w: n]
                            d += 1
                else:
                    raise Exception('unknown mode : %s' % mode)
            else:
                new_h = int(np.ceil(h / n))
                new_w = int(np.ceil(w / n))
            
                if mode == 'toChannel':
                    lf2d = np.squeeze(lf2d)
                    lf_extra = np.zeros([new_h, new_w, n * n])
                
                    d = 0
                    for i in range(n):
                        for j in range(n):
                            lf_extra[:, :, d] = lf2d[i: h: n, j: w: n]
                            d += 1
                elif mode == 'toDepth':
                    lf_extra = np.zeros([n * n, new_h, new_w, c])  # [depth, h, w, c]
                    d = 0
                    for i in range(n):
                        for j in range(n):
                            lf_extra[d, :, :, :] = lf2d[i: h: n, j: w: n, :]
                            d += 1
                else:
                    raise Exception('unknown mode : %s' % mode)
        
            return lf_extra
    
        def normalize(x):
            max_ = np.max(x) * 1.1
            x = x / (max_ / 2.)
            x = x - 1
            return x
    
        def _load_imgs(img_file, t2d=True):
            if t2d:
                image = imageio.imread(img_file)
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]
                img = normalize(image)
                img = lf_extract_fn(img, n_num=self.n_num, padding=False)
            else:
                image = imageio.volread(img_file)
                img = normalize(image)
                img = rearrange3d_fn(img)
    
            img = img.astype(np.float32, casting='unsafe')
            return img
        
        training_data_lf2d = _load_imgs(self.nm_lr2d[idx], True)
        X = np.transpose(training_data_lf2d, (2, 0, 1))
        
        if self.test_only:
            training_data_hr3d = _load_imgs(self.nm_hr3d[idx], False)
            name = os.path.basename(self.nm_hr3d[idx])[:-4]
            Y = np.transpose(training_data_hr3d, (2, 0, 1))
        else:
            training_data_hr3d = _load_imgs(self.nm_hr3d[idx], False)
            Y = np.transpose(training_data_hr3d, (2, 0, 1))
            name = ''
        
        return Y, X, name

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        
        hr, lr, filename = self._load_dataset(idx)
        
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.rgb_range)).float()
        return lr, hr, filename

    def __len__(self):
        if self.istrain:
            if self.length < 0:
                return self.lenth
            else:
                return self.length
        else:
            return self.lenth

    def _get_index(self, idx):
        if self.istrain:
            if self.length < 0:
                return idx % self.lenth
            else:
                return idx % self.length + self.length * random.randint(0, self.lenth // self.length - 1)
        else:
            return idx % self.lenth

   
# Inheritted from CARE
class PercentileNormalizer(object):
    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=torch.float32, **kwargs):
        if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
            raise ValueError
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs
    
    def before(self, img, axes):
        if len(axes) != img.ndim:
            raise ValueError
        channel = None if axes.find('C') == -1 else axes.find('C')
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img.detach().cpu().numpy(), self.pmin, axis=axes, keepdims=True).astype(np.float32, copy=False)
        self.ma = np.percentile(img.detach().cpu().numpy(), self.pmax, axis=axes, keepdims=True).astype(np.float32, copy=False)
        return (img - self.mi) / (self.ma - self.mi + 1e-20)
    
    def after(self, img):
        if not self.do_after():
            raise ValueError
        alpha = self.ma - self.mi
        beta = self.mi
        return (alpha * img + beta).astype(np.float32, copy=False)
    
    def do_after(self):
        return self._do_after

