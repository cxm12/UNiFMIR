import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import random
torch.backends.cudnn.enabled = False
import utility
from utility import savecolorim
import loss
import argparse
from mydata import normalize, PercentileNormalizer, np2Tensor, Image, glob, imread, imsave
from torch.utils.data import dataloader
import torch.nn.utils as utils
import model
import os
from decimal import Decimal
import imageio
import numpy as np
from scipy import signal


datamin, datamax = 0, 100  #
gpu = torch.cuda.is_available()


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--test_only', action='store_true', default=test_only,  #
                        help='set this option to test the model')
    parser.add_argument('--cpu', action='store_true', default=not gpu, help='cpu only')
    parser.add_argument('--task', type=int, default=task)
    parser.add_argument('--resume', type=int, default=resume, help='-2:best;-1:latest; 0:pretrain; >0: resume')
    parser.add_argument('--pre_train', type=str, default=pretrain, help='pre-trained model directory')
    parser.add_argument('--save', type=str, default=savename, help='file name to save')
    
    # Data specifications
    parser.add_argument('--test_every', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=100, help='')
    parser.add_argument('--epochs', type=int, default=epoch, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=batch, help='input batch size for training')
    parser.add_argument('--patch_size', type=int, default=Patch, help='input batch size for training')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)

    parser.add_argument('--load', type=str, default='', help='file name to load')
    parser.add_argument('--lr', type=float, default=learningrate, help='learning rate')
    parser.add_argument('--decay', type=str, default='200', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')

    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument('--n_resblocks', type=int, default=8, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=32, help='number of feature maps')
    parser.add_argument('--save_models', action='store_true', default=True, help='save all intermediate models')
    parser.add_argument('--template', default='.', help='You can set various templates in option.py')
    parser.add_argument('--scale', type=str, default='1', help='super resolution scale')
    parser.add_argument('--chop', action='store_true', default=True, help='enable memory-efficient forward')
    parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for test')
    # Model specifications
    parser.add_argument('--act', type=str, default='relu', help='activation function')
    parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
    parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'), help='FP precision for test (single | half)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    # Optimization specifications
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')
    
    # Loss specifications
    parser.add_argument('--loss', type=str, default='1*L1+1*L2', help='loss function configuration')
    
    args = parser.parse_args()
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    
    return args


class DeepBacsdataloader(data.Dataset):
    def __init__(self, name='', train=True, patch_size=64, test_only=False, rootdatapath='', scale=2):
        self.patch_size = patch_size
        self.rgb_range = 1
        self.name = name
        self.train = train
        self.test_only = test_only
        self.rootdatapath = rootdatapath
        self.scale = scale
        if train:
            if task == 1:
                self.filenames = glob.glob(rootdatapath + 'train/SIM/*.tif')
            elif task == 2:
                self.filenames = glob.glob(rootdatapath + 'train/high_SNR/*.tif')
        else:
            if task == 1:
                self.filenames = glob.glob(rootdatapath + 'test/SIM/*.tif')
                if scale == 2:
                    self.images_hr, self.images_lr, self.name = self._scan()
                else:
                    self.images_hr, self.images_lr, self.name = self._scan_s1()
            elif task == 2:
                self.filenames = glob.glob(rootdatapath + 'test/high_SNR/*.tif')
                self.images_hr, self.images_lr, self.name = self._scan_s1()
    
    def _scan(self):
        assert task == 1
        list_hr, list_lr, nm = [], [], []
        for fi in self.filenames:
            lr = np.array(imread(fi.replace('SIM', 'WF')))
            h, w = lr.shape
            hr = Image.fromarray(imread(fi))
            hr = np.array(hr.resize((w * 2, h * 2)))
            nm.append(fi[len(self.rootdatapath + 'test/SIM/'):])
            # nm.append(fi[fi.rfind('\\') + 2:])
            list_hr.append(np.expand_dims(hr, -1))
            list_lr.append(np.expand_dims(lr, -1))
        return list_hr, list_lr, nm

    def _scan_s1(self):
        list_hr, list_lr, nm = [], [], []
        for fi in self.filenames:
            hr = np.array(imread(fi))
            if task == 1:
                lr = np.array(imread(fi.replace('SIM', 'WF')))
                nm.append(fi[len(self.rootdatapath + 'test/SIM/'):])
            elif task == 2:
                lr = np.array(imread(fi.replace('high_SNR', 'low_SNR')))
                nm.append(fi[len(self.rootdatapath + 'test/high_SNR/'):])
                
            list_hr.append(np.expand_dims(hr, -1))
            list_lr.append(np.expand_dims(lr, -1))
        return list_hr, list_lr, nm

    def loadtr(self, idx):
        assert task == 1
        fi = self.filenames[idx]
        nm = fi[len(self.rootdatapath + 'train/SIM/'):]
        # nm = fi[fi.rfind('\\') + 2:]
        # 'JE2NileRed_oilp22_PMP_101220_.tif'  uint16
        lr = np.array(imread(fi.replace('SIM', 'WF')))  # 1024*1024
        height, width = lr.shape
        hr = Image.fromarray(imread(fi))
        hr = np.array(hr.resize((width * 2, height * 2)))

        i = np.random.randint(0, height - self.patch_size)
        j = np.random.randint(0, width - self.patch_size)
        lr1 = np.expand_dims(lr[i:i + self.patch_size, j:j + self.patch_size], -1)
        hr1 = np.expand_dims(hr[i * 2:i * 2 + self.patch_size * 2, j * 2:j * 2 + self.patch_size * 2], -1)
        # X1 = []
        # Y1 = []
        # for n in range(len(X)):
        #     for i in range(0, width, self.patch_size):
        #         for j in range(0, height, self.patch_size):
        #             if j + self.patch_size >= height and i + self.patch_size >= width:
        #                 X1.append(X[n][height - self.patch_size:, width - self.patch_size:, :])
        #                 Y1.append(Y[n][height * 2 - self.patch_size * 2:, width * 2 - self.patch_size * 2:, :])
        #             elif j + self.patch_size >= height:
        #                 X1.append(X[n][height - self.patch_size:, i:i + self.patch_size, :])
        #                 Y1.append(Y[n][height * 2 - self.patch_size * 2:, i * 2:i * 2 + self.patch_size * 2, :])
        #             elif i + self.patch_size >= width:
        #                 X1.append(X[n][j:j + self.patch_size, width - self.patch_size:, :])
        #                 Y1.append(Y[n][j * 2:j * 2 + self.patch_size * 2:, width * 2 - self.patch_size * 2:, :])
        #             else:
        #                 X1.append(X[n][j:j + self.patch_size, i:i + self.patch_size, :])
        #                 Y1.append(Y[n][j * 2:j * 2 + self.patch_size * 2, i * 2:i * 2 + self.patch_size * 2, :])
        
        return lr1, hr1, nm

    def loadtr_s1(self, idx):
        fi = self.filenames[idx]
        hr = np.array(imread(fi))
        if task == 1:
            nm = fi[len(self.rootdatapath + 'train/SIM/'):]
            lr = np.array(imread(fi.replace('SIM', 'WF')))  # 1024*1024
        elif task == 2:
            nm = fi[len(self.rootdatapath + 'train/high_SNR/'):]
            lr = np.array(imread(fi.replace('high_SNR', 'low_SNR')))  # 1024*1024
        height, width = lr.shape

        i = np.random.randint(0, height - self.patch_size)
        j = np.random.randint(0, width - self.patch_size)
        lr1 = np.expand_dims(lr[i:i + self.patch_size, j:j + self.patch_size], -1)
        hr1 = np.expand_dims(hr[i:i + self.patch_size, j:j + self.patch_size], -1)
    
        return lr1, hr1, nm

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        if self.train:
            if self.scale == 2:
                lr, hr, filename = self.loadtr(idx)
            else:
                lr, hr, filename = self.loadtr_s1(idx)
        else:
            lr, hr, filename = self.images_lr[idx], self.images_hr[idx], self.name[idx]
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.rgb_range
        if task == 2:
            lr = np.concatenate([lr, hr, lr, lr, lr], -1)
            # print(lr.shape, ' = lr.shape')
            
        pair = (lr, hr)
        pair_t = np2Tensor(*pair)
        
        return pair_t[0], pair_t[1], filename
    
    def __len__(self):
        print('len(self.images_hr)', len(self.filenames))
        if self.train:
            # return 16
            return len(self.filenames)
        else:
            return len(self.images_hr)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.filenames)
        else:
            return idx


class DeepBacsdataloaderK_fold(data.Dataset):
    def __init__(self, name='', train=True, patch_size=64, test_only=False, rootdatapath='', k=0):
        self.patch_size = patch_size
        self.rgb_range = 1
        self.name = name
        self.train = train
        self.test_only = test_only
        self.rootdatapath = rootdatapath
        if task == 1:
            self.filenameste = glob.glob(rootdatapath + 'test/SIM/*.tif')
            self.filenames = glob.glob(rootdatapath + 'train/SIM/*.tif')
        elif task == 2:
            self.filenameste = glob.glob(rootdatapath + 'test/high_SNR/*.tif')
            self.filenames = glob.glob(rootdatapath + 'train/high_SNR/*.tif')
        print('rootdatapath = ', rootdatapath + 'test/high_SNR/*.tif')
        print(len(self.filenames), len(self.filenameste))
        self.filenames.extend(self.filenameste)
        self.filenames.sort()
        startid = k * (len(self.filenames) // 5)
        endid = (k + 1) * (len(self.filenames) // 5)
        
        print('K=%d \ntrain data range = %d~%d +  %d~%d \ntest data range = %d~%d' % (k, 0, startid, endid, -1, startid, endid))
        self.filenameste = self.filenames[startid:endid]
        self.filenamestr = self.filenames[0:startid] + self.filenames[endid:]
        
        if not train:
            if task == 1:
                if scale == 2:
                    self.images_hr, self.images_lr, self.name = self._scan()
                else:
                    self.images_hr, self.images_lr, self.name = self._scan_s1()
            elif task == 2:
                self.images_hr, self.images_lr, self.name = self._scan_s1()
    
    def _scan(self):
        assert task == 1
        list_hr, list_lr, nm = [], [], []
        for fi in self.filenameste:
            nm.append(fi[fi.rfind('SIM/') + 4:])
            lr = np.array(imread(fi.replace('SIM', 'WF')))
            h, w = lr.shape
            hr = Image.fromarray(imread(fi))
            hr = np.array(hr.resize((w * 2, h * 2)))
            
            list_hr.append(np.expand_dims(hr, -1))
            list_lr.append(np.expand_dims(lr, -1))
        return list_hr, list_lr, nm

    def _scan_s1(self):
        list_hr, list_lr, nm = [], [], []
        for fi in self.filenameste:
            hr = np.array(imread(fi))
            if task == 1:
                lr = np.array(imread(fi.replace('SIM', 'WF')))
                nm.append(fi[fi.rfind('SIM') + 4:])
                # nm.append(fi[len(self.rootdatapath + 'test/SIM/'):].replace('\\', ''))
            elif task == 2:
                lr = np.array(imread(fi.replace('high_SNR', 'low_SNR')))
                nm.append(fi[fi.rfind('high_SNR') + 9:])
                
            list_hr.append(np.expand_dims(hr, -1))
            list_lr.append(np.expand_dims(lr, -1))
            # print('nm = ', nm)
        return list_hr, list_lr, nm

    def loadtr(self, idx):
        assert task == 1
        fi = self.filenamestr[idx]

        nm = fi[fi.rfind('SIM') + 4:]
        # print(nm)
        lr = np.array(imread(fi.replace('SIM', 'WF')))  # 1024*1024
        height, width = lr.shape
        hr = Image.fromarray(imread(fi))
        hr = np.array(hr.resize((width * 2, height * 2)))

        i = np.random.randint(0, height - self.patch_size)
        j = np.random.randint(0, width - self.patch_size)
        lr1 = np.expand_dims(lr[i:i + self.patch_size, j:j + self.patch_size], -1)
        hr1 = np.expand_dims(hr[i * 2:i * 2 + self.patch_size * 2, j * 2:j * 2 + self.patch_size * 2], -1)

        return lr1, hr1, nm

    def loadtr_s1(self, idx):
        fi = self.filenamestr[idx]
        hr = np.array(imread(fi))
        if task == 1:
            nm = fi[fi.rfind('SIM') + 4:]
            lr = np.array(imread(fi.replace('SIM', 'WF')))  # 1024*1024
        elif task == 2:
            nm = fi[fi.rfind('high_SNR') + 9:]
            lr = np.array(imread(fi.replace('high_SNR', 'low_SNR')))  # 1024*1024
        height, width = lr.shape
    
        i = np.random.randint(0, height - self.patch_size)
        j = np.random.randint(0, width - self.patch_size)
        lr1 = np.expand_dims(lr[i:i + self.patch_size, j:j + self.patch_size], -1)
        hr1 = np.expand_dims(hr[i:i + self.patch_size, j:j + self.patch_size], -1)

        return lr1, hr1, nm

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        if self.train:
            if scale == 2:
                lr, hr, filename = self.loadtr(idx)
            else:
                lr, hr, filename = self.loadtr_s1(idx)
        else:
            lr, hr, filename = self.images_lr[idx], self.images_hr[idx], self.name[idx]
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.rgb_range
        if task == 2 and ('CARE' not in savename):
            lr = np.concatenate([lr, hr, lr, lr, lr], -1)
            # print(lr.shape, ' = lr.shape')
        
        pair = (lr, hr)
        pair_t = np2Tensor(*pair)
        
        return pair_t[0], pair_t[1], filename
    
    def __len__(self):
        if self.train:
            print('Train len(self.filenames)', len(self.filenamestr))
            return len(self.filenamestr)
        else:
            print('Test len(self.images_hr)', len(self.images_hr))
            return len(self.images_hr)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.filenamestr)
        else:
            return idx


def to_color(arr, pmin=1, pmax=99.8, gamma=1., colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """Converts a 2D or 3D stack to a colored image (maximal 3 channels).

            Parameters
            ----------
            arr : numpy.ndarray
                2D or 3D input data
            pmin : float
                lower percentile, pass -1 if no lower normalization is required
            pmax : float
                upper percentile, pass -1 if no upper normalization is required
            gamma : float
                gamma correction
            colors : list
                list of colors (r,g,b) for each channel of the input

            Returns
            -------
            numpy.ndarray
                colored image
            """
    if not arr.ndim in (2, 3):
        raise ValueError("only 2d or 3d arrays supported")
    
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    
    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)
    
    out = np.zeros(arr.shape[1:] + (3,))
    
    eps = 1.e-20
    if pmin >= 0:
        mi = np.percentile(arr, pmin, axis=(1, 2), keepdims=True)
    else:
        mi = 0
    
    if pmax >= 0:
        ma = np.percentile(arr, pmax, axis=(1, 2), keepdims=True)
    else:
        ma = 1. + eps
    
    arr_norm = (1. * arr - mi) / (ma - mi + eps)
    
    for i_stack, col_stack in enumerate(colors):
        if i_stack >= len(arr):
            break
        for j, c in enumerate(col_stack):
            out[..., j] += c * arr_norm[i_stack]
    
    return np.clip(out, 0, 1)


def compute_ssim(X, Y, win_size=7, data_range=None):
    """ compute structural similarity of two images Parameters: -----------
            X, Y: numpy array two images data win_size: int window size of image patch for computing ssim
            of one single position data_range: int or float maximum dynamic range of image data type Returns:
            --------
            mssim: float mean structural similarity ssim_map: numpy array (float) structural similarity map,
            same shape as input images """
    
    def _ssim_one_channel(X, Y, win_size, data_range):
        """ compute structural similarity of two single channel images Parameters: ----------- X, Y: numpy array two images data win_size: int window size of image patch for computing ssim of one single position data_range: int or float maximum dynamic range of image data type Returns: -------- mssim: float mean structural similarity ssim_map: numpy array (float) structural similarity map, same shape as input images """
        X, Y = normalize1(X, Y, data_range)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        num = win_size ** 2
        kernel = np.ones([win_size, win_size]) / num
        mean_map_x = convolve2d(X, kernel)
        mean_map_y = convolve2d(Y, kernel)
        mean_map_xx = convolve2d(X * X, kernel)
        mean_map_yy = convolve2d(Y * Y, kernel)
        mean_map_xy = convolve2d(X * Y, kernel)
        cov_norm = num / (num - 1)
        var_x = cov_norm * (mean_map_xx - mean_map_x ** 2)
        var_y = cov_norm * (mean_map_yy - mean_map_y ** 2)
        covar_xy = cov_norm * (mean_map_xy - mean_map_x * mean_map_y)
        A1 = 2 * mean_map_x * mean_map_y + C1
        A2 = 2 * covar_xy + C2
        B1 = mean_map_x ** 2 + mean_map_y ** 2 + C1
        B2 = var_x + var_y + C2
        ssim_map = (A1 * A2) / (B1 * B2)
        mssim = np.mean(ssim_map)
        return mssim, ssim_map
    
    def convolve2d(image, kernel):
        """ convolve single channel image and kernel Parameters: ----------- image: numpy array single channel image data kernel: numpy array kernel data Returns: -------- result: numpy array image data, same shape as input image """
        result = signal.convolve2d(image, kernel, mode='same', boundary='fill')
        return result
    
    def normalize1(X, Y, data_range):
        """ convert dtype of two images to float64, and then normalize them by data_range Paramters: ---------- X, Y: numpy array two images data data_range: int or float maximum dynamic range of image data type Returns: -------- X, Y: numpy array two images """
        X = X.astype(np.float64) / data_range
        Y = Y.astype(np.float64) / data_range
        return X, Y
    
    assert X.shape == Y.shape, "X, Y must have same shape"
    assert X.dtype == Y.dtype, "X, Y must have same dtype"
    assert win_size <= np.min(X.shape[0:2]), \
        "win_size should be <= shorter edge of image"
    assert win_size % 2 == 1, "win_size must be odd"
    if data_range is None:
        if 'float' in str(X.dtype):
            data_range = 1
        elif 'uint8' in str(X.dtype):
            data_range = 255
        else:
            raise ValueError(
                'image dtype must be uint8 or float when data_range is None')
    X = np.squeeze(X)
    Y = np.squeeze(Y)
    if X.ndim == 2:
        mssim, ssim_map = _ssim_one_channel(X, Y, win_size, data_range)
    elif X.ndim == 3:
        ssim_map = np.zeros(X.shape)
        for i in range(X.shape[2]):
            _, ssim_map[:, :, i] = _ssim_one_channel(
                X[:, :, i], Y[:, :, i], win_size, data_range)
        mssim = np.mean(ssim_map)
    else:
        raise ValueError("image dimension must be 2 or 3")
    return mssim, ssim_map


class PreTrainer():
    def __init__(self, args, my_model, my_loss, ckp, k=-1):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.scale = args.scale
        self.ckp = ckp
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.normalizer = PercentileNormalizer(2, 99.8)
        self.normalizerhr = PercentileNormalizer(2, 99.8)
        self.sepoch = args.resume
        self.test_only = args.test_only

        self.bestpsnr = 0
        self.bestep = 0
        self.epoch = 0
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        
        self.error_last = 1e8
        rp = os.path.dirname(__file__)
        if k == -1:
            if not test_only:
                self.loader_train = dataloader.DataLoader(
                    DeepBacsdataloader(name=datasetname, train=True, rootdatapath=traindatapath,
                                       scale=scale, patch_size=args.patch_size),
                    batch_size=args.batch_size, shuffle=False, pin_memory=not args.cpu, num_workers=0)
            self.loader_test = [dataloader.DataLoader(
                DeepBacsdataloader(name=datasetname, train=False, rootdatapath=traindatapath, scale=scale),
                batch_size=1, shuffle=False, pin_memory=not args.cpu, num_workers=0)]
            self.dir = os.path.join(rp, 'experiment', self.args.save)
        else:
            if not test_only:
                self.loader_train = dataloader.DataLoader(
                    DeepBacsdataloaderK_fold(name=datasetname, train=True, rootdatapath=traindatapath, k=k),
                    batch_size=args.batch_size, shuffle=False, pin_memory=not args.cpu, num_workers=0)
            self.loader_test = [dataloader.DataLoader(
                DeepBacsdataloaderK_fold(name=datasetname, train=False, rootdatapath=traindatapath, k=k),
                batch_size=1, shuffle=False, pin_memory=not args.cpu, num_workers=0)]
            self.dir = os.path.join(rp, 'experiment', self.args.save, 'K%d/' % k)
        
        os.makedirs(self.dir, exist_ok=True)
        print('Loaded data done!  self.dir = ', self.dir)

        self.tsk = task
        self.model.scale = scale
        print('Task/model.scale = ', self.tsk, self.model.scale)

        if not self.args.test_only:
            self.testsave = self.dir + '/Valid/'
            os.makedirs(self.testsave, exist_ok=True)
            self.file = open(self.testsave + "TrainPsnr.txt", 'w')
    
    def finetune(self):
        self.pslst = []
        self.sslst = []
        self.msslst = []

        self.loss.step()
        if self.sepoch > 0:
            epoch = self.sepoch
            self.sepoch = 0
            self.epoch = epoch
        else:
            epoch = self.epoch
        
        lr = self.optimizer.get_lr()
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        timer_data, timer_model = utility.timer(), utility.timer()
        
        self.model.train()
        
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            sr = self.model(lr, self.tsk)
            loss = self.loss(sr, hr)
            
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()
            timer_model.hold()
            
            if batch % self.args.test_every == 0:
                self.loss.end_log(len(self.loader_train))
                self.error_last = self.loss.log[-1, -1]
                self.optimizer.schedule()
                if task == 1:
                    psnr, ssim, mssim = self.testsr(batch, epoch)
                elif task == 2:
                    psnr, ssim = self.testdenoise(batch, epoch)
                self.pslst.append(psnr)
                self.sslst.append(ssim)
                self.msslst.append(mssim)

                self.model.train()
                self.loss.step()
                lr = self.optimizer.get_lr()
                print('Evaluation -- Batch%d/Epoch%d' % (batch, epoch))
                self.ckp.write_log(
                    'Batch%d/Epoch%d' % (batch, epoch) + '\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
                self.loss.start_log()
                
            if batch % self.args.print_every == 0:
                print('Batch%d/Epoch%d, Loss = ' % (batch, epoch), loss)
                print('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format((batch + 1) * self.args.batch_size,
                                len(self.loader_train.dataset), self.loss.display_loss(batch), timer_model.release(), timer_data.release()))
            timer_data.tic()
        
        self.file.write('\n PSNR \n' + str(self.pslst) + '\n SSIM \n' + str(self.sslst) + '\n MSSIM \n' + str(self.msslst))
        self.file.flush()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        self.model.save(self.dir, epoch, is_best=False)
        print('save model Epoch%d' % epoch)

    def finetune_kfold(self):
        self.pslst = []
        self.sslst = []
        self.losslst = []
    
        self.loss.step()
        self.loss.start_log()
        timer_data, timer_model = utility.timer(), utility.timer()
    
        self.model.train()
    
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
        
            self.optimizer.zero_grad()
            sr = self.model(lr, self.tsk)
            loss = self.loss(sr, hr)
        
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()
            timer_model.hold()
        
            if batch % self.args.test_every == 0:
                sr2dim = np.float32(
                    normalize(np.squeeze(sr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                hr2dim = np.float32(
                    normalize(np.squeeze(hr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                self.losslst.append(loss)
            
                self.loss.end_log(len(self.loader_train))
                self.error_last = self.loss.log[-1, -1]
                self.optimizer.schedule()
                if task == 1:
                    psnr, ssim, mssim = self.testsr(batch, self.epoch)
                elif task == 2:
                    psnr, ssim = self.testdenoise(batch, self.epoch)
                self.pslst.append(psnr)
                self.sslst.append(ssim)
            
                self.model.train()
                self.loss.step()
                print('Evaluation -- Batch%d/Epoch%d' % (batch, self.epoch))
                self.loss.start_log()
        
            if batch % self.args.print_every == 0:
                print('training patch- PSNR/SSIM = %f/%f' % (psm, ssmm))
                print('[{}/{}]\t{}'.format((batch + 1) * self.args.batch_size,
                                           len(self.loader_train.dataset), self.loss.display_loss(batch)))
            timer_data.tic()
    
        self.file.write('\n PSNR \n' + str(self.pslst) + '\n SSIM \n' + str(self.sslst))
        self.file.flush()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        if self.epoch > self.args.epochs - 1:
            self.model.save(self.dir, self.epoch, is_best=False)
            print('save model Epoch%d' % self.epoch)
        
    # # -------------------------- SR --------------------------
    def testsr(self, batch=0, epoch=None):
        if self.test_only:
            self.testsave = self.dir + '/results/model%d/' % self.args.resume
            os.makedirs(self.testsave, exist_ok=True)
            print('self.testsave = ', self.testsave)

        torch.set_grad_enabled(False)
        self.ckp.write_log('\nEvaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        num = 0
        pslst = []
        sslst = []
        nmlst = []
        mssimlst = []
        for idx_data, (lr, hr, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            num += 1
            if not self.test_only and num > 1:
                break
            lr, hr = self.prepare(lr, hr)
            
            sr = self.model(lr, 1)
            sr = utility.quantize(sr, self.args.rgb_range)
            hr = utility.quantize(hr, self.args.rgb_range)
            print('hr.shape = sr.shape =', hr.shape, sr.shape)

            name = '{}.tif'.format(filename[0][:-4])
            sr0 = sr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            hr0 = hr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            ps, ss = utility.compute_psnr_and_ssim(sr0, hr0)
            pslst.append(ps)
            sslst.append(ss)
            print(ps, ss)
            
            if self.test_only:
                print('save to ', self.testsave + 'SROri-' + name)
                imageio.imwrite(self.testsave + 'SROri-' + name, sr0)
                mssim, ssimmap = compute_ssim(np.uint8(hr0), np.uint8(sr0), win_size=7, data_range=None)
                mssimlst.append(mssim)
                print(' ps, ss = ', ps, ss, mssim)
                im = ssimmap
                im = np.stack(map(to_color, im)) if 1 < im.shape[-1] <= 3 else im
                ndim_allowed = 2 + int(1 <= im.shape[-1] <= 3)
                proj_axis = tuple(range(1, 1 + max(0, im[0].ndim - ndim_allowed)))
                im = np.max(im, axis=proj_axis)
                plt.rc('font', family='Times New Roman')
                plt.imsave(self.testsave + name[:-4] + '-ssim.png', im, cmap='YlOrRd_r')
                
                normalized = sr.mul(255 / self.args.rgb_range).byte().cpu()
                sr = np.squeeze(normalized.detach().numpy())
                imageio.imwrite(self.testsave + name, sr)
                savecolorim(self.testsave + name[:-4] + '-Color.png', sr, norm=False)
    
                normalizedhr = hr[0].mul(255 / self.args.rgb_range).byte().cpu()
                hr1 = np.squeeze(normalizedhr.detach().numpy())
                savecolorim(self.testsave + name[:-4] + '-ColorHR.png', hr1, norm=False)

                normalizedlr = lr[0].mul(255 / self.args.rgb_range).byte().cpu()
                lr1 = np.squeeze(normalizedlr.detach().numpy())
                savecolorim(self.testsave + name[:-4] + '-ColorLR.png', lr1, norm=False)

                hr = np.float32(normalize(hr1, datamin, datamax, clip=True)) * 255
                sr = np.float32(normalize(sr, datamin, datamax, clip=True)) * 255
                savecolorim(self.testsave + name[:-4] + '-MeandfnoNormC.png', sr - hr, norm=False)
                
        psnrall = np.mean(np.array(pslst))
        ssimall = np.mean(np.array(sslst))
        mssimall = np.mean(np.array(mssimlst))
        if self.test_only:
            file = open(self.testsave + "result.txt", 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(pslst) + '\n SSIM \n' + str(sslst))
            file.close()
        else:
            if psnrall > self.bestpsnr:
                self.bestpsnr = psnrall
                self.bestep = epoch
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        
        print('num psnrall, ssimall, mssimall = ', num, psnrall, ssimall, mssimall)
        print('bestpsnr/epoch = ', self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrall, ssimall, mssimall

    def testdenoise(self, batch=0, epoch=None):
        if self.test_only:
            self.testsave = self.dir + '/results/model%d/' % self.args.resume
            os.makedirs(self.testsave, exist_ok=True)
    
        datamin, datamax = self.args.datamin, self.args.datamax
        torch.set_grad_enabled(False)
    
        self.ckp.write_log('\nEvaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
    
        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.test_only and num >= 1:
                break
    
            num += 1
            nmlst.append(filename)
            print('filename = ', filename)
            if filename[0] == '':
                name = 'im%d' % idx_data
            else:
                name = '{}'.format(filename[0][:-4])
            
            # 1.3D norm 2 998
            lrt = self.normalizer.before(lrt, 'CZYX')  # [1,5,512,512]
            hrt = self.normalizerhr.before(hrt, 'CZYX')  # [1,1,512,512]
            lrt, hrt = self.prepare(lrt, hrt)
            
            denoiseim = self.model(lrt, 2)  # [1,1,512,512]
            sr = np.float32(denoiseim.cpu().detach().numpy())
            
            lr = np.squeeze(lrt.cpu().detach().numpy())[0, :, :]
            hr = np.squeeze(hrt.cpu().detach().numpy())
            print('hr.shape = ', hr.shape)
            
            # 3.3D norm 2 998 tiff save；
            sr = np.squeeze(self.normalizer.after(sr))
            hr = np.squeeze(self.normalizerhr.after(hr))

            # 4.3D norm 0 100 psnr
            sr255 = np.squeeze(np.float32(normalize(sr, datamin, datamax, clip=True))) * 255
            hr255 = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255
            lr255 = np.float32(normalize(lr, datamin, datamax, clip=True)) * 255
            
            if self.test_only:
                savecolorim(self.testsave + name + '-HR.png', hr255)
                imsave(self.testsave + name + '.tif', sr)

            savecolorim(self.testsave + name + '-dfnoNormC.png', sr255 - hr255, norm=False)
            savecolorim(self.testsave + name + '-C.png', sr255)

            ##  PSNR/SSIM
            mse = np.sum(np.power(sr255 - hr255, 2))
            psnr1, ssim = utility.compute_psnr_and_ssim(sr255, hr255)
            psml, ssmml = utility.compute_psnr_and_ssim(lr255, hr255)
            print('Normalized Image %s - PSNR/SSIM/MSE = %f/%f/%f' % (name, psnr1, ssim, mse))
            print('Normalized LR PSNR/SSIM = %f/%f' % (psml, ssmml))
            sslst.append(ssim)
            pslst.append(psnr1)
    
        psnrm = np.mean(np.array(pslst))
        ssimm = np.mean(np.array(sslst))
        print('psnr, num, psnrall1, ssimall = ', psnrm, num, ssimm)
    
        if self.test_only:
            print('+++++++++ mean denoise++++++++++++', psnrm, ssimm)
            file = open(self.testsave + '/Psnrssim_Im_patch.txt', 'w')
            file.write('\n \n +++++++++ mean denoise ++++++++++++ \n PSNR/SSIM \n ')
            file.write('Name \n' + str(nmlst) + '\n PSNR = ' + str(pslst) + '\n SSIM = ' + str(sslst))
            file.close()
        else:
            if psnrm > self.bestpsnr:
                self.bestpsnr = psnrm
                self.bestep = epoch
                self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrm, ssimm
        
    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]
    
    def terminate(self):
        if self.test_only:
            return False
        else:
            self.epoch = self.epoch + 1
            if self.epoch > self.args.epochs:
                self.file.close()
            return self.epoch <= self.args.epochs


def finetune_kfold():
    resultlst = []
    
    for k in range(0, 5):
        checkpoint = utility.checkpoint(args)
        assert checkpoint.ok
        
        unimodel = model.UniModel(args, tsk=task, srscale=scale)
        
        if args.test_only:
            args.pre_train = './experiment/Uni-SwinIR%s/K-fold/K%d/model_best.pt' % (datasetname, k)

        _model = model.Model(args, checkpoint, unimodel)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        
        t = PreTrainer(args, _model, _loss, checkpoint, k=k)
        
        if args.test_only:
            if task == 1:
                psnrall, ssimall, mssim = t.testsr(0, 0)
                resultlst.append('dataset %s-K%d \n' % (datasetname, k))
                resultlst.append('psnrall, ssimall, mssim = %s/%s/%s' % (str(psnrall), str(ssimall), str(mssim)))
            elif task == 2:
                psnrm, ssimm = t.testdenoise(0, 0)
                resultlst.append('dataset %s-K%d \n' % (datasetname, k))
                resultlst.append('psnrall, ssimall = %s/%s' % (str(psnrm), str(ssimm)))
        else:
            while t.terminate():
                t.finetune_kfold()
        
        checkpoint.done()
        print(resultlst)


if __name__ == '__main__':
    K_fold = True  # False  #
    test_only =  True  # True  #
    task =   1  # 2  # 
    batch = 8
    resume = 0  # -2  #
    learningrate = 1e-4  # 0.00005  #
    scale = 1
    epoch = 100  # 500  #
    Patch = 128  # 64  #

    pretrain = './experiment/Uni-SwinIR/model_best.pt'  # '.'  #
    if task == 1:
        datasetname = 'DeepBacs_Data_Super-resolution_prediction_S.aureus'
        # datasetname = 'DeepBacs_Data_Super-resolution_prediction_E.coli'
    else:
        # datasetname = 'DeepBacs_Data_Denoising_E.coli_H-NS-mScarlet-I_dataset'
        datasetname = 'DeepBacs_Data_Denoising_E.coli_MreB'
        
    traindatapath = '/home/user2/dataset/microscope/DeepBacs/%s/' % datasetname

    if K_fold:
        savename = 'Uni-SwinIR%s/K-fold/' % (datasetname)
        args = options()
        torch.manual_seed(args.seed)
        finetune_kfold()
    else:
        savename = 'Uni-SwinIR%s/' % (datasetname)
        args = options()
        torch.manual_seed(args.seed)

        checkpoint = utility.checkpoint(args)
        assert checkpoint.ok
        unimodel = model.UniModel(args, tsk=task, srscale=scale)
        _model = model.Model(args, checkpoint, unimodel)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None

        t = PreTrainer(args, _model, _loss, checkpoint)

        if args.test_only:
            if task == 1:
                t.testsr(0, 0)
            elif task == 2:
                t.testdenoise(0, 0)
        else:
            while t.terminate():
                t.finetune()

        checkpoint.done()
