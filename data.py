import imageio
import glob
import torch.utils.data as data
from PIL import Image
import torch
import sys
import os
import numpy as np
from tifffile import imread
from scipy.ndimage.interpolation import zoom

sys.path.append('..')
from csbdeep.utils import normalize, axes_dict, axes_check_and_normalize, backend_channels_last, move_channel_for_backend
from csbdeep.func import IS_TF_1
from csbdeep.io import load_training_data


datamin, datamax = 0, 100  #


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
    print(X.shape, Y.shape)  # (18468, 128, 128, 1) (18468, 256, 256, 1)
    return X, Y, X_val, Y_val


class SR(data.Dataset):
    def __init__(self, args, name='CCPs', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.benchmark = benchmark
        self.dir_data = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/train/%s/my_training_data.npz' % args.data_test
        self.dir_demo = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/test/%s/LR/' % args.data_test
        
        self.input_large = (self.dir_demo != '')
        self.scale = args.scale
        if train:
            X, Y, X_val, Y_val = self.loadData()  # (18468, 128, 128, 1) (18468, 256, 256, 1)
            print('np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
            list_hr, list_lr = Y, X
        else:
            self.filenames = glob.glob(self.dir_demo + '*.tif')
            list_hr, list_lr, name = self._scan()            
            self.name = name

        self.images_hr, self.images_lr = list_hr, list_lr
        self.repeat = 1
        
    def loadData(self):
        patch_size = self.args.patch_size
        X, Y, X_val, Y_val = loadData(self.dir_data)  # (18468, 128, 128, 1) (18468, 256, 256, 1)

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
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.args.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.args.rgb_range
        pair = (lr, hr)   # (128, 128, 1) (256, 256, 1)
        pair_t = np2Tensor(*pair)
        
        return pair_t[0], pair_t[1], filename

    def __len__(self):
        print('len(self.images_hr)', len(self.images_hr))
        return len(self.images_hr) * self.repeat
        
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx


class Flourescenedenoise(data.Dataset):
    def __init__(self, args, istrain=True, c=1):
        self.args = args
        self.batch = 1  # args.batch_size  #
        self.datamin, self.datamax = args.datamin, args.datamax
        self.istrain = istrain
        
        if self.args.data_test:
            self.denoisegt = [self.args.data_test]
        else:
            if IS_TF_1:
                self.denoisegt = ['Denoising_Planaria']
            else:
                self.denoisegt = ['Denoising_Planaria', 'Denoising_Tribolium', 'Synthetic_tubulin_granules', 'Synthetic_tubulin_gfp']
        
        if istrain:
            self._scandenoisenpy()
        else:
            self._scandenoisetif(c)
        
        self.lenthdenoise = len(self.nm_lrdenoise)
        self.lenth = self.lenthdenoise // self.batch
        
        if istrain:
            print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
    
    def _scandenoisenpy(self):
        hr = []
        lr = []
        patch_size = self.args.patch_size
        
        if not IS_TF_1:
            datapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/'
        else:
            datapath = 'E:/file/python_project/Medical/CSBDeep-master/DataSet/'
        for i in self.denoisegt:
            # Planaria: X/Y  (17005, 16, 64, 64, 1)(895, 16, 64, 64, 1)  float32
            # Tr  (14725, 16, 64, 64, 1) (775, 16, 64, 64, 1)
            # Synthetic_tubulin_granules/channel_tubules (5872, 128, 128, 1)
            # Synthetic_tubulin_granules/channel_granules (2872, 128, 128, 1)
            # Synthetic_tubulin_gfp (5872, 128, 128, 1)
            if 'Denoising' in i:
                X, Y, X_val, Y_val = loadData(datapath + i + '/train_data/data_label.npz', axes='SCZYX')
            else:
                X, Y, X_val, Y_val = loadData(datapath + i + '/train_data/data_label.npz')
                
            print('Dataset:', i, 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
            print('X.shape, Y.shape, X_val.shape, Y_val.shape = ', X.shape, Y.shape, X_val.shape, Y_val.shape)
            height, width = X.shape[-3:-1]
            X = np.reshape(X, [-1, height, width, 1])
            Y = np.reshape(Y, [-1, height, width, 1])
            assert len(X) == len(Y)
            if not 'Denoising' in i:
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
        
        if not IS_TF_1:
            datapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/'
        else:
            datapath = 'E:/file/python_project/Medical/CSBDeep-master/DataSet/'
        
        if ('Planaria' in self.denoisegt[0]) or ('Tribolium' in self.denoisegt[0]):
            lr.extend(sorted(glob.glob(datapath + '%s/test_data/condition_%d/*.tif' % (self.denoisegt[0], c))))
            self.hrpath = datapath + '%s/test_data/GT/' % self.denoisegt[0]
            
        lr.sort()
        self.nm_lrdenoise = lr
    
    def __getitem__(self, idx):
        idx = idx % self.lenth
        if self.istrain:
            lr, hr, filename = self._load_file_denoise_npy(idx + self.args.inputchannel//2)
        else:
            lr, hr, filename, d = self._load_file_denoise(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()
        return lr, hr, filename
    
    def __len__(self):
        if self.istrain:
            return self.lenth - 2 * (self.args.inputchannel//2)
        else:
            return self.lenth
    
    def _load_file_denoise(self, idn):
        ps = 64
        filename, fmt = os.path.splitext(os.path.basename(self.nm_lrdenoise[idn]))
        # P [95,1024,1024] uint16 [0~33000]
        # T [45, 486, 954]

        rgb = np.float32(imread(self.hrpath + filename + fmt))  # / 65535
        rgblr = np.float32(imread(self.nm_lrdenoise[idn]))  # / 65535
                
        print('Test Denoise, ----> rgblr.max/min', rgblr.max(), rgblr.min(), rgblr.shape)  # 718.0 362.0 (2, 64, 64)
        return rgblr, rgb, filename, self.denoisegt[0]

    def _load_file_denoise_npy(self, idx):
        lr = []
        hr = []
        if self.args.inputchannel > 1:
            for i in range(self.batch):
                idn = (idx + i) % self.lenthdenoise
                hr.extend(self.nm_hrdenoise[idn:idn + 1])
                lr.extend(self.nm_lrdenoise[idn - self.args.inputchannel // 2:idn + self.args.inputchannel // 2 + 1])
            rgb = np.concatenate(hr, -1)  # 0~4.548696  [B, 64, 64, 1]
            rgblr = np.squeeze(np.concatenate(lr, -1))  # 0~87.93965  [B, 64, 64, 5]
            rgb = np.transpose(np.float32(rgb), (2, 0, 1))  # [5, 256, 256]
            rgblr = np.transpose(np.float32(rgblr), (2, 0, 1))
        else:
            for i in range(self.batch):
                idn = (idx + i) % self.lenthdenoise
                hr.append(self.nm_hrdenoise[idn])
                lr.append(self.nm_lrdenoise[idn])
    
            rgb = np.squeeze(np.concatenate(hr, -1))  # 0~4.548696
            rgblr = np.squeeze(np.concatenate(lr, -1))  # 0~87.93965
            rgb = np.transpose(np.float32(rgb), (2, 0, 1))  # [1, 256, 256]
            rgblr = np.transpose(np.float32(rgblr), (2, 0, 1))
        
        # print('Denoise, ----> rgblr.max()', rgblr.max(), rgblr.shape)
        return rgblr, rgb, ''

    def _load_file_denoisenorm(self, idn):
        filename, fmt = os.path.splitext(os.path.basename(self.nm_lrdenoise[idn]))
        # P [95,1024,1024] uint16 [0~33000]
        # T [45, 486, 954]
        rgb = imread(self.hrpath + filename + fmt)  # uint16
        rgb = np.float32(normalize(rgb, self.datamin, self.datamax, clip=True))
        rgblr = imread(self.nm_lrdenoise[idn])
        rgblr = np.float32(normalize(rgblr, self.datamin, self.datamax, clip=True))
    
        # print('Test Denoise, ----> rgblr.max/min', rgblr.max(), rgblr.min(), rgblr.shape)
        return rgblr, rgb, filename, self.denoisegt[0]

    def _load_file_denoise_npynorm(self, idx):
        lr = []
        hr = []
        for i in range(self.batch):
            idn = (idx + i) % self.lenthdenoise
            hr.append(self.nm_hrdenoise[idn])
            lr.append(self.nm_lrdenoise[idn])
    
        rgb = np.concatenate(hr, -1)  # 0~65535
        rgblr = np.concatenate(lr, -1)
        rgb = np.transpose(np.float32(normalize(rgb, self.datamin, self.datamax, clip=True)),
                           (2, 0, 1)) * self.args.rgb_range  # [1, 256, 256]
        rgblr = np.transpose(np.float32(normalize(rgblr, self.datamin, self.datamax, clip=True)),
                             (2, 0, 1)) * self.args.rgb_range
    
        rgb = np.float32(normalize(rgb, self.datamin, self.datamax, clip=True))
        rgblr = np.float32(normalize(rgblr, self.datamin, self.datamax, clip=True))
        # print('Denoise, ----> rgblr.max()', rgblr.max(), rgblr.shape)
        return rgblr, rgb, ''


class Flouresceneiso(data.Dataset):
    def __init__(self, args, istrain=True):
        self.args = args
        self.batch = 1
        self.datamin, self.datamax = 0, 100  # 0.1, 99.90  #
        self.istrain = istrain

        self.iso = ['Isotropic_Liver']  # ['Isotropic_Retina', 'Isotropic_Drosophila', 'Isotropic_Liver']  #
        if istrain:
            self._scanisonpy()
        else:
            self._scaniso()
                    
        if istrain:
            print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        
    def _scanisonpy(self):
        hr = []
        lr = []
        patch_size = self.args.patch_size
    
        if not IS_TF_1:
            datapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/Isotropic/'
        else:
            datapath = 'E:/file/python_project/Medical/CSBDeep-master/DataSet/Isotropic/'
        
        for i in self.iso:
            # Liver X/Y (3872, 128, 128, 1)
            # 'Isotropic_Drosophila': X/Y [(40270, 128, 128, 1)] X_val/Y_val [(4474, 128, 128, 1)]
            # 'Isotropic_Retina': X/Y [22852, 128, 128, 2] X_val/Y_val [2539, 128, 128, 2]
            X, Y, _, _ = loadData(datapath + '%s/train_data/data_label.npz' % i, axes='SCYX', validation_split=0.0)
            
            print('Dataset:', i, 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
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
        self.lenth = len(self.nm_lriso)

    def _scaniso(self):
        hr = []
        lr = []
        for i in self.iso:
            self.dir_lr = 'E:/file/python_project/Medical/CSBDeep-master/DataSet/Isotropic/%s/test_data/' % i
        
            if i == 'Isotropic_Liver':
                hr.append(self.dir_lr + 'input_subsample_1_groundtruth.tif')  # Liver [301, 752, 752]
                lr.append(self.dir_lr + 'input_subsample_8.tif')
            else:
                filenames = os.listdir(self.dir_lr)
                for fi in range(len(filenames)):
                    name = filenames[fi][:-4]
                    lr.append(self.dir_lr + name + '.tif')
        self.nm_hr, self.nm_lr = hr, lr
        self.lenth = len(self.nm_lr)

    def __getitem__(self, idx):
        idx = idx % self.lenth
        if self.istrain:
            lr, hr, filename = self._load_file_iso_npy(idx)
        else:
            lr, hr, filename = self._load_file_isotest(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()
        
        return lr, hr, filename
    
    def __len__(self):
        return self.lenth

    def _load_file_iso_npy(self, idx):
        lr = []
        hr = []
        for i in range(self.batch):
            idn = (idx + i) % self.lenth
            hr.append(self.nm_hriso[idn])
            lr.append(self.nm_lriso[idn])
    
        rgb = np.concatenate(hr, -1)  # 0~
        rgblr = np.concatenate(lr, -1)  # 0~
        rgb = np.transpose(np.float32(rgb), (2, 0, 1))  # [1, 128, 128]
        rgblr = np.transpose(np.float32(rgblr), (2, 0, 1))
    
        return rgblr, rgb, ''

    def _load_file_isotest(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))
        
        rgblr = np.float32(imread(self.nm_lr[idx]))
        
        if 'Isotropic_Liver' in self.nm_lr[idx]:
            # Liver [301, 752, 752]
            hrp = self.nm_lr[idx].replace('_8.tif', '_1_groundtruth.tif')
            rgb = np.float32(imread(hrp))
            return rgblr, rgb, filename
        elif 'Retina' in self.nm_lr[idx]:
            rgblr = np.transpose(zoom(rgblr, (10.2, 1, 1, 1), order=1), [0, 2, 3, 1])  # [35, 1024, 1024, 2]
        
        print('ISO Testset $$$$ ', i, ', ----> rgblr.max()', rgblr.max(), rgblr.shape)
        return rgblr, rgblr, filename


class Flouresceneproj(data.Dataset):
    def __init__(self, args, istrain=True, condition=0):
        self.args = args
        self.batch = 1
        self.istrain = istrain
        self.iso = ['Projection_Flywing']
        if istrain:
            self._scannpy()
        else:
            self._scan(condition)
            
        if istrain:
            print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')

    def load_training_data(self, file, axes=None, n_images=None, verbose=False):
        print('Begin np.load(file)', file)
        f = np.load(file)
        X, Y = f['X'], f['Y']
        Y = np.expand_dims(Y, 2)
        print(Y.ndim, Y.shape)
        # flying data_label  (17780,1,50,64,64), (17780,1,64,64)
        if axes is None:
            axes = f['axes']  # 'SCZYX'
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
    
        return X, Y  # , axes

    def _scannpy(self):
        patch_size = self.args.patch_size
        mytraindata = 2  # 0  # 1  #

        if mytraindata == 1:
            datapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/projection/data/my_training_data.npz'
            X, Y, _, _ = loadData(datapath, axes=None, validation_split=0.0)
        elif mytraindata == 2:
            if IS_TF_1:
                datapath = 'E:/file/python_project/Medical/CSBDeep-master/examples/projection/data/my_training_data.npz'
                datapath2 = 'E:/file/python_project/Medical/CSBDeep-master/DataSet/%s/train_data/data_label.npz' % \
                            self.iso[0]
            else:
                datapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/examples/projection/data/my_training_data.npz'
                datapath2 = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/%s/train_data/data_label.npz' % \
                            self.iso[0]
            X1, Y1, _, _ = loadData(datapath, axes=None, validation_split=0.0)
            X1l = []
            Y1l = []
            for n in range(len(X1)):  # 裁剪并保存patch
                for i in range(0, 128, 64):
                    for j in range(0, 128, 64):
                        X1l.append(X1[n][:, j:j + 64, i:i + 64, :])
                        Y1l.append(Y1[n][:, j:j + 64, i:i + 64, :])
            X1 = np.array(X1l)  # [3136, 50, 64, 64, 1]
            Y1 = np.array(Y1l)
            X2, Y2 = self.load_training_data(datapath2, axes='SCZYX', verbose=True)  # 0~38.15789, 0~4.73316
            X = np.concatenate([X1, X2], 0)  # [20916, 50, 64, 64, 1]
            Y = np.concatenate([Y1, Y2], 0)
        else:
            # data_label  (17780, 50, 64, 64, 1) (17780, 1, 64, 64, 1)
            datapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/%s/train_data/data_label.npz' % self.iso[0]
    
            X, Y = self.load_training_data(datapath, axes='SCZYX', verbose=True)  # 0~38.15789, 0~4.73316
                    
        print('Dataset:', self.iso[0], 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
        print('X.shape, Y.shape = ', X.shape, Y.shape)
        height, width = X.shape[2:4]
        assert len(X) == len(Y)

        if patch_size < height:
            X1 = []
            Y1 = []
            for n in range(len(X)):  # 裁剪并保存patch
                for i in range(0, width, patch_size):
                    for j in range(0, height, patch_size):
                        X1.append(X[n][:, j:j + patch_size, i:i + patch_size, :])
                        Y1.append(Y[n][:, j:j + patch_size, i:i + patch_size, :])
        else:
            Y1 = Y
            X1 = X
        self.nm_hr, self.nm_lr = Y1, X1
        self.lenth = len(self.nm_lr)
    
    def _scan(self, condition):
        hr = []
        lr = []
        for i in self.iso:
            self.dir_lr = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/%s/test_data/' % i
            
            lr.extend(glob.glob(self.dir_lr + 'Input/C%d/*.tif' % condition))  # [50, 520, 692]
            hr.extend(glob.glob(self.dir_lr + 'GT/C2/*.tif'))  # [692, 520]
        hr.sort()  # proj_C2_T026.tif
        lr.sort()  # C1_T026.tif  #

        self.nm_hr, self.nm_lr = hr, lr
        self.lenth = len(self.nm_lr)
    
    def __getitem__(self, idx):
        idx = idx % self.lenth
        if self.istrain:
            lr, hr, filename = self._load_file_npy(idx)
        else:
            lr, hr, filename = self._load_file_test(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()
        
        return lr, hr, filename
    
    def __len__(self):
        return self.lenth
    
    def _load_file_npy(self, idn):
        hr = self.nm_hr[idn]  # [1, 128, 128, 1]
        lr = self.nm_lr[idn]  # [50, 128, 128, 1]
        
        rgb = np.float32(np.squeeze(hr, -1))
        rgblr = np.float32(np.squeeze(lr))
        
        # print('Projection, ----> rgblr.max()', rgblr.max(), rgblr.shape)
        return rgblr, rgb, ''
    
    def _load_file_test(self, idx):
        filename, i = os.path.splitext(os.path.basename(self.nm_lr[idx]))
        
        rgblr = np.float32(imread(self.nm_lr[idx]))  # [50, 520, 692] 0~310
        rgb = np.expand_dims(np.float32(imread(self.nm_hr[idx])), 0)  # [1, 520, 692] 0~147
        
        return rgblr, rgb, filename


class FlouresceneVCD:
    def __init__(self, args, istrain=True, subtestset='to_predict'):
        self.path = '/mnt/home/user1/MCX/Medical/VCD-Net-main/vcdnet/vcd-example-data/data/'
        self.istrain = istrain
        self.args = args
        self.lf2d_base_size = args.patch_size // 11
        self.n_slices = 61
        self.n_num = 11
        self.shuffle = True
        if args.test_only and (subtestset != 'traindata'):
            self.nm_lr2d = sorted(glob.glob(self.path + '%s/*.tif' % subtestset))
            self.nm_hr3d = sorted(glob.glob('/mnt/home/user1/MCX/Medical/VCD-Net-main/vcdnet/results/VCD_tubulin/*.tif'))  #_40x_n11_[m30-30]_step1um_xl9_num120_sparse
        else:
            self.nm_hr3d = sorted(glob.glob(self.path + 'train/WF/*.tif'))
            self.nm_lr2d = sorted(glob.glob(self.path + 'train/LF/*.tif'))
            
        assert len(self.nm_hr3d) == len(self.nm_lr2d)

        self.lenth = len(self.nm_lr2d)

        if istrain:
            print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
        else:
            print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')

    def _load_dataset(self, idx):
        def rearrange3d_fn(image):
            """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
            """
        
            image = np.squeeze(image)  # remove channels dimension
            # print('reshape : ' + str(image.shape))
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
                    image = image[:, :, np.newaxis]  # uint8 0~48 (176,176,1) (649, 649,1)
                img = normalize(image)  # float64 -1~1 (176,176,1)
                img = lf_extract_fn(img, n_num=self.n_num, padding=False)  # (16, 16, 121) (59, 59, 121)
            else:
                image = imageio.volread(img_file)  # uint8 0~132  [61,176,176]
                img = normalize(image)  # float64 -1~1 (61,176,176)
                img = rearrange3d_fn(img)  # (176,176,61)
    
            img = img.astype(np.float32, casting='unsafe')
            return img
        
        training_data_lf2d = _load_imgs(self.nm_lr2d[idx], True)  # (16, 16, 121)
        X = np.transpose(training_data_lf2d, (2, 0, 1))
        if self.args.test_only:
            training_data_hr3d = _load_imgs(self.nm_hr3d[idx], False)  # (176, 176, 61)
            name = os.path.basename(self.nm_hr3d[idx])[:-4]
            Y = np.transpose(training_data_hr3d, (2, 0, 1))
        else:
            training_data_hr3d = _load_imgs(self.nm_hr3d[idx], False)  # (176, 176, 61)
            Y = np.transpose(training_data_hr3d, (2, 0, 1))
            name = ''
        return Y, X, name

    def __getitem__(self, idx):
        idx = idx % self.lenth
        hr, lr, filename = self._load_dataset(idx)
        
        lr = torch.from_numpy(np.ascontiguousarray(lr * self.args.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.args.rgb_range)).float()
        return lr, hr, filename

    def __len__(self):
        return self.lenth

    
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

