import torch
import torch.utils.data as data
torch.backends.cudnn.enabled = False
import utility
from utility import savecolorim
import loss
from mydata import normalize, PercentileNormalizer, np2Tensor, Image, glob, imread, imsave
from torch.utils.data import dataloader
import torch.nn.utils as utils
import model
import os
import numpy as np
import argparse
import torch
import contextual_loss as cl


datamin, datamax = 0, 100
gpu = torch.cuda.is_available()


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--test_only', action='store_true', default=test_only,
                        help='set this option to test the model')
    parser.add_argument('--cpu', action='store_true', default=not gpu, help='cpu only')
    parser.add_argument('--task', type=int, default=task)
    parser.add_argument('--resume', type=int, default=resume, help='-2:best;-1:latest; 0:pretrain; >0: resume')
    parser.add_argument('--pre_train', type=str, default=pretrain, help='pre-trained model directory')
    parser.add_argument('--save', type=str, default=savename, help='file name to save')
    
    # Data specifications
    parser.add_argument('--test_every', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=100, help='')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=batch, help='input batch size for training')
    parser.add_argument('--patch_size', type=int, default=Patch, help='input batch size for training')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)
    
    parser.add_argument('--load', type=str, default='', help='file name to load')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    
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
    parser.add_argument('--decay', type=str, default='200', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
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


class Sharelocdataloader(data.Dataset):
    def __init__(self, patch_size, name='', train=True, rootdatapath=''):
        self.patch_size = patch_size
        self.rgb_range = 1
        self.name = name
        self.train = train
        self.rootdatapath = rootdatapath
        self.validid = 0
        self.filenamelst = glob.glob(traindatapath + '/*/data.smlm')
        
        self.namelst = []
        self.hrlst = []
        self.lrlst = []
        self.namevallst = []
        self.hrvallst = []
        self.lrvallst = []
        if self.train:
            self.scan_s1()
            print('len(self.hr)', len(self.hrlst))
        else:
            self.scan_test_s1()
            print('len(self.hrval)', len(self.hrvallst))

    def scan_s1(self):
        patch_size = self.patch_size
    
        for filename in self.filenamelst:
            if not 'cell-5' in filename:
                path = filename[:-len('data.smlm')]
                dir = path[len(traindatapath) + 1:]
                name = datasetname + '_' + dir + '_'
                Y = imread(path + '/HR.tif')
                if widefield_LR:
                    if 'cell-7' in filename:
                        continue
                    else:
                        h, w = Y.shape
                        hc = (h - 1280) // 2 - 10
                        wc = (w - 1280) // 2 - 10
                        Y = Y[hc:hc + 1280, wc:wc + 1280]
                        X = imread(path + 'widefield.tif')
                else:
                    X = imread(path + 'LR.tif')

                # if server == 0:
                # Y = Y[:patch_size * 2, :patch_size * 2]
                # X = X[:patch_size * 2, :patch_size * 2]
            
                height, width = X.shape
                X = np.reshape(X, [height, width, 1])
                Y = np.reshape(Y, [height, width, 1])
            
                # patch size
                pid = 0
                for st in range(0, height - patch_size, patch_size // 2):
                    for stw in range(0, width - patch_size, patch_size // 2):
                        self.hrlst.append(Y[st:st + patch_size, stw:stw + patch_size, :])
                        self.lrlst.append(X[st:st + patch_size, stw:stw + patch_size, :])
                        self.namelst.append(name + 'patch%d.tif' % pid)
                        pid += 1
        print('Number of Validation Examples (PatchSize=%d): %d' % (patch_size, len(self.hrlst)))

    def scan_test_s1(self):
        patch_size = self.patch_size  # X_norm.shape[0]
        for file in self.filenamelst:
            if 'cell-5' in file:
                filename = file
        path = filename[:-len('data.smlm')]
        dir = path[len(traindatapath):]
        name = dir  # datasetname + '_' + dir + '_'  #
        Y = imread(path + '/HR.tif')
        if widefield_LR:
            h, w = Y.shape
            hc = (h - 1280) // 2 - 10
            wc = (w - 1280) // 2 - 10
            Y = Y[hc:hc+1280, wc:wc+1280]
            X = imread(path + 'widefield.tif')
        else:
            X = imread(path + 'LR.tif')
            
        height, width = X.shape
        X = np.float32(np.reshape(X, [height, width, 1]))
        Y = np.float32(np.reshape(Y, [height, width, 1]))
        
        pid = 0
        for st in range(0, height - patch_size, patch_size // 2):
            for stw in range(0, width - patch_size, patch_size // 2):
                self.hrvallst.append(Y[st:st + patch_size, stw:stw + patch_size, :])
                self.lrvallst.append(X[st:st + patch_size, stw:stw + patch_size, :])
                self.namevallst.append(name + 'patch%d.tif' % pid)
                pid += 1
        print('Number of Validation Examples (PatchSize=%d): %d' % (patch_size, len(self.hrvallst)))

    def __getitem__(self, idx):
        idx = self._get_index(idx)
        if self.train:
            lr = self.lrlst[idx]
            hr = self.hrlst[idx]
            name = self.namelst[idx]
        else:
            lr = self.lrvallst[idx]
            hr = self.hrvallst[idx]
            name = self.namevallst[idx]
        
        hr = normalize(hr, datamin, datamax, clip=True) * self.rgb_range
        lr = normalize(lr, datamin, datamax, clip=True) * self.rgb_range
        # savecolorim('hr_norm.png', hr[:, :, 0], norm=False)
        # savecolorim('hr_norm255.png', hr[:, :, 0]*255, norm=False)
        pair = (lr, hr)
        pair_t = np2Tensor(*pair)
        
        return pair_t[0], pair_t[1], name
    
    def __len__(self):
        if self.train:
            return len(self.hrlst)
        else:
            return len(self.hrvallst)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.hrlst)
        else:
            return idx


class PreTrainer():
    def __init__(self, args, my_model, my_loss, ckp):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.scale = args.scale
        self.bestpsnr = 0
        self.bestep = 0
        self.ckp = ckp
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.normalizer = PercentileNormalizer(2, 99.8)
        self.normalizerhr = PercentileNormalizer(2, 99.8)
        self.sepoch = args.resume
        self.epoch = 0
        
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        
        self.error_last = 1e8
        rp = os.path.dirname(__file__)
        if args.pre_train == '.':
            self.dir = os.path.join(rp, 'experiment', self.args.save) + '/from_scratch/'
            os.makedirs(self.dir + '/model/', exist_ok=True)
        else:
            self.dir = os.path.join(rp, 'experiment', self.args.save)
        
        os.makedirs(self.dir, exist_ok=True)
        self.tsk = task
        print('Task = ', self.tsk)
        
        print('Load data for SR')
        if not self.args.test_only:
            self.loader_train = dataloader.DataLoader(
                Sharelocdataloader(name=datasetname, train=True, patch_size=args.patch_size,
                                   rootdatapath=traindatapath),
                batch_size=args.batch_size, shuffle=False, pin_memory=not args.cpu, num_workers=0)
        self.loader_test = dataloader.DataLoader(
            Sharelocdataloader(name=datasetname, train=False, patch_size=512, rootdatapath=traindatapath),
            batch_size=1, shuffle=False, pin_memory=not args.cpu, num_workers=0)
        
        if not self.args.test_only:
            self.testsave = self.dir + '/Valid/'
            os.makedirs(self.testsave, exist_ok=True)
            self.file = open(self.testsave + "TrainPsnr.txt", 'w')
    
    def finetune(self):
        self.pslst = []
        self.sslst = []
        self.losslst = []
        
        self.loss.step()
        if self.sepoch > 0:
            epoch = self.sepoch
            self.sepoch = 0
            self.epoch = epoch
        else:
            epoch = self.epoch
        
        self.loss.start_log()
        timer_data, timer_model = utility.timer(), utility.timer()
        
        self.model.train()
        
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 1)

            criterion = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4')
            losscv = criterion(torch.concat([sr, sr, sr], 1), torch.concat([hr, hr, hr], 1))  # 3.7493

            l1 = self.loss(sr, hr)  # 0.4334
            loss = l1 + 1000 * losscv
            
            loss.backward()
            self.losslst.append(loss.detach().cpu().numpy())
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()
            timer_model.hold()
            if batch % self.args.print_every == 0:
                print('Batch%d/Epoch%d, Loss = ' % (batch, epoch), loss.detach().cpu().numpy(),
                      l1.detach().cpu().numpy(), losscv.detach().cpu().numpy())
                print('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format((batch + 1) * self.args.batch_size,
                                                           len(self.loader_train.dataset),
                                                           self.loss.display_loss(batch), timer_model.release(),
                                                           timer_data.release()))
            
            if batch % self.args.test_every == 0:
                self.loss.end_log(len(self.loader_train))
                self.error_last = self.loss.log[-1, -1]
                self.optimizer.schedule()
                psnr, ssim = self.testvalidsr(epoch)
                self.pslst.append(psnr)
                self.sslst.append(ssim)
                
                self.model.train()
                self.loss.step()
                print('Evaluation -- Batch%d/Epoch%d' % (batch, epoch))
                self.loss.start_log()
            
            timer_data.tic()
        
        self.file.write('\n PSNR \n' + str(self.pslst) + '\n SSIM \n' + str(self.sslst))
        self.file.flush()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        self.model.save(self.dir + '/model/', epoch, is_best=False)
        print('save model Epoch%d' % epoch)
    
    # # -------------------------- SR --------------------------
    def testvalidsr(self, epoch=0, woft=False):
        if self.args.test_only:
            if woft:
                self.testsave = pretrain[:-len('model_best.pt')-1] + '/results/%s/' % datasetname
            else:
                self.testsave = self.dir + '/results/model%d/' % self.args.resume
            os.makedirs(self.testsave, exist_ok=True)
        torch.set_grad_enabled(False)
        
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        num = 0
        pslst = []
        sslst = []
        psbiclst = []
        ssbiclst = []
        nmlst = []
        for idx_data, (lr, hr, filename) in enumerate(self.loader_test):
            filename = filename[0]
            nmlst.append(filename)
            num += 1
            if dataset == 'Shareloc':
                os.makedirs(self.testsave + filename[:filename.rfind('patch')], exist_ok=True)
            if not self.args.test_only and num > 2:
                break
            lr, hr = self.prepare(lr, hr)
            
            sr_model = self.model(lr, 1)
            
            sr = utility.quantize(sr_model, self.args.rgb_range)
            hr = utility.quantize(hr, self.args.rgb_range)
            print('hr.shape = sr.shape =', hr.shape, sr.shape)
            
            ps, ss = utility.compute_psnr_and_ssim(
                sr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :],
                hr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :])
            pslst.append(ps)
            sslst.append(ss)
            print(ps, ss)
            
            name = '{}'.format(filename[:-4])
            sr = np.squeeze(sr.cpu().detach().numpy())
            imsave(self.testsave + name + '_SR.tif', sr)  # imageio.imwrite(self.testsave + name + '_SR.tif', sr)
            savecolorim(self.testsave + name + '-Color.png', sr * 255, norm=False)

            if self.args.test_only:
                lr1 = np.float32(np.squeeze(lr.detach().cpu().numpy()))  # 512*512  0~1
                hr1 = np.squeeze(hr.cpu().detach().numpy())
                hr = np.float32(normalize(hr1, datamin, datamax, clip=True)) * 255
                sr = np.float32(normalize(sr, datamin, datamax, clip=True)) * 255
                savecolorim(self.testsave + name + '-MeandfnoNormC.png', sr - hr, norm=False)

                lr1 = np.float32(normalize(lr1, datamin, datamax, clip=True)) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(lr1, hr)
                print('bicubic pnsr/ssim = ', psm, ssmm)
                psbiclst.append(psm)
                ssbiclst.append(ssmm)
                savecolorim(self.testsave + name + '_LRcolor.png', lr1 * 255, norm=False)
                savecolorim(self.testsave + name + '-ColorHR.png', hr1 * 255, norm=False)
                savecolorim(self.testsave + name + '-MeandfnoNormC-Bic.png', lr1 - hr, norm=False)
                
        psnrall = np.mean(np.array(pslst))
        ssimall = np.mean(np.array(sslst))
        if self.args.test_only:
            print('Mean of bicubic pnsr/ssim = ', np.mean(np.array(psbiclst)), np.mean(np.array(ssbiclst)))
            file = open(self.testsave + "result.txt", 'w')
            file.write('Mean of bicubic pnsr/ssim = ' + str(np.mean(np.array(psbiclst))) + str(np.mean(np.array(ssbiclst))))
            file.write('\nName \n' + str(nmlst) + '\n PSNR \n' + str(pslst) + '\n SSIM \n' + str(sslst))
            file.close()
        else:
            self.loader_train.train = True
            if psnrall > self.bestpsnr:
                self.bestpsnr = psnrall
                self.bestep = epoch
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
        
        print('num', num, psnrall, ssimall)
        print('bestpsnr/epoch = ', self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrall, ssimall
    
    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        
        return [_prepare(a) for a in args]
    
    def terminate(self):
        if self.args.test_only:
            return False
        else:
            self.epoch = self.epoch + 1
            if self.epoch > self.args.epochs:
                self.file.close()
            return self.epoch <= self.args.epochs


if __name__ == '__main__':
    task = 1
    test_only =  True  #  False  #
    dataset = 'Shareloc'
    Patch = 128
    batch = 8
    
    datasetname = 'Shareloc'
    widefield_LR = True  # False  #
    
    resume = 0
    pretrain = 'experiment/Uni-SwinIRShareloc/WF/model/model_2.pt'
    
    if widefield_LR:
        savename = 'Uni-SwinIR%s/WF/' % datasetname
    else:
        savename = 'Uni-SwinIR%s/' % datasetname
    
    traindatapath = '/home/user2/dataset/microscope/%s/' % datasetname
    
    args = options()
    torch.manual_seed(args.seed)
    
    checkpoint = utility.checkpoint(args)
    assert checkpoint.ok
    unimodel = model.UniModel(args, tsk=task, srscale=1)
    
    _model = model.Model(args, checkpoint, unimodel)
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = PreTrainer(args, _model, _loss, checkpoint)
    
    if test_only:
        t.testvalidsr(0, woft=False)
    else:
        while t.terminate():
            t.finetune()
    
    checkpoint.done()
