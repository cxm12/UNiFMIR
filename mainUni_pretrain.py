import torch
torch.backends.cudnn.enabled = False
import utility
from utility import savecolorim
import loss
import argparse
from mydata import SR, FlouresceneVCD, Flouresceneproj, Flouresceneiso, Flourescenedenoise, \
    normalize, PercentileNormalizer
from torch.utils.data import dataloader
import torch.nn.utils as utils
import model
import os
from decimal import Decimal
import imageio
import numpy as np
from tifffile import imsave
import random


gpu = torch.cuda.is_available()


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--task', type=int, default=-1)
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--save', type=str, default='Uni-SwinIR-pretrain', help='file name to save')
    parser.add_argument('--test_only', action='store_true', default=testonly, help='set this option to test the model')
    parser.add_argument('--cpu', action='store_true', default=not gpu, help='cpu only')
    parser.add_argument('--resume', type=int, default=0, help='-2:best;-1:latest; 0:pretrain; >0: resume')
    parser.add_argument('--pre_train', type=str, default=pretrain, help='pre-trained model directory')
    
    # Data specifications
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--patch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)

    parser.add_argument('--print_every', type=int, default=200, help='')
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--load', type=str, default='', help='file name to load')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    
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
        self.epoch_tsk5 = 0
        self.epoch_tsk4 = 0
        self.test_only = args.test_only
        
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        
        self.error_last = 1e8
        rp = os.path.dirname(__file__)
        self.dir = os.path.join(rp, 'experiment', self.args.save)
        os.makedirs(self.dir, exist_ok=True)
        if not self.args.test_only:
            self.testsave = self.dir + '/Valid/'
            os.makedirs(self.testsave, exist_ok=True)
            self.file = open(self.testsave + "TrainPsnr.txt", 'w')
    
    def pretrain(self):
        self.pslst = []
        self.sslst = []
        
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
        self.changeTask(t=args.task, subd=-1)
        
        if self.tsk == 4: self.epoch_tsk4 += 1
        if self.tsk == 5: self.epoch_tsk5 += 1
        
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            if (self.tsk == 1) or (self.tsk == 2) or (self.tsk == 3):
                sr = self.model(lr, self.tsk)
                loss = self.loss(sr, hr)
            elif self.tsk == 4:
                sr_stg1, sr = self.model(lr, self.tsk)
                if self.epoch_tsk4 <= 30:
                    loss = 0.001 * self.loss(sr_stg1, hr) + self.loss(sr, hr)
                else:
                    loss = self.loss(sr, hr)
            elif self.tsk == 5:
                sr_stg1, sr = self.model(lr, self.tsk)
                if self.epoch_tsk5 <= 30:
                    loss = self.loss(sr_stg1, hr)
                else:
                    loss = 0.1 * self.loss(sr_stg1, hr) + self.loss(sr, hr)
            
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()
            timer_model.hold()
            if batch % self.args.print_every == 0:
                sr2dim = np.float32(normalize(np.squeeze(sr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                hr2dim = np.float32(normalize(np.squeeze(hr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print('training patch- PSNR/SSIM = %f/%f' % (psm, ssmm))
                
                if self.tsk == 4 or self.tsk == 5:
                    sr2dimu = np.float32(
                        normalize(np.squeeze(sr_stg1[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(sr2dimu, hr2dim)
                    print('sr_stg1 training patch = %f/%f' % (psm, ssmm))
                
                print('Batch%d/Epoch%d, Loss = ' % (batch, epoch), loss)
                print('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format((batch + 1) * self.args.batch_size,
                                                           len(self.loader_train.dataset),
                                                           self.loss.display_loss(batch),
                                                           timer_model.release(),
                                                           timer_data.release()))
            
            timer_data.tic()
            
            if batch % self.args.test_every == 0:
                self.loss.end_log(len(self.loader_train))
                self.error_last = self.loss.log[-1, -1]
                self.optimizer.schedule()
                if self.tsk == 1:
                    psnr, ssim = self.testSR(epoch)
                elif self.tsk == 2:
                    psnr, ssim = self.test3Ddenoise(epoch, condition=1)
                elif self.tsk == 3:
                    psnr, ssim = self.testiso(epoch)
                elif self.tsk == 4:
                    psnr, ssim = self.testproj(epoch)
                elif self.tsk == 5:
                    psnr, ssim = self.test2to3(epoch)
                
                self.pslst.append(psnr)
                self.sslst.append(ssim)
                
                self.model.train()
                self.loss.step()
                lr = self.optimizer.get_lr()
                print('Evaluation -- Batch%d/Epoch%d' % (batch, epoch))
                self.ckp.write_log('Batch%d/Epoch%d' % (batch, epoch) +
                                   '\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
                self.loss.start_log()
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        
        self.file.write('Name \n PSNR \n' + str(self.pslst) + '\n SSIM \n' + str(self.sslst))
        self.model.save(self.dir + '/model/', epoch, is_best=False)
        self.model.scale = 1
        print('save model Epoch%d' % epoch, loss)
    
    def testall(self, tsk, subd=-1, condition=1):
        datasetname = self.changeTask(tsk, subd, condition=condition)
        
        if tsk == 1:
            p, s = self.testSR()
        elif tsk == 2:
            p, s = self.test3Ddenoise(condition=condition, data_test=datasetname)
        elif tsk == 3:
            p, s = self.testiso()
        elif tsk == 4:
            p, s = self.testproj(condition=condition)
        elif tsk == 5:
            p, s = self.test2to3()
        return p, s
    
    # # -------------------------- SR --------------------------
    def testSR(self, epoch=0):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/' % self.args.resume
        os.makedirs(self.testsave, exist_ok=True)
        self.model.scale = 2
        
        torch.set_grad_enabled(False)
        
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lr, hr, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            if not self.args.test_only and num >= 5:
                break
            num += 1
            lr, hr = self.prepare(lr, hr)  # torch.tensor(random).float().
            sr = self.model(lr, 0)
            sr = utility.quantize(sr, self.args.rgb_range)
            hr = utility.quantize(hr, self.args.rgb_range)
            
            pst = utility.calc_psnr(sr, hr, self.scale[0], self.args.rgb_range, dataset=None)
            sr = sr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            hr = hr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            ps, ss = utility.compute_psnr_and_ssim(sr, hr)
            sr255 = np.float32(normalize(sr, 0, 100, clip=True)) * 255
            hr255 = np.float32(normalize(hr, 0, 100, clip=True)) * 255
            ps255, ss255 = utility.compute_psnr_and_ssim(sr255, hr255)
            
            pslst.append(np.max([ps, pst, ps255]))
            sslst.append(ss255)
            print('pst, ps, ss, ps255, ss255 = ', pst, ps, ss, ps255, ss255)
            if self.args.test_only:
                name = '{}.png'.format(filename[0][:-4])
                imageio.imwrite(self.testsave + name, sr)
                savecolorim(self.testsave + name[:-4] + '-Color.png', sr, norm=False)
                
                sr = np.round(np.maximum(0, np.minimum(255, sr)))
                hr2 = np.round(np.maximum(0, np.minimum(255, hr)))
                res = np.clip(np.abs(sr - hr2), 0, 255)
                savecolorim(self.testsave + name[:-4] + '-MeandfnoNormC.png', res, norm=False)
        
        psnrmean = np.mean(pslst)
        ssimmean = np.mean(sslst)
        
        if self.args.test_only:
            file = open(self.testsave + "Psnrssim100_norm.txt", 'w')
            file.write('Mean = ' + str(psnrmean) + str(ssimmean))
            file.write('\nName \n' + str(nmlst) + '\n PSNR t \n'
                       + '\n PSNR max \n' + str(pslst)
                       + '\n SSIM \n' + str(sslst))
            file.close()
        else:
            if psnrmean > self.bestpsnr:
                self.bestpsnr = psnrmean
                self.bestep = epoch
                self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        
        print('num = ', num, 'psnrmean SSIM = ', psnrmean, ssimmean)
        torch.set_grad_enabled(True)
        return psnrmean, ssimmean
    
    # # -------------------------- 3D denoise --------------------------
    def test3Ddenoise(self, epoch=0, condition=1, data_test='Denoising_Tribolium'):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/condition_%d/' % (self.args.resume, condition)
            os.makedirs(self.testsave, exist_ok=True)
            file = open(self.testsave + '/Psnrssim_Im_patch_c%d.txt' % condition, 'w')
        
        datamin, datamax = self.args.datamin, self.args.datamax
        patchsize = 600
        torch.set_grad_enabled(False)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.args.test_only and num >= 1:
                break
            num += 1
            
            nmlst.append(filename)
            print('filename = ', filename)
            if filename[0] == '':
                name = 'im%d' % idx_data
            else:
                name = '{}'.format(filename[0])
            if not self.args.test_only:
                name = 'EP{}_{}'.format(epoch, filename[0])
            
            # 1.3D norm 2 998
            lrt = self.normalizer.before(lrt, 'CZYX')  # [0~806] -> [0~1.]
            hrt = self.normalizerhr.before(hrt, 'CZYX')  # [0~806] -> [0~1.]
            lrt, hrt = self.prepare(lrt, hrt)
            
            lr = np.squeeze(lrt.cpu().detach().numpy())
            hr = np.squeeze(hrt.cpu().detach().numpy())
            print('hr.shape = ', hr.shape)
            denoiseim = torch.zeros_like(hrt, dtype=hrt.dtype)
            
            batchstep = 5  # 10  #
            inputlst = []
            for ch in range(0, len(hr)):  # [45, 486, 954]  0~44
                if ch < 5 // 2:  # 0, 1
                    lr1 = [lrt[:, ch:ch + 1, :, :] for _ in range(5 // 2 - ch)]
                    lr1.append(lrt[:, :5 // 2 + ch + 1])
                    lrt1 = torch.concat(lr1, 1)  # [B, inputchannel, h, w]
                elif ch >= (len(hr) - 5 // 2):  # 43, 44
                    lr1 = []
                    lr1.append(lrt[:, ch - 5 // 2:])
                    numa = (5 // 2 - (len(hr) - ch)) + 1
                    lr1.extend([lrt[:, ch:ch + 1, :, :] for _ in range(numa)])
                    lrt1 = torch.concat(lr1, 1)  # [B, inputchannel, h, w]
                else:
                    lrt1 = lrt[:, ch - 5 // 2:ch + 5 // 2 + 1]
                assert lrt1.shape[1] == 5
                inputlst.append(lrt1)
            
            for dp in range(0, len(inputlst), batchstep):
                if dp + batchstep >= len(hr):
                    dp = len(hr) - batchstep
                print(dp)  # 0, 10, .., 90
                lrtn = torch.concat(inputlst[dp:dp + batchstep], 0)  # [batch, inputchannel, h, w]
                a = self.model(lrtn, 2)
                a = torch.transpose(a, 1, 0)  # [1, batch, h, w]
                denoiseim[:, dp:dp + batchstep, :, :] = a
            
            sr = np.float32(denoiseim.cpu().detach().numpy())
            # 3.3D norm 2 998 tiff save；
            sr = np.squeeze(self.normalizer.after(sr))
            hr = np.squeeze(self.normalizerhr.after(hr))
            
            # 4.3D norm0100 psnr
            sr255 = np.squeeze(np.float32(normalize(sr, datamin, datamax, clip=True))) * 255  # [0, 1]
            hr255 = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255  # [0, 1]
            lr255 = np.float32(normalize(lr, datamin, datamax, clip=True)) * 255  # [0, 1]
            
            cpsnrlst = []
            cssimlst = []
            step = 1
            if self.args.test_only:
                imsave(self.testsave + name + '.tif', sr)
                if 'Planaria' in data_test:
                    if condition == 1:
                        randcs = 10
                        randce = hr.shape[0] - 10
                        step = (hr.shape[0] - 20) // 5
                    else:
                        randcs = 85
                        randce = 87
                        step = 1
                        if randce >= hr.shape[0]:
                            randcs = hr.shape[0] - 3
                            randce = hr.shape[0]
                    
                    for dp in range(randcs, randce, step):
                        savecolorim(self.testsave + name + '-dfnoNormC%d.png' % dp, sr[dp] - hr[dp], norm=False)
                        savecolorim(self.testsave + name + '-C%d.png' % dp, sr[dp])
                        savecolorim(self.testsave + name + '-HRC%d.png' % dp, hr[dp])
                        srpatch255 = sr255[dp, :patchsize, :patchsize]
                        hrpatch255 = hr255[dp, :patchsize, :patchsize]
                        lrpatch255 = lr255[dp, :patchsize, :patchsize]
                        
                        ##  PSNR/SSIM
                        psm, ssmm = utility.compute_psnr_and_ssim(srpatch255, hrpatch255)
                        psml, ssmml = utility.compute_psnr_and_ssim(lrpatch255, hrpatch255)
                        print('SR Image %s - C%d- PSNR/SSIM/MSE = %f/%f' % (name, dp, psm, ssmm))  # /%f, mse
                        print('LR PSNR/SSIM = %f/%f' % (psml, ssmml))
                        cpsnrlst.append(psm)
                        cssimlst.append(ssmm)
                elif 'Tribolium' in data_test:
                    if condition == 1:
                        randcs = 2
                        randce = hr.shape[0] - 2
                        step = (hr.shape[0] - 4) // 6
                    else:
                        randcs = hr.shape[0] // 2 - 1
                        randce = randcs + 3
                        step = 1
                    for randc in range(randcs, randce, step):
                        hrpatch = normalize(hr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        srpatchour = normalize(sr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        psm, ssmm = utility.compute_psnr_and_ssim(srpatchour, hrpatch)
                        cpsnrlst.append(psm)
                        cssimlst.append(ssmm)
                file.write('Image \"%s\" Channel = %d-%d \n' % (name, randcs, randce) + 'PSNR = ' + str(
                    cpsnrlst) + '\n SSIM = ' + str(cssimlst))
            else:
                randcs = 0
                randce = hr.shape[0]
                for randc in range(randcs, randce, step):
                    hrpatch = normalize(hr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                    srpatchour = normalize(sr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(srpatchour, hrpatch)
                    cpsnrlst.append(psm)
                    cssimlst.append(ssmm)
            
            psnr1, ssim = np.mean(np.array(cpsnrlst)), np.mean(np.array(cssimlst))
            print('SR im:', psnr1, ssim)
            sslst.append(ssim)
            pslst.append(psnr1)
        
        psnrm = np.mean(np.array(pslst))
        ssimm = np.mean(np.array(sslst))
        if self.args.test_only:
            print('+++++++++ condition%d meanSR++++++++++++' % condition, sum(pslst) / len(pslst),
                  sum(sslst) / len(sslst))
            file.write('\n \n +++++++++ condition%d meanSR ++++++++++++ \n PSNR/SSIM \n  patchsize = %d \n' % (
                condition, patchsize))
            file.write('Name \n' + str(nmlst) + '\n PSNR = ' + str(pslst) + '\n SSIM = ' + str(sslst))
            file.close()
        else:
            if psnrm > self.bestpsnr:
                self.bestpsnr = psnrm
                self.bestep = epoch
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        print('psnrm, np.mean(np.array(sslst)) = ', psnrm, ssimm)
        torch.set_grad_enabled(True)
        return psnrm, ssimm
    
    # # -------------------------- Isotropic Reconstruction --------------------------
    def testiso(self, epoch=0):
        def _rotate(arr, k=1, axis=1, copy=True):
            """Rotate by 90 degrees around the first 2 axes."""
            if copy:
                arr = arr.copy()
            k = k % 4
            arr = np.rollaxis(arr, axis, arr.ndim)
            if k == 0:
                res = arr
            elif k == 1:
                res = arr[::-1].swapaxes(0, 1)
            elif k == 2:
                res = arr[::-1, ::-1]
            else:
                res = arr.swapaxes(0, 1)[::-1]
            
            res = np.rollaxis(res, -1, axis)
            return res
        
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/' % self.args.resume
            os.makedirs(self.testsave, exist_ok=True)
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)
        if epoch == None: epoch = self.optimizer.get_last_epoch()
        
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        num = 0
        pslst = []
        pslstall = []
        sslst = []
        sslstall = []
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            num += 1
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            
            lrt = self.normalizer.before(lrt, 'CZYX')
            hrt = self.normalizerhr.before(hrt, 'CZYX')  # [0~4095] -> [0~2.9262]
            lrt, hrt = self.prepare(lrt, hrt)  # [B, 301, 752, 752]
            
            lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))
            
            if len(lr.shape) <= 3:
                lr = np.expand_dims(lr, -1)
                hr = np.expand_dims(hr, -1)
            isoim1 = np.zeros_like(hr, dtype=np.float32)  # [301, 752, 752, 2]
            isoim2 = np.zeros_like(hr, dtype=np.float32)  # [301, 752, 752, 2]
            
            print('filename = ', filename, 'hr.shape, lr.shape = ', hr.shape, lr.shape)  # [301, 752, 752, 2]
            
            batchstep = 100
            for wp in range(0, hr.shape[2], batchstep):
                if wp + batchstep >= hr.shape[2]:
                    wp = hr.shape[2] - batchstep
                x_rot1 = _rotate(lr[:, :, wp:wp + batchstep, :], axis=1, copy=False)
                x_rot1 = np.expand_dims(np.squeeze(x_rot1), 1)
                
                x_rot1 = torch.from_numpy(np.ascontiguousarray(x_rot1)).float()
                x_rot1 = self.prepare(x_rot1)[0]
                a1 = self.model(x_rot1, 0)
                
                a1 = np.expand_dims(np.squeeze(a1.cpu().detach().numpy()), -1)
                u1 = _rotate(a1, -1, axis=1, copy=False)
                isoim1[:, :, wp:wp + batchstep, :] = u1
            for hp in range(0, hr.shape[1], batchstep):
                if hp + batchstep >= hr.shape[1]:
                    hp = hr.shape[1] - batchstep
                
                x_rot2 = _rotate(_rotate(lr[:, hp:hp + batchstep, :, :], axis=2, copy=False), axis=0, copy=False)
                x_rot2 = np.expand_dims(np.squeeze(x_rot2), 1)
                
                x_rot2 = torch.from_numpy(np.ascontiguousarray(x_rot2)).float()
                a2 = self.model(self.prepare(x_rot2)[0], 0)
                
                a2 = np.expand_dims(np.squeeze(a2.cpu().detach().numpy()), -1)
                u2 = _rotate(_rotate(a2, -1, axis=0, copy=False), -1, axis=2, copy=False)
                isoim2[:, hp:hp + batchstep, :, :] = u2
            
            sr = np.sqrt(np.maximum(isoim1, 0) * np.maximum(isoim2, 0))
            print('sr.shape = ', sr.shape)
            
            sr = np.squeeze(self.normalizer.after(sr))
            lr = np.squeeze(self.normalizer.after(lr))
            imsave(self.testsave + name + '.tif', sr)
            
            hr = np.squeeze(self.normalizerhr.after(hr))
            c, h, w = hr.shape
            
            cpsnrlst = []
            cssimlst = []
            for dp in range(1, h, h // 5):
                if self.args.test_only:
                    savecolorim(self.testsave + name + '-dfnoNormCz%d.png' % dp, sr[:, dp, :] - hr[:, dp, :],
                                norm=False)
                    savecolorim(self.testsave + name + '-C%d.png' % dp, sr[:, dp, :])
                    savecolorim(self.testsave + name + '-GTC%d.png' % dp, hr[:, dp, :])
                    savecolorim(self.testsave + name + '-LRC%d.png' % dp, lr[:, dp, :])
                
                # 5.2D norm0100 psnr
                hrpatch = normalize(hr[:, dp, :], datamin, datamax, clip=True) * 255
                print('hrpatch.shape ', hrpatch.shape)
                lrpatch = normalize(lr[:, dp, :], datamin, datamax, clip=True) * 255
                srpatch = normalize(sr[:, dp, :], datamin, datamax, clip=True) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(srpatch, hrpatch)
                psml, ssmml = utility.compute_psnr_and_ssim(lrpatch, hrpatch)
                print('Normalized Patch %s - C%d- PSNR/SSIM/MSE = %f/%f' % (name, dp, psm, ssmm))
                print('Normalized LR PSNR/SSIM = %f/%f' % (psml, ssmml))
                
                cpsnrlst.append(psm)
                cssimlst.append(ssmm)
            psnr1, ssim = np.mean(np.array(cpsnrlst)), np.mean(np.array(cssimlst))
            print('SR im:', psnr1, ssim)
            sslst.append(ssim)
            pslst.append(psnr1)
            sslstall.append(cssimlst)
            pslstall.append(cpsnrlst)
        
        psnrm = np.mean(np.array(pslst))
        ssmm = np.mean(np.array(sslst))
        print('psnr, num, psnrall1, ssimall = ', psnrm, ssmm, num)
        
        if self.args.test_only:
            file = open(self.testsave + "Psnrssim.txt", 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(pslst) + '\n SSIM \n' + str(sslst))
            file.write(
                'PSNR of 2D image in different depth of each image \n PSNR \n' + str(pslstall) + '\n SSIM \n' + str(
                    sslstall))
            file.close()
        else:
            if psnrm > self.bestpsnr:
                self.bestpsnr = psnrm
                self.bestep = epoch
                self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrm, ssmm
    
    # # -------------------------- Projection --------------------------
    def testproj(self, epoch=0, condition=2):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/c%d/' % (self.args.resume, condition)
            os.makedirs(self.testsave, exist_ok=True)
        print('save to', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnrall, ssimall = [], []
        psnralls1, ssimalls1 = [], []
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.args.test_only and num >= 1:
                break
            num += 1
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            
            # 1.3D norm 2 998
            lrt, hrt = self.prepare(lrt, hrt)
            
            a_stg1, a = self.model(lrt, 4)  # [1, 1, h, w]
            
            sr_stg1 = np.float32(np.squeeze(a_stg1.cpu().detach().numpy()))
            sr = np.float32(np.squeeze(a.cpu().detach().numpy()))
            
            # 3D norm 2 998 tiff save
            srtf = sr
            if self.args.test_only:
                axes_restored = 'YX'
                utility.save_tiff_imagej_compatible(self.testsave + name + '.tif', srtf, axes_restored)
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))  # [1, 1, h, w]
            
            ##  PSNR/SSIM
            # 2.(2D norm 0100 PSnr color save)
            hr2dim = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255  # [0, 1]
            sr2dim = np.float32(normalize(np.float32(srtf), datamin, datamax, clip=True)) * 255  # norm_srtf
            sr2dim_stg1 = np.float32(normalize(np.float32(sr_stg1), datamin, datamax, clip=True)) * 255  # norm_srtf
            psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
            psm_stg1, ssmm_stg1 = utility.compute_psnr_and_ssim(sr2dim_stg1, hr2dim)
            print('2D img Norm-%s - PSNR/SSIM = %f/%f / Output of StageI = %f/%f' % (
                name, psm, ssmm, psm_stg1, ssmm_stg1))
            
            psnralls1.append(psm_stg1)
            ssimalls1.append(ssmm_stg1)
            psnrall.append(psm_stg1)
            ssimall.append(ssmm_stg1)
        
        psnrallm = np.mean(np.array(psnrall))
        ssimallm = np.mean(np.array(ssimall))
        psnrallms1 = np.mean(np.array(psnralls1))
        ssimallms1 = np.mean(np.array(ssimalls1))
        
        if self.args.test_only:
            file = open(self.testsave + "Psnrssim.txt_c%d.txt" % condition, 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(psnrall) + '\n SSIM \n' + str(ssimall))
            file.close()
        else:
            if psnrallm > self.bestpsnr:
                self.bestpsnr = psnrallm
                self.bestep = epoch
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        
        print('+++++++++ condition %d StageI/II ++++++++++++' % condition, psnrallm, ssimallm, psnrallms1, ssimallms1)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrallm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrallm, ssimallm
    
    # # -------------------------- 2D to 3D --------------------------
    def test2to3(self, epoch=0, subtestset='to_predict'):
        if self.args.test_only:
            self.testsave = self.dir + 'results/model_%d/%s/' % (self.args.resume, subtestset)
            os.makedirs(self.testsave, exist_ok=True)
            print('make save path', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnralls1 = []
        psnrall = []
        ssimalls1 = []
        ssimall = []
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.args.test_only and num >= 2:
                break
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            if name == '':
                name = 'im%d' % idx_data
            print('image %s ' % (name + '.tif'))
            lrt, hrt = self.prepare(lrt, hrt)  # [1, 121, h//11, w//11]
            
            as1, a = self.model(lrt, 5)
            
            sr = np.float32(a.cpu().detach().numpy())
            srs1 = np.float32(as1.cpu().detach().numpy())
            
            # [1,649,649,61]
            imsave(self.testsave + name + 'norm.tif', np.squeeze(sr))
            print('Save TIF image \' %s \' ' % (self.testsave + name + '.tif'))
            
            sr = (np.clip(np.squeeze(sr), -1, 1) + 1) / 2
            srs1 = (np.clip(np.squeeze(srs1), -1, 1) + 1) / 2
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))  # [61, h, w]
            hr = (np.clip(hr, -1, 1) + 1) / 2
            lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))  # [121, h//11, w//11]
            lr = (np.clip(lr, -1, 1) + 1) / 2
            
            if self.args.test_only:
                c, h, w = hr.shape
                if h == sr.shape[1]:
                    savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr[0])
                    wf2d = np.zeros([h, w])
                    d = 0
                    for i in range(11):
                        for j in range(11):
                            wf2d[i: h: 11, j: w: 11] = lr[d, :, :]
                            d += 1
                    savecolorim(self.testsave + 'OriIm' + name + '-LR.png', wf2d)
                    
                    ##  PSNR/SSIM
                    for i in range(0, len(hr), 10):
                        savecolorim(self.testsave + 'OriIm' + name + '-Result%d.png' % i, sr[i])
                        num += 1
                        hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255  # [0, 1]
                        sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                        sr2dims1 = np.float32(normalize(srs1[i], datamin, datamax, clip=True)) * 255
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                        psms1, ssmms1 = utility.compute_psnr_and_ssim(sr2dims1, hr2dim)
                        psnrall.append(psm)
                        ssimall.append(ssmm)
                        psnralls1.append(psms1)
                        ssimalls1.append(ssmms1)
                        print('I%d, 2D img Norm-%s - PSNR/SSIM/MSE = %f/%f StageI  %f/%f' % (
                            i, name, psm, ssmm, psms1, ssmms1))
                else:
                    h, w = h * 11, w * 11
                    print('hr.shape = ', (h, w, 61))
                    wf2d = np.zeros([h, w])
                    d = 0
                    for i in range(11):
                        for j in range(11):
                            wf2d[i: h: 11, j: w: 11] = hr[d, :, :]
                            d += 1
                    savecolorim(self.testsave + 'OriIm' + name + '-LR.png', wf2d)
                    for i in range(0, len(sr), 10):
                        savecolorim(self.testsave + 'OriIm' + name + '.png', sr[i])
                        lr2dim = np.float32(normalize(wf2d, datamin, datamax, clip=True)) * 255  # [0, 1]
                        sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                        sr2dims1 = np.float32(normalize(srs1[i], datamin, datamax, clip=True)) * 255
                        num += 1
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, lr2dim)
                        psms1, ssmms1 = utility.compute_psnr_and_ssim(sr2dims1, lr2dim)
                        psnrall.append(psm)
                        ssimall.append(ssmm)
                        psnralls1.append(psms1)
                        ssimalls1.append(ssmms1)
                        print('I%d, 2D img Norm-%s - PSNR/SSIM/MSE = %f/%f StageI  %f/%f' % (
                            i, name, psm, ssmm, psms1, ssmms1))
            else:
                savecolorim(self.testsave + 'OriIm' + name + '-Result.png', sr[0])
                savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr[0])
                # #  PSNR/SSIM
                for i in range(0, len(hr), 10):
                    num += 1
                    hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255  # [0, 1]
                    sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                    sr2dims1 = np.float32(normalize(srs1[i], datamin, datamax, clip=True)) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                    psms1, ssmms1 = utility.compute_psnr_and_ssim(sr2dims1, hr2dim)
                    psms1 = np.max([0, np.min([100, psms1])])
                    psnralls1.append(psms1)
                    ssimalls1.append(ssmms1)
                    psnrall.append(psms1)
                    ssimall.append(ssmms1)
                    
                    print('Stage I 2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psms1, ssmms1))
                    print('Stage II 2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psm, ssmm))
        psnrmeans1 = np.mean(psnralls1)
        ssmeans1 = np.mean(ssimalls1)
        psnrmeans2 = np.mean(psnrall)
        ssmeans2 = np.mean(ssimall)
        psnrmean = psnrmeans2
        ssmean = ssmeans2
        
        if psnrmean > self.bestpsnr:
            self.bestpsnr = psnrmean
            self.bestep = epoch
        if not self.args.test_only:
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch))
        print('+++++++++ StageI/II ++++++++++++', psnrmeans1, ssmeans1, psnrmeans2, ssmeans2)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrmean, self.bestpsnr, self.bestep)
        
        torch.set_grad_enabled(True)
        return psnrmean, ssmean
    
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
    
    def changeTask(self, t=-1, subd=-1, condition=1):
        if t == -1:
            self.tsk = random.randint(1, 5)  # 5  #
        else:
            self.tsk = t
        
        if self.tsk == 1:
            taskname = ['SR', 'Denoising', 'Isotropic', 'Projection', 'Volume']
            print('Change Task to ', taskname[self.tsk - 1])
            self.model.scale = 2
            print('Load data for SR')
            srlst = ['F-actin', 'ER', 'Microtubules', 'CCPs']
            if subd == -1:
                testset = srlst[random.randint(0, 3)]
            else:
                testset = srlst[subd]
            if not testonly:
                self.loader_train = dataloader.DataLoader(
                    SR(scale=2, name=testset, train=True, rootdatapath=srdatapath
                       , patch_size=args.patch_size, length=20),
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=not args.cpu,
                    num_workers=0)
            self.loader_test = [dataloader.DataLoader(
                SR(scale=2, name=testset, train=False, test_only=args.test_only,
                   rootdatapath=srdatapath, patch_size=args.patch_size, length=20),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
        elif self.tsk == 2:
            nlst = ['Denoising_Planaria', 'Denoising_Tribolium', 'Synthetic_tubulin_granules', 'Synthetic_tubulin_gfp']
            if subd == -1:
                testset = nlst[random.randint(0, 1)]
            else:
                testset = nlst[subd]
            if not testonly:
                self.loader_train = dataloader.DataLoader(
                    Flourescenedenoise(name=testset, istrain=True, c=condition, rootdatapath=denoisedatapath,
                                       patch_size=args.patch_size, length=2000),
                    batch_size=args.batch_size, shuffle=True, pin_memory=not args.cpu, num_workers=0)
            self.loader_test = [dataloader.DataLoader(
                Flourescenedenoise(name=testset, istrain=False, c=condition, rootdatapath=denoisedatapath,
                                   test_only=args.test_only, patch_size=args.patch_size, length=2000),
                batch_size=1, shuffle=False, pin_memory=not args.cpu, num_workers=0)]
        elif self.tsk == 3:
            # isotropic
            testset = 'Isotropic_Liver'
            self.loader_test = [dataloader.DataLoader(
                Flouresceneiso(name=testset, istrain=False, rootdatapath=isodatapath, patch_size=args.patch_size,
                               test_only=args.test_only, length=2000),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
            if not testonly:
                self.loader_train = dataloader.DataLoader(
                    Flouresceneiso(name=testset, istrain=True, rootdatapath=isodatapath, patch_size=args.patch_size,
                                   length=2000),
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=not args.cpu,
                    num_workers=0)
        elif self.tsk == 4:
            # projection
            testset = 'Projection_Flywing'
            self.loader_test = [dataloader.DataLoader(
                Flouresceneproj(name=testset, istrain=False, condition=condition, test_only=args.test_only,
                                rootdatapath=prodatapath, patch_size=args.patch_size, length=2000),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
            if not testonly:
                self.loader_train = dataloader.DataLoader(
                    Flouresceneproj(name=testset, istrain=True, condition=condition,
                                    rootdatapath=prodatapath, patch_size=args.patch_size, length=2000),
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=not args.cpu,
                    num_workers=0)
        elif self.tsk == 5:
            print('Load data for volumetric reconstruction')
            # 2D to 3D
            testset = 'to_predict'
            self.loader_test = [dataloader.DataLoader(
                FlouresceneVCD(istrain=False, subtestset=testset, test_only=args.test_only,
                               rootdatapath=voldatapath, patch_size=args.patch_size, length=2000),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
            if not testonly:
                self.loader_train = dataloader.DataLoader(
                    FlouresceneVCD(istrain=True, subtestset=testset, test_only=False,
                                   rootdatapath=voldatapath,
                                   patch_size=args.patch_size, length=2000),
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=0)
        
        return testset


if __name__ == '__main__':
    srdatapath = '/home/user2/dataset/microscope/CSB/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/'
    denoisedatapath = '/home/user2/dataset/microscope/CSB/DataSet/'
    isodatapath = '/home/user2/dataset/microscope/CSB/DataSet/Isotropic/'
    prodatapath = '/home/user2/dataset/microscope/CSB/DataSet/'
    voldatapath = '/home/user2/dataset/microscope/VCD/vcdnet/'
    
    pretrain = '.'
    testonly = False  # True  #
    
    args = options()
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    assert checkpoint.ok
    unimodel = model.UniModel(args, tsk=-1)

    _model = model.Model(args, checkpoint, unimodel)

    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = PreTrainer(args, _model, _loss, checkpoint)
    
    if testonly:
        for i in range(0, 4):
            t.testall(tsk=1, subd=i)
    else:
        while t.terminate():
            t.pretrain()
    
    checkpoint.done()
