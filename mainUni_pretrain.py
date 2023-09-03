import torch
torch.backends.cudnn.enabled = False
import utility
from utility import savecolorim
import loss
import argparse
import template
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


def options():
    parser = argparse.ArgumentParser(description='FMIR Model')
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--test_only', action='store_true', default=False, help='set this option to test the model')
    parser.add_argument('--cpu', action='store_true', default=False, help='cpu only')
    parser.add_argument('--task', type=int, default=-1)
    parser.add_argument('--resume', type=int, default=20, help='-2:best;-1:latest; 0:pretrain; >0: resume')
    parser.add_argument('--pre_train', type=str, default='.', help='pre-trained model directory')
    parser.add_argument('--save', type=str, default='Uni-SwinIR', help='file name to save')
    
    # Data specifications
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
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
    template.set_template(args)
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    
    return args


def Pretrain():
    _model = model.Model(args, checkpoint, unimodel)
    print('Total params: %.4fM' % (sum(p.numel() for p in _model.parameters()) / 1000000.0))
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = PreTrainer(args, _model, _loss, checkpoint)
    
    while t.terminate():
        t.pretrain()
    
    checkpoint.done()


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
        self.test_only = False
        
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        rp = os.path.dirname(__file__)
        self.dir = os.path.join(rp, 'experiment', self.args.save)
        os.makedirs(self.dir, exist_ok=True)
    
    def pretrain(self):
        self.loss.step()
        if self.sepoch > 0:
            epoch = self.sepoch
            self.sepoch = 0
        else:
            epoch = self.epoch

        lr = self.optimizer.get_lr()
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        timer_data, timer_model = utility.timer(), utility.timer()
        
        self.model.train()
        comparative_loss = None
        tsk = random.randint(1, 5)
        print('Task = ', tsk)
        if tsk == 1:
            self.model.scale = 2
            print('Load data for SR')
            testset = 'F-actin'
            self.loader_train = dataloader.DataLoader(
                SR(scale=2, name=testset, train=True, benchmark=False,
                   rootdatapath=srdatapath),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=0)
            self.loader_test = [dataloader.DataLoader(
                SR(scale=2, name=testset, train=False, benchmark=False,
                   rootdatapath=srdatapath),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
        elif tsk == 2:
            condition = 1
            testset = 'Denoising_Tribolium'
            self.loader_train = dataloader.DataLoader(
                Flourescenedenoise(name=testset, istrain=True, c=condition,
                                   rootdatapath=denoisedatapath),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=0)
            self.loader_test = [dataloader.DataLoader(
                Flourescenedenoise(name=testset, istrain=False, c=condition,
                                   rootdatapath=denoisedatapath),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
        elif tsk == 3:
            # isotropic
            testset = 'Isotropic_Liver'
            self.loader_test = [dataloader.DataLoader(
                Flouresceneiso(name=testset, istrain=False, rootdatapath=isodatapath),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
            self.loader_train = dataloader.DataLoader(
                Flouresceneiso(name=testset, istrain=True, rootdatapath=isodatapath),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=0)
        elif tsk == 4:
            # projection
            condition = 1
            testset = 'Projection_Flywing'
            self.loader_test = [dataloader.DataLoader(
                Flouresceneproj(name=testset, istrain=False, condition=condition,
                                rootdatapath=prodatapath),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
            self.loader_train = dataloader.DataLoader(
                Flouresceneproj(name=testset, istrain=True, condition=condition,
                                rootdatapath=prodatapath),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=0)
        elif tsk == 5:
            print('Load data for volumetric reconstruction')
            # 2D to 3D
            subtestset = 'to_predict'
            self.loader_test = [dataloader.DataLoader(
                FlouresceneVCD(istrain=False, subtestset=subtestset, test_only=False,
                               rootdatapath=voldatapath),
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)]
            self.loader_train = dataloader.DataLoader(
                FlouresceneVCD(istrain=True, subtestset=subtestset, test_only=False,
                               rootdatapath=voldatapath),
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=0)

        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            if (tsk == 1) or (tsk == 2) or (tsk == 3):
                sr = self.model(lr, tsk)
                loss = self.loss(sr, hr)
            elif tsk == 4:
                sr_stg1, sr, comparative_loss = self.model(lr, tsk)
                loss = 0.001 * self.loss(sr_stg1, hr) + self.loss(sr, hr)
                comparative_loss = torch.mean(sum(comparative_loss) / len(comparative_loss))
                if epoch > 100:
                    loss += 0.001 * comparative_loss
            elif tsk == 5:
                # 2D to 3D:
                sr_stg1, sr = self.model(lr, tsk)
                if epoch > 100:
                    for param in self.model.model.conv_first0.parameters():
                        param.requires_grad = False
                    loss = self.loss(sr, hr)
                else:
                    loss = self.loss(sr_stg1, hr)
            
            if comparative_loss is None:
                comparative_loss = torch.zeros_like(loss)
            
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()
            timer_model.hold()
            if batch % self.args.print_every == 0:
                sr2dim = np.float32(normalize(np.squeeze(sr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                hr2dim = np.float32(normalize(np.squeeze(hr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                mse = np.sum(np.power(sr2dim - hr2dim, 2))
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print('training patch- PSNR/SSIM/MSE = %f/%f/%f' % (psm, ssmm, mse))
                
                if tsk == 4 or tsk == 5:
                    sr2dimu = np.float32(
                        normalize(np.squeeze(sr_stg1[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(sr2dimu, hr2dim)
                    print('sr_stg1 training patch = %f/%f' % (psm, ssmm))
                
                print('Batch%d/Epoch%d, Loss = ' % (batch, epoch), loss)
                print('[{}/{}]\t{}/comparative_loss:{}\t{:.1f}+{:.1f}s'.format((batch + 1) * self.args.batch_size,
                                                                               len(self.loader_train.dataset),
                                                                               self.loss.display_loss(batch),
                                                                               comparative_loss.item(),
                                                                               timer_model.release(),
                                                                               timer_data.release()))
            
            timer_data.tic()
            
            if batch % self.args.test_every == 0:
                self.loss.end_log(len(self.loader_train))
                self.error_last = self.loss.log[-1, -1]
                self.optimizer.schedule()
                if tsk == 1:
                    self.testsr(batch, epoch)
                elif tsk == 2:
                    self.test3Ddenoise(batch, epoch)
                elif tsk == 3:
                    self.testiso(batch, epoch)
                elif tsk == 4:
                    self.testproj(batch, epoch)
                elif tsk == 5:
                    self.test2to3(batch, epoch)
                
                self.model.train()
                self.loss.step()
                lr = self.optimizer.get_lr()
                print('Evaluation -- Batch%d/Epoch%d' % (batch, epoch))
                self.ckp.write_log(
                    'Batch%d/Epoch%d' % (batch, epoch) + '\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
                self.loss.start_log()
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        
        self.model.save(self.dir + '/model/', epoch, is_best=False)
        self.model.scale = 1
        print('save model Epoch%d' % epoch, loss)
    
    def testall(self, tsk):
        if tsk == 1:
            self.testsr()
        elif tsk == 2:
            self.test3Ddenoise()
        elif tsk == 3:
            self.testiso()
        elif tsk == 4:
            self.testproj()
        elif tsk == 5:
            self.test2to3()
    
    # # -------------------------- SR --------------------------
    def testsr(self, batch=0, epoch=None, datasetname='CCPs'):
        if self.test_only:
            self.testsave = self.dir + '/results/model%d/' % self.args.resume
        else:
            self.testsave = self.dir + '/Validresults-{}/'.format(datasetname)
        os.makedirs(self.testsave, exist_ok=True)
        
        torch.set_grad_enabled(False)
        
        self.ckp.write_log('\nEvaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnrall1, ssimall = 0, 0
        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lr, hr, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            if not self.test_only and num >= 5:
                break
            num += 1
            lr, hr = self.prepare(lr, hr)

            sr = self.model(lr, 1)
            sr = utility.quantize(sr, self.args.rgb_range)
            hr = utility.quantize(hr, self.args.rgb_range)
            print('hr.shape = ', hr.shape)
            print('sr.shape = ', sr.shape)

            ps, ss = utility.compute_psnr_and_ssim(
                sr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :],
                hr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :])
            pslst.append(ps)
            sslst.append(ss)
            psnrall1 += ps
            ssimall += ss
            print(ps, ss)
            
            if os.path.exists(self.dir):
                name = '{}.png'.format(filename[0][:-4])
                normalized = sr.mul(255 / self.args.rgb_range).byte().cpu()
                sr = np.squeeze(normalized.detach().numpy())
                imageio.imwrite(self.testsave + name, sr)
                savecolorim(self.testsave + name[:-4] + '-Color.png', sr, norm=False)
                
                normalizedhr = hr[0].mul(255 / self.args.rgb_range).byte().cpu()
                hr1 = np.squeeze(normalizedhr.detach().numpy())
                savecolorim(self.testsave + name[:-4] + '-ColorHR.png', hr1, norm=False)
                
                sr = np.round(np.maximum(0, np.minimum(255, sr)))
                hr2 = np.round(np.maximum(0, np.minimum(255, hr1)))
                res = np.clip(np.abs(sr - hr2), 0, 255)
                savecolorim(self.testsave + name[:-4] + '-MeandfnoNormC.png', res, norm=False)
        
        psnrall1 = psnrall1 / num
        if self.test_only:
            file = open(self.testsave + "result.txt", 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(pslst) + '\n SSIM \n' + str(sslst))
            file.close()
        else:
            if psnrall1 > self.bestpsnr:
                self.bestpsnr = psnrall1
                self.bestep = epoch
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
        
        print('num', num, psnrall1 / num, ssimall / num)
        print('bestpsnr/epoch = ', self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
    
    # # -------------------------- 3D denoise --------------------------
    def test3Ddenoise(self, batch=0, epoch=None, condition=1, data_test=''):
        if self.test_only:
            file = open(self.testsave + '/Psnrssim_Im_patch_c%d.txt' % condition, 'w')
            self.testsave = self.dir + '/results/model%d/condition_%d/' % (self.args.resume, self.args.condition)
        else:
            self.testsave = self.dir + '/Valid/'
        os.makedirs(self.testsave, exist_ok=True)
        
        datamin, datamax = self.args.datamin, self.args.datamax
        patchsize = 600
        
        torch.set_grad_enabled(False)
        
        self.ckp.write_log('\nEvaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnrall, psnrall1, ssimall = 0, 0, 0
        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            num += 1
            nmlst.append(filename)
            print('filename = ', filename)
            if filename[0] == '':
                name = 'im%d' % idx_data
            else:
                name = '{}'.format(filename[0])
            if not self.test_only:
                name = 'EP{}_{}'.format(epoch, filename[0])
                if num >= 3:
                    break
            
            
            # 1.3D norm 2 998
            lrt = self.normalizer.before(lrt, 'CZYX')
            hrt = self.normalizerhr.before(hrt, 'CZYX')
            lrt, hrt = self.prepare(lrt, hrt)
            
            lr = np.squeeze(lrt.cpu().detach().numpy())
            hr = np.squeeze(hrt.cpu().detach().numpy())
            print('hr.shape = ', hr.shape)
            denoiseim = torch.zeros_like(hrt, dtype=hrt.dtype)
            
            batchstep = 10
            inputlst = []
            for ch in range(0, len(hr)):
                if ch < 5 // 2:
                    lr1 = [lrt[:, ch:ch + 1, :, :] for _ in range(5 // 2 - ch)]
                    lr1.append(lrt[:, :5 // 2 + ch + 1])
                    lrt1 = torch.concat(lr1, 1)
                elif ch >= (len(hr) - 5 // 2):
                    lr1 = []
                    lr1.append(lrt[:, ch - 5 // 2:])
                    numa = (5 // 2 - (len(hr) - ch)) + 1
                    lr1.extend([lrt[:, ch:ch + 1, :, :] for _ in range(numa)])
                    lrt1 = torch.concat(lr1, 1)
                else:
                    lrt1 = lrt[:, ch - 5 // 2:ch + 5 // 2 + 1]
                assert lrt1.shape[1] == 5
                inputlst.append(lrt1)
            
            for dp in range(0, len(inputlst), batchstep):
                if dp + batchstep >= len(hr):
                    dp = len(hr) - batchstep
                print(dp)
                lrtn = torch.concat(inputlst[dp:dp + batchstep], 0)
                a = self.model(lrtn, 2)
                a = torch.transpose(a, 1, 0)
                denoiseim[:, dp:dp + batchstep, :, :] = a
            
            sr = np.float32(denoiseim.cpu().detach().numpy())
            # 3.3D norm 2 998 tiff save；
            sr = np.squeeze(self.normalizer.after(sr))
            hr = np.squeeze(self.normalizerhr.after(hr))
            
            # 4.3D norm 0 100 psnr
            sr255 = np.squeeze(np.float32(normalize(sr, datamin, datamax, clip=True))) * 255
            hr255 = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255
            lr255 = np.float32(normalize(lr, datamin, datamax, clip=True)) * 255
            
            cpsnrlst = []
            cssimlst = []
            randh = randw = 0
            step = 1
            if self.test_only:
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
                elif 'Tribolium' in data_test:
                    randcs = hr.shape[0] // 2 - 1
                    randce = randcs + 3
            else:
                randcs = 0
                randce = hr.shape[0]
            
            for dp in range(randcs, randce, step):
                savecolorim(self.testsave + name + '-dfnoNormC%d.png' % dp, sr[dp] - hr[dp], norm=False)
                savecolorim(self.testsave + name + '-C%d.png' % dp, sr[dp])
                savecolorim(self.testsave + name + '-HRC%d.png' % dp, hr[dp])
                
                srpatch255 = sr255[dp, randh:randh + patchsize, randw:randw + patchsize]
                hrpatch255 = hr255[dp, randh:randh + patchsize, randw:randw + patchsize]
                lrpatch255 = lr255[dp, randh:randh + patchsize, randw:randw + patchsize]
                
                ##  PSNR/SSIM
                mse = np.sum(np.power(srpatch255 - hrpatch255, 2))
                psm, ssmm = utility.compute_psnr_and_ssim(srpatch255, hrpatch255)
                psml, ssmml = utility.compute_psnr_and_ssim(lrpatch255, hrpatch255)
                print('Normalized Image %s - C%d- PSNR/SSIM/MSE = %f/%f/%f' % (name, dp, psm, ssmm, mse))
                print('Normalized LR PSNR/SSIM = %f/%f' % (psml, ssmml))
                cpsnrlst.append(psm)
                cssimlst.append(ssmm)
            
            if self.test_only:
                file.write('Image \"%s\" Channel = %d-%d \n' % (name, randcs, randce) + 'PSNR = ' + str(
                    cpsnrlst) + '\n SSIM = ' + str(cssimlst))
            
            psnr1, ssim = np.mean(np.array(cpsnrlst)), np.mean(np.array(cssimlst))
            print('SR im:', psnr1, ssim)
            sslst.append(ssim)
            pslst.append(psnr1)
        
        psnrm = np.mean(np.array(pslst))
        print('psnr, num, psnrall1, ssimall = ', psnrm, num, psnrall1 / num, ssimall / num)
        
        if self.test_only:
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
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
    
    # # -------------------------- Isotropic Reconstruction --------------------------
    def testiso(self, batch=0, epoch=None, data_test='Isotropic_Liver'):
        self.testsave = self.dir + '/results/model%d/' % self.args.resume
        os.makedirs(self.testsave, exist_ok=True)
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)
        
        self.ckp.write_log('\n Evaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnrall, psnrall1, ssimall = 0, 0, 0
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
            if not self.test_only and num >= 5:
                break
                
            # 1.3D norm 2 998
            lrt = self.normalizer.before(lrt, 'CZYX')
            hrt = self.normalizerhr.before(hrt, 'CZYX')
            lrt, hrt = self.prepare(lrt, hrt)
            isoim = torch.zeros_like(hrt, dtype=hrt.dtype)
            
            lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))
            print('filename = ', filename, 'hr.shape, lr.shape = ', hr.shape, lr.shape)  # [301, 752, 752]
            
            batchstep = 10
            for dp in range(0, len(hr), batchstep):
                if dp + batchstep >= len(hr):
                    dp = len(hr) - batchstep
                
                lrtn = torch.transpose(lrt[:, dp:dp + batchstep, :, :], 1, 0)
                a = self.model(lrtn, 3)
                isoim[:, dp:dp + batchstep, :, :] = torch.transpose(a, 1, 0)
                
                sr = np.float32(np.squeeze(a.cpu().detach().numpy()))
                savecolorim(self.testsave + 'OriIm' + name + '-C%dLR.png' % dp, lr[dp])
                savecolorim(self.testsave + 'OriIm' + name + '-C%dHR.png' % dp, hr[dp])
                savecolorim(self.testsave + 'OriIm' + name + '-C%d.png' % dp, sr[0])
                
                ##  PSNR/SSIM
                # 3.(2D norm 0100 PSnr color save)
                hr2dim = np.float32(normalize(hr[dp], datamin, datamax, clip=True)) * 255
                sr2dim = np.float32(normalize(sr[0], datamin, datamax, clip=True)) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print('2D img Norm-%s - C%d- PSNR/SSIM/MSE = %f/%f' % (name, dp, psm, ssmm))
            
            sr = np.float32(isoim.cpu().detach().numpy())
            print('sr.shape = ', sr.shape)  #
            # 4.3D norm 2 998 tiff save
            sr = np.squeeze(self.normalizer.after(sr))
            lr = np.squeeze(self.normalizer.after(lr))
            imsave(self.testsave + name + '.tif', sr)
            hr = np.squeeze(self.normalizerhr.after(hr))
            
            if 'Retina' in data_test:
                # Retina: [35, 2, 1024, 1024]
                # Liver [301, 752, 752]
                # Drosophila (108, 1352, 532)
                hr = hr[:, 0, :, :]
                lr = lr[:, 0, :, :]
            
            c, h, w = hr.shape
            print('hr.shape = ', hr.shape)
            savenum = 5  #
            
            cpsnrlst = []
            cssimlst = []
            for dp in range(1, h, h // savenum):
                savecolorim(self.testsave + name + '-dfnoNormCz%d.png' % dp, sr[:, dp, :] - hr[:, dp, :], norm=False)
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
        print('psnr, num, psnrall1, ssimall = ', psnrm, ssmm, num, psnrall1 / num, ssimall / num)
        
        if self.test_only:
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
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
        
        print('+++++++++ meanSR++++++++++++', sum(pslst) / len(pslst), sum(sslst) / len(sslst))
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
    
    # # -------------------------- 3D to 2D projection --------------------------
    def testproj(self, batch=0, epoch=None, condition=0):
        if self.test_only:
            self.testsave = self.dir + '/results/model%d/NoNormc%d_C2GT/' % (self.args.resume, condition)
        else:
            self.testsave = self.dir + '/valid_C2GT/'
        os.makedirs(self.testsave, exist_ok=True)
        print('save to', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)

        self.ckp.write_log('\n Evaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnrall, ssimall = [], []
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            num += 1
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            if not self.test_only and num >= 5:
                break
                
            # 1.3D norm 2 998
            lrt, hrt = self.prepare(lrt, hrt)
            
            a = self.model(lrt, 4)
            sr = np.float32(np.squeeze(a.cpu().detach().numpy()))
            
            # 3D norm 2 998 tiff save
            srtf = np.squeeze(sr)
            axes_restored = 'YX'
            utility.save_tiff_imagej_compatible(self.testsave + name + '.tif', srtf, axes_restored)
            
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))
            
            ##  PSNR/SSIM
            # 2.(2D norm 0100 color save)
            hr2dim = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255
            sr2dim = np.float32(normalize(np.float32(srtf), datamin, datamax, clip=True)) * 255
            psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
            print('2D img Norm - [Stage II output] - %s - PSNR/SSIM = %f/%f' % (name, psm, ssmm))

            psnrall.append(psm)
            ssimall.append(ssmm)
        psnrallm = np.mean(np.array(psnrall))
        ssimallm = np.mean(np.array(ssimall))
        if self.test_only:
            file = open(self.testsave + "Psnrssim.txt_c%d.txt" % condition, 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(psnrall) + '\n SSIM \n' + str(ssimall))
            file.close()
        else:
            if psnrallm > self.bestpsnr:
                self.bestpsnr = psnrallm
                self.bestep = epoch
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
        
        print('+++++++++ meanSR condition %d++++++++++++' % condition, psnrallm, ssimallm)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrallm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
    
    # # -------------------------- 2D to 3D --------------------------
    def test2to3(self, batch=0, epoch=None, subtestset='to_predict'):
        def write3d(x, path, bitdepth=16, norm_max=True):
            """
            x : [batch, depth, height, width, channels] or [batch, height, width, channels>3]
            """
            
            def _write3d(x, path, bitdepth=16, norm_max=True):
                """
                x : [depth, height, width, channels=1]
                """
                assert (bitdepth in [8, 16, 32])
                
                if bitdepth == 32:
                    x = x.astype(np.float32)
                else:
                    if norm_max:
                        x = (x + 1) / 2
                    
                    x[:, :16, :16, :], x[:, -16:, -16:, :] = 0, 0  # suppress the corners
                    x[:, -16:, :16, :], x[:, :16, -16:, :] = 0, 0
                    
                    if bitdepth == 8:
                        x = x * 255
                        x = x.astype(np.uint8)
                    else:
                        x = x * 65535
                        x = x.astype(np.uint16)
                
                imageio.volwrite(path, x[..., 0])
            
            # print(x.shape)  # (1, 176, 176, 61)
            dims = len(x.shape)
            
            if dims == 4:
                batch, height, width, n_channels = x.shape
                x_re = np.zeros([batch, n_channels, height, width, 1])
                for d in range(n_channels):
                    slice = x[:, :, :, d]
                    x_re[:, d, :, :, :] = slice[:, :, :, np.newaxis]
            elif dims == 5:
                x_re = x
            else:
                raise Exception('unsupported dims : %s' % str(x.shape))
            
            batch = x_re.shape[0]
            if batch == 1:
                _write3d(x_re[0], path, bitdepth, norm_max)
            else:
                fragments = path.split('.')
                new_path = ''
                for i in range(len(fragments) - 1):
                    new_path = new_path + fragments[i]
                for index, image in enumerate(x_re):
                    print(image.shape)
                    _write3d(image, new_path + '_' + str(index) + '.' + fragments[-1], bitdepth, norm_max)
        
        self.testsave = self.dir + '/results/model_%d/%s/' % (self.args.resume, subtestset)
        os.makedirs(self.testsave, exist_ok=True)
        print('make save path', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)
        
        self.ckp.write_log('\n Evaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnrall, ssimall = 0, 0
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            if name == '':
                name = 'im%d' % idx_data
            # print('image %s ' % (name + '.tif'))
            num += 1
            if not self.test_only and num >= 5:
                break
                
            lrt, hrt = self.prepare(lrt, hrt)
            
            a = self.model(lrt, 5)
            
            sr = np.float32(a.cpu().detach().numpy())
            write3d(np.transpose(sr, [0, 2, 3, 1]), self.testsave + name + '.tif', bitdepth=16, norm_max=True)
            print('Save TIF image \' %s \' ' % (self.testsave + name + '.tif'))
            
            sr = (np.clip(np.squeeze(sr), -1, 1) + 1) / 2
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))
            hr = (np.clip(hr, -1, 1) + 1) / 2
            lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))
            lr = (np.clip(lr, -1, 1) + 1) / 2
            
            if self.test_only:
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
                        hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255
                        sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                        psnrall += psm
                        ssimall += ssmm
                        print('2D img Norm-%s - PSNR/SSIM/MSE = %f/%f' % (name, psm, ssmm))
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
                        lr2dim = np.float32(normalize(wf2d, datamin, datamax, clip=True)) * 255
                        sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                        num += 1
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, lr2dim)
                        print('I%d, 2D img Norm-%s - PSNR/SSIM/MSE = %f/%f' % (i, name, psm, ssmm))
            else:
                savecolorim(self.testsave + 'OriIm' + name + '-Result.png', sr[0])
                savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr[0])
                
                ##  PSNR/SSIM
                for i in range(0, len(hr), 10):
                    num += 1
                    hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255
                    sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                    psnrall += psm
                    ssimall += ssmm
                    print('2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psm, ssmm))
        if psnrall > 0:
            psnrall = psnrall / num
            ssimall = ssimall / num
            if psnrall > self.bestpsnr:
                self.bestpsnr = psnrall
                self.bestep = epoch
            if not self.test_only:
                self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
            print('+++++++++ meanSR++++++++++++', psnrall, ssimall)
            print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrall, self.bestpsnr, self.bestep)
        
        torch.set_grad_enabled(True)
    
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
            return self.epoch <= self.args.epochs


if __name__ == '__main__':
    srdatapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/'
    denoisedatapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/'
    isodatapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/Isotropic/'
    prodatapath = '/mnt/home/user1/MCX/Medical/CSBDeep-master/DataSet/'
    voldatapath = '/mnt/home/user1/MCX/Medical/VCD-Net-main/vcdnet/'
    
    args = options()
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    assert checkpoint.ok
    unimodel = model.UniModel(args, tsk=-1)
    Pretrain()
