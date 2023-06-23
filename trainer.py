import os
from decimal import Decimal
import utility
import torch
import torch.nn.utils as utils
import imageio
from utility import savecolorim
import numpy as np
from data import normalize, IS_TF_1, PercentileNormalizer
from tifffile import imsave



class Trainer():
    def __init__(self, args, loader_train, loader_test, datasetname, my_model, my_loss, ckp):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.scale = args.scale
        self.datasetname = datasetname
        self.bestpsnr = 0
        self.bestep = 0
        self.ckp = ckp
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.normalizer = PercentileNormalizer(2, 99.8)  # 逼近npz
        self.normalizerhr = PercentileNormalizer(2, 99.8)
        self.sepoch = args.resume

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        rp = os.path.dirname(__file__)
        self.dir = os.path.join(rp, 'experiment', self.args.save)
        os.makedirs(self.dir, exist_ok=True)
        
    def trainUni(self, tsk=1):
        self.loss.step()
        if self.sepoch > 0:
            epoch = self.sepoch
            self.sepoch = 0
        else:
            epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
    
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        timer_data, timer_model = utility.timer(), utility.timer()
    
        self.model.train()
        comparative_loss = None
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
        
            self.optimizer.zero_grad()
            if (tsk == 1) or (tsk == 2) or (tsk == 3):
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)
            elif tsk == 4:
                sr_stg1, sr, comparative_loss = self.model(lr, 0)
                loss = 0.001 * self.loss(sr_stg1, hr) + self.loss(sr, hr)
                # if epoch <= 50:
                #     loss = self.loss(srgvt, hr)
                # else:
                #     for param in self.model.model.project.parameters():
                #         param.requires_grad = False
                #     loss = 0.001 * self.loss(srgvt, hr) + self.loss(sr, hr)
                comparative_loss = torch.mean(sum(comparative_loss) / len(comparative_loss))
                if self.optimizer.get_last_epoch() > 100:
                    loss += 0.001 * comparative_loss
            elif tsk == 5:
                # 2D to 3D:
                sr_stg1, sr = self.model(lr, 0)
                for param in self.model.model.conv_first0.parameters():
                    param.requires_grad = False
                loss = self.loss(sr, hr)
                
            if comparative_loss is None:
                comparative_loss = torch.zeros_like(loss)
                    
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(), self.args.gclip)
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
                    self.test(batch, epoch)
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
    
        print('save model Epoch%d' % epoch, loss)
        self.model.save(self.dir + '/model/', epoch, is_best=False)

    # # -------------------------- SR --------------------------
    def test(self, batch=0, epoch=None):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/' % self.args.resume
        else:
            self.testsave = self.dir + '/Validresults-{}/'.format(self.datasetname)
        os.makedirs(self.testsave, exist_ok=True)
        
        torch.set_grad_enabled(False)
        if epoch == None:
            epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        psnrall, psnrall1, ssimall = 0, 0, 0
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
            if ('2stg_enlcn' in self.args.model) and (not self.args.test_only):
                srstage1, sr, _ = self.model(lr, 0)  # [1, 1, h, w]
            else:
                sr = self.model(lr, 0)
            sr = utility.quantize(sr, self.args.rgb_range)
            hr = utility.quantize(hr, self.args.rgb_range)
    
            pst = utility.calc_psnr(sr, hr, self.scale[0], self.args.rgb_range, dataset=None)
            psnrall += pst
            # print('sr.shape', sr.shape)
            ps, ss = utility.compute_psnr_and_ssim(
                sr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :],
                hr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0,0,:,:])
            pslst.append(pst)
            sslst.append(ss)
            psnrall1 += ps
            ssimall += ss
            print(ps, ss, psnrall)
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
                # res1 = (normalized - normalizedhr).detach().numpy()
                # res = np.squeeze(np.clip(np.abs(res1), 0, 255))
                savecolorim(self.testsave + name[:-4] + '-MeandfnoNormC.png', res, norm=False)
                # print('Save to', name)
                
        psnrall = psnrall/num
        if self.args.test_only:
            file = open(self.testsave + "Psnrssim100.txt", 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(pslst) + '\n SSIM \n' + str(sslst))
            file.close()
        else:
            if psnrall > self.bestpsnr:
                self.bestpsnr = psnrall
                self.bestep = epoch
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
            # self.ckp.save(self, epoch, is_best=(self.bestep == epoch))
            
        print('num', num, psnrall1/num, ssimall/num)
        print('psnrm, self.bestpsnr, self.bestep', psnrall, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)

    # # -------------------------- 3D denoise --------------------------
    def test3Ddenoise(self, batch=0, epoch=None, condition=1):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/condition_%d/' % (self.args.resume, self.args.condition)
        else:
            self.testsave = self.dir + '/Valid/'
        os.makedirs(self.testsave, exist_ok=True)
    
        datamin, datamax = self.args.datamin, self.args.datamax
        patchsize = 600
        if self.args.test_only:
            file = open(self.testsave + '/Psnrssim_Im_patch_c%d.txt' % condition, 'w')
    
        torch.set_grad_enabled(False)
        if epoch == None:
            epoch = self.optimizer.get_last_epoch()
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
            if not self.args.test_only:
                name = 'EP{}_{}'.format(epoch, filename[0])
        
            if not self.args.test_only and num >= 3:
                break
        
            # 1.3D norm 2 998
            lrt = self.normalizer.before(lrt, 'CZYX')  # [0~806] -> [0~1.]
            hrt = self.normalizerhr.before(hrt, 'CZYX')  # [0~806] -> [0~1.]
            lrt, hrt = self.prepare(lrt, hrt)
            # print('lrt.max.min = 1~0', lrt.max(), lrt.min())
        
            lr = np.squeeze(lrt.cpu().detach().numpy())
            hr = np.squeeze(hrt.cpu().detach().numpy())
            # print('hr.max(), hr.min() = 1.0 0.0 ', hr.max(), hr.min())  #
            print('hr.shape = ', hr.shape)
            denoiseim = torch.zeros_like(hrt, dtype=hrt.dtype)
        
            batchstep = 10  # 5  #
            inputlst = []
            for ch in range(0, len(hr)):  # [45, 486, 954]  0~44
                # print(ch)
                if ch < self.args.inputchannel//2:  # 0, 1
                    lr1 = [lrt[:, ch:ch+1, :, :] for _ in range(self.args.inputchannel//2-ch)]
                    lr1.append(lrt[:, :self.args.inputchannel//2+ch+1])
                    lrt1 = torch.concat(lr1, 1)  # [B, inputchannel, h, w]
                elif ch >= (len(hr) - self.args.inputchannel//2):  # 43, 44
                    lr1 = []
                    lr1.append(lrt[:, ch-self.args.inputchannel // 2:])
                    numa = (self.args.inputchannel // 2 - (len(hr) - ch)) + 1
                    lr1.extend([lrt[:, ch:ch+1, :, :] for _ in range(numa)])
                    lrt1 = torch.concat(lr1, 1)  # [B, inputchannel, h, w]
                else:
                    lrt1 = lrt[:, ch-self.args.inputchannel // 2:ch + self.args.inputchannel // 2 + 1]
                assert lrt1.shape[1] == self.args.inputchannel
                inputlst.append(lrt1)
                    
            for dp in range(0, len(inputlst), batchstep):
                if dp + batchstep >= len(hr):
                    dp = len(hr) - batchstep
                print(dp)  # 0, 10, .., 90
                lrtn = torch.concat(inputlst[dp:dp + batchstep], 0)  # [batch, inputchannel, h, w]
                a = self.model(lrtn, 0)
                a = torch.transpose(a, 1, 0)  # [1, batch, h, w]
                denoiseim[:, dp:dp + batchstep, :, :] = a
            
                # # 2.(2D norm 0100 PSnr color save)
                # hr2dim = np.float32(normalize(hr[dp], datamin, datamax, clip=True)) * 255  # [0, 1]
                # lr2dim = np.float32(normalize(lr[dp], datamin, datamax, clip=True)) * 255  # [0, 1]
                # sr2dim = np.float32(
                #     normalize(np.squeeze(a)[0].detach().cpu().numpy(), datamin, datamax, clip=True)) * 255
            
                # savecolorim(self.testsave + name + '-Im255C%dLR.png' % dp, lr2dim)
                # savecolorim(self.testsave + name + '-Im255C%dHR.png' % dp, hr2dim)
                # savecolorim(self.testsave + name + '-Im255C%d.png' % dp, sr2dim)
            
                # ##  PSNR/SSIM
                # mse = np.sum(np.power(sr2dim - hr2dim, 2))
                # psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                # print('2D img Norm255 -%s - C%d- PSNR/SSIM/MSE = %f/%f/%f' % (name, dp, psm, ssmm, mse))
        
            sr = np.float32(denoiseim.cpu().detach().numpy())
            # 3.3D norm 2 998 tiff save；
            sr = np.squeeze(self.normalizer.after(sr))
            hr = np.squeeze(self.normalizerhr.after(hr))
        
            # 4.3D norm0100 psnr
            sr255 = np.squeeze(np.float32(normalize(sr, datamin, datamax, clip=True))) * 255  # [0, 1]
            hr255 = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255  # [0, 1]
            lr255 = np.float32(normalize(lr, datamin, datamax, clip=True)) * 255  # [0, 1]
            # print('sr.shape  denoiseim[0].shape = ', sr.shape, denoiseim[0].shape)  # [1, 2, 100, 100] [1, 1, 100, 100]
            # print('sr3dnorm.max() .min() = ', sr.max(), sr.min())  # 520.6648 312.37643
            # P [95,1024,1024] uint16 [0~33000] # T [45, 486, 954]
        
            cpsnrlst = []
            cssimlst = []
            randh = randw = 0  # a[i][1], a[i][2]  # np.random.randint(0, w-patchsize)
            step = 1
            if self.args.test_only:
                imsave(self.testsave + name + '.tif', sr)
                if 'Planaria' in self.args.data_test:
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
                elif 'Tribolium' in self.args.data_test:
                    randcs = hr.shape[0] // 2 - 1
                    randce = randcs + 3
            else:
                randcs = 0
                randce = hr.shape[0]
        
            for dp in range(randcs, randce, step):
                savecolorim(self.testsave + name + '-dfnoNormC%d.png' % dp, sr[dp] - hr[dp], norm=False)
                savecolorim(self.testsave + name + '-C%d.png' % dp, sr[dp])
                savecolorim(self.testsave + name + '-HRC%d.png' % dp, hr[dp])
            
                # srpatch = sr[dp, randh:randh + patchsize, randw:randw + patchsize]
                # print('srpatch.max() .min() = ', srpatch.max(), srpatch.min())  # 520.6648 312.37643
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
            
                # savecolorim(self.testsave + name + '-dfnoNormC255-%d.png' % dp, srpatch255 - hrpatch255, norm=False)
                # savecolorim(self.testsave + name + '-C255-%d.png' % dp, srpatch255)
                # savecolorim(self.testsave + name + '-HRC255-%d.png' % dp, hrpatch255)
        
            if self.args.test_only:
                file.write('Image \"%s\" Channel = %d-%d \n' % (name, randcs, randce) + 'PSNR = ' + str(
                    cpsnrlst) + '\n SSIM = ' + str(cssimlst))
        
            psnr1, ssim = np.mean(np.array(cpsnrlst)), np.mean(np.array(cssimlst))
            print('SR im:', psnr1, ssim)
            sslst.append(ssim)
            pslst.append(psnr1)
    
        psnrm = np.mean(np.array(pslst))
        print('psnr, num, psnrall1, ssimall = ', psnrm, num, psnrall1 / num, ssimall / num)
    
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
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)

    # # -------------------------- Isotropic Reconstruction --------------------------              
    def testiso(self, batch=0, epoch=None):
        self.testsave = self.dir + '/results/model%d/' % self.args.resume
        os.makedirs(self.testsave, exist_ok=True)
        datamin, datamax = self.args.datamin, self.args.datamax
        
        torch.set_grad_enabled(False)
        if epoch == None:
            epoch = self.optimizer.get_last_epoch()
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
            
            # 1.3D norm 2 998
            lrt = self.normalizer.before(lrt, 'CZYX')  # [0~3972] -> [0~2.8979]
            hrt = self.normalizerhr.before(hrt, 'CZYX')  # [0~4095] -> [0~2.9262]
            lrt, hrt = self.prepare(lrt, hrt)  # [B, 301, 752, 752]
            isoim = torch.zeros_like(hrt, dtype=hrt.dtype)

            lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))  # * 255
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))
            print('filename = ', filename, 'hr.shape, lr.shape = ', hr.shape, lr.shape)  # [301, 752, 752]
        
            batchstep = 10  # 5  #
            for dp in range(0, len(hr), batchstep):
                if dp + batchstep >= len(hr):
                    dp = len(hr) - batchstep
                # print(dp)  # 0, 10, .., 90
                # # # 2.2D norm 0–100
                # # lrtn = torch.from_numpy(normalize(lrt[:, dp:dp + batchstep, :, :].detach().cpu().numpy(), 0, 100)).to(
                # #     self.device)
                
                lrtn = torch.transpose(lrt[:, dp:dp + batchstep, :, :], 1, 0)  # [batch, 1, h, w]
                a = self.model(lrtn, 0)
                isoim[:, dp:dp + batchstep, :, :] = torch.transpose(a, 1, 0)  # [1, batch, h, w]

                sr = np.float32(np.squeeze(a.cpu().detach().numpy()))
                savecolorim(self.testsave + 'OriIm' + name + '-C%dLR.png' % dp, lr[dp])
                savecolorim(self.testsave + 'OriIm' + name + '-C%dHR.png' % dp, hr[dp])
                savecolorim(self.testsave + 'OriIm' + name + '-C%d.png' % dp, sr[0])
                
                ##  PSNR/SSIM
                # 3.(2D norm 0100 PSnr color save)
                hr2dim = np.float32(normalize(hr[dp], datamin, datamax, clip=True)) * 255  # [0, 1]
                sr2dim = np.float32(normalize(sr[0], datamin, datamax, clip=True)) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print('2D img Norm-%s - C%d- PSNR/SSIM/MSE = %f/%f' % (name, dp, psm, ssmm))
        
            sr = np.float32(isoim.cpu().detach().numpy())
            print('sr.shape = ', sr.shape)  #
            # 4.3D norm 2 998 tiff save
            sr = np.squeeze(self.normalizer.after(sr))
            lr = np.squeeze(self.normalizer.after(lr))
            imsave(self.testsave + name + '.tif', sr)
            # print('Save TIF image \' %s \' ' % (self.testsave + name + '.tif'))
            hr = np.squeeze(self.normalizerhr.after(hr))
        
            if 'Retina' in self.args.data_test:
                # Retina: [35, 2, 1024, 1024]
                hr = hr[:, 0, :, :]
                lr = lr[:, 0, :, :]
                # Liver [301, 752, 752]
                # Drosophila (108, 1352, 532)
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
                ##  PSNR/SSIM ~= 29.637/0.6842
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
        
        if self.args.test_only:
            file = open(self.testsave + "Psnrssim.txt", 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(pslst) + '\n SSIM \n' + str(sslst))
            file.write('PSNR of 2D image in different depth of each image \n PSNR \n' + str(pslstall) + '\n SSIM \n' + str(sslstall))
            file.close()
        else:
            if psnrm > self.bestpsnr:
                self.bestpsnr = psnrm
                self.bestep = epoch
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
        
        print('+++++++++ meanSR++++++++++++', sum(pslst) / len(pslst), sum(sslst) / len(sslst))
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)

    # # -------------------------- Isotropic Reconstruction --------------------------
    def testproj(self, batch=0, epoch=None, condition=0):
        if self.args.test_only:
            # self.testsave = self.dir + '/results/model%d/condition%d/' % (self.args.resume, condition)
            # self.testsave = self.dir + '/results/model%d/condition%d_C2GT/' % (self.args.resume, condition)
            self.testsave = self.dir + '/results/model%d/NoNormc%d_C2GT/' % (self.args.resume, condition)
        else:
            self.testsave = self.dir + '/valid_C2GT/'
        os.makedirs(self.testsave, exist_ok=True)
        print('save to', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax
    
        torch.set_grad_enabled(False)
        if epoch == None: epoch = self.optimizer.get_last_epoch()
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
        
            # 1.3D norm 2 998
            # lrt = self.normalizer.before(lrt, 'CZYX')  # [0~13.57]
            # hrt = self.normalizerhr.before(hrt, 'CYX')  # [0~5]
            lrt, hrt = self.prepare(lrt, hrt)

            if (not self.args.test_only) and \
                    (('2stg_gvt' in self.args.model) or ('2stg_enlcn' in self.args.model)):
                agvt, a = self.model(lrt, 0)  # [1, 1, h, w]
            elif (not self.args.test_only) and ('Uni-' in self.args.model):
                agvt, a, _ = self.model(lrt, 0)  # [1, 1, h, w]
            else:
                a = self.model(lrt, 0)  # [1, 1, h, w]
            sr = np.float32(np.squeeze(a.cpu().detach().numpy()))

            # 3D norm 2 998 tiff save
            # srtf = np.squeeze(self.normalizer.after(sr))
            srtf = np.squeeze(sr)
            # imsave(self.testsave + name + '.tif', srtf)
            axes_restored = 'YX'
            utility.save_tiff_imagej_compatible(self.testsave + name + '.tif', srtf, axes_restored)
            # print('Save TIF image \' %s \' ' % (self.testsave + name + '.tif'))
            
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))  # [1, 1, h, w]
            # # lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))  # [1, 50, h, w]
            # # lrm = np.mean(lr, 0, keepdims=False)
            # # print('filename = ', filename, 'hr.shape, lr.shape = ', hr.shape, lr.shape)  # [B, 301, 752, 752]
            # # savecolorim(self.testsave + 'OriIm' + name + '-LR.png', lrm)
            # savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr)
            # savecolorim(self.testsave + 'OriIm' + name + '-Result.png', sr)
            # # print('srtf.shape, sr.shape, hr.shape = ', srtf.shape, sr.shape, hr.shape)

            ##  PSNR/SSIM
            # 2.(2D norm 0100 PSnr color save)
            hr2dim = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255  # [0, 1]
            if epoch <= 30 and (not self.args.test_only) and \
                    (('2stg_gvt' in self.args.model) or ('2stg_enlcn' in self.args.model)):
                srgvt = np.float32(np.squeeze(agvt.cpu().detach().numpy()))
                sr2dim = np.float32(normalize(srgvt, datamin, datamax, clip=True)) * 255  # norm_srtf
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print('2D img Norm- !_(*@^@*)_! [GVTOs/ENLCN output] -%s - PSNR/SSIM = %f/%f' % (name, psm, ssmm))
            else:
                # sr2dim = np.float32(normalize(sr, datamin, datamax, clip=True)) * 255  # norm_sr
                sr2dim = np.float32(normalize(np.float32(srtf), datamin, datamax, clip=True)) * 255  # norm_srtf
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print('2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psm, ssmm))
            
            psnrall.append(psm)
            ssimall.append(ssmm)
        psnrallm = np.mean(np.array(psnrall))
        ssimallm = np.mean(np.array(ssimall))
        if self.args.test_only:
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

    # # -------------------------- 2D to 3D (VCDNet) --------------------------
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
    
        self.testsave = self.dir + 'results/model_%d/%s/' % (self.args.resume, subtestset)
        os.makedirs(self.testsave, exist_ok=True)
        print('make save path', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax

        torch.set_grad_enabled(False)
        if epoch == None:
            epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\n Evaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        # print(epoch, ' = epoch')

        psnrall, ssimall = 0, 0
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            if name == '':
                name = 'im%d' % idx_data
            print('image %s ' % (name + '.tif'))
            lrt, hrt = self.prepare(lrt, hrt)  # [1, 121, h//11, w//11]
            
            if ('_unfix' in self.args.model) or ('stage2' in self.args.model):
                if self.args.test_only:
                    # a = self.model(lrt, -1)  # -1: unet output
                    a = self.model(lrt, 0)  # 0: swinir output
                else:
                    au, a = self.model(lrt, 0)
            # elif ('SwinIR2t3' == self.args.model):
            #     if self.args.test_only:
            #         if epoch <= 30:
            #             a = self.model(lrt, -1)
            #         else:
            #             a = self.model(lrt, 0)
            #     else:
            #         if epoch <= 30:
            #             a, _ = self.model(lrt, 0)
            #         else:
            #             _, a = self.model(lrt, 0)
            elif (not self.args.test_only) and ('Uni-' in self.args.model):
                au, a = self.model(lrt, 0)
            else:
                a = self.model(lrt, 0)  # [1, 61, h, w]

            sr = np.float32(a.cpu().detach().numpy())
            # [1,649,649,61]
            write3d(np.transpose(sr, [0, 2, 3, 1]), self.testsave + name + '.tif', bitdepth=16, norm_max=True)
            # imsave(self.testsave + name + '.tif', np.squeeze(sr))
            print('Save TIF image \' %s \' ' % (self.testsave + name + '.tif'))
            
            sr = (np.clip(np.squeeze(sr), -1, 1) + 1) / 2
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
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                        psnrall += psm
                        ssimall += ssmm
                        print('2D img Norm-%s - PSNR/SSIM/MSE = %f/%f' % (name, psm, ssmm))
                else:
                    # hr == lr  # (16, 16, 121)
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
                        num += 1
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, lr2dim)
                        print('I%d, 2D img Norm-%s - PSNR/SSIM/MSE = %f/%f' % (i, name, psm, ssmm))
            else:
                savecolorim(self.testsave + 'OriIm' + name + '-Result.png', sr[0])
                savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr[0])
                ##  PSNR/SSIM
                for i in range(0, len(hr), 10):
                    num += 1
                    hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255  # [0, 1]
                    sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                    psnrall += psm
                    ssimall += ssmm
                    print('2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psm, ssmm))
                if (epoch <= 1) and (('_unfix' in self.args.model) or ('stage2' in self.args.model)):
                    print('valid a = au')
                    sru = np.float32(au.cpu().detach().numpy())
                    sru = (np.clip(np.squeeze(sru), -1, 1) + 1) / 2
                    for i in range(0, len(hr), 10):
                        hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255  # [0, 1]
                        sr2dimu = np.float32(normalize(sru[i], datamin, datamax, clip=True)) * 255
                        psmu, ssmmu = utility.compute_psnr_and_ssim(sr2dimu, hr2dim)
                        print('UNet 2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psmu, ssmmu))

        if psnrall > 0:
            psnrall = psnrall / num
            ssimall = ssimall / num
            if psnrall > self.bestpsnr:
                self.bestpsnr = psnrall
                self.bestep = epoch
            if not self.args.test_only:
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
        if self.args.test_only:
            # self.test()
            return False
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch <= self.args.epochs
