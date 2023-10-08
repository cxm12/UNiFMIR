import os
from decimal import Decimal
import utility
import torch.nn.utils as utils
import imageio
from utility import savecolorim
from div2k import normalize, PercentileNormalizer, np, torch
from tifffile import imsave
from tqdm import tqdm


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
        
    def train(self):
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

        for batch, (lr, hr, _,) in tqdm(list(enumerate(self.loader_train))):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            # 2D to 3D:
            if 'stage2' in self.args.model:
                srunet, sr = self.model(lr, 0)
                for param in self.model.model.conv_first0.parameters():
                    param.requires_grad = False
                loss = self.loss(sr, hr)
            # Projection:
            elif '2stg_enlcn' in self.args.model:
                srgvt, sr, comparative_loss = self.model(lr, 0)
                if epoch <= 20:
                    loss = 0.001 * self.loss(srgvt, hr) + self.loss(sr, hr)
                else:
                    loss = self.loss(sr, hr)
                
                comparative_loss = torch.mean(sum(comparative_loss) / len(comparative_loss))
                if self.optimizer.get_last_epoch() > 100:
                    loss += 0.001 * comparative_loss
            else:  #
                sr = self.model(lr, 0)
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

                if 'SwinIR2t3' in self.args.model:
                    sr2dimu = np.float32(
                        normalize(np.squeeze(srunet[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(sr2dimu, hr2dim)
                    print('UNet training patch = %f/%f' % (psm, ssmm))
                if '2stg_enlcn' in self.args.model:
                    sr2dimu = np.float32(
                        normalize(np.squeeze(srgvt[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(sr2dimu, hr2dim)
                    print('GVTOs stage1 training patch = %f/%f' % (psm, ssmm))

                print('Batch%d/Epoch%d, Loss = ' % (batch, epoch), loss)
                print('[{}/{}]\t{}/[comparative_loss:{}]\t{:.1f}+{:.1f}s'.format((batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset), self.loss.display_loss(batch),
                    comparative_loss.item(), timer_model.release(), timer_data.release()))
            timer_data.tic()
            
            if batch % self.args.test_every == 0:
                self.loss.end_log(len(self.loader_train))
                self.error_last = self.loss.log[-1, -1]
                self.optimizer.schedule()
                if 'Denoising' in self.args.data_test:
                    if 'mto1' in self.args.model:
                        self.test3DdenoiseInchannel5(batch, epoch)
                    else:
                        self.test3DdenoiseT(batch, epoch)
                elif 'Isotropic' in self.args.data_test:
                    self.testiso_rotate(batch, epoch)
                elif 'Flywing' in self.args.data_test:
                    self.testproj(batch, epoch)
                elif 'VCD' in self.args.data_test:
                    self.test2to3(batch, epoch)
                else:
                    self.test(batch, epoch)
                self.model.train()
                self.loss.step()
                lr = self.optimizer.get_lr()
                print('Evaluation -- Batch%d/Epoch%d' % (batch, epoch))
                self.ckp.write_log('Batch%d/Epoch%d' % (batch, epoch) + '\tLearning rate: {:.2e}'.format(Decimal(lr)))
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
        
        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lr, hr, filename) in tqdm(list(enumerate(self.loader_test[0]))):
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
            print('ps, pst, ps255 = ', ps, pst, ps255)
            pslst.append(np.max([ps, pst, ps255]))
            sslst.append(ss)
            
            if os.path.exists(self.dir):
                name = '{}.tif'.format(filename[0][:-4])
                # normalized = sr.mul(255 / self.args.rgb_range).byte().cpu()
                # sr = np.squeeze(normalized.detach().numpy())
                imageio.imwrite(self.testsave + name, sr)
                savecolorim(self.testsave + name[:-4] + '-Color.png', sr, norm=False)

                # normalizedhr = hr[0].mul(255 / self.args.rgb_range).byte().cpu()
                # hr1 = np.squeeze(normalizedhr.detach().numpy())
                savecolorim(self.testsave + name[:-4] + '-ColorHR.png', hr, norm=False)
                
                sr = np.round(np.maximum(0, np.minimum(255, sr)))
                hr2 = np.round(np.maximum(0, np.minimum(255, hr)))
                res = np.clip(np.abs(sr - hr2), 0, 255)
                # res1 = (normalized - normalizedhr).detach().numpy()
                # res = np.squeeze(np.clip(np.abs(res1), 0, 255))
                savecolorim(self.testsave + name[:-4] + '-MeandfnoNormC.png', res, norm=False)
                # print('Save to', name)
                
        psnrmean = np.mean(pslst)
        ssmean = np.mean(sslst)
        if self.args.test_only:
            # print('write in txt file')
            file = open(self.testsave + "Psnrssim100.txt", 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(pslst) + '\n SSIM \n' + str(sslst))
            file.close()
        else:
            if psnrmean > self.bestpsnr:
                self.bestpsnr = psnrmean
                self.bestep = epoch
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
            
        print('num', num, psnrmean, ssmean)
        print('psnrm, self.bestpsnr, self.bestep', psnrmean, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)

    # # -------------------------- 3D denoise --------------------------
    # test3Ddenoise --> input 5 channels
    def test3DdenoiseInchannel5(self, batch=0, epoch=None, condition=1):
        if self.args.test_only:
            self.testsave = self.dir + '/results/condition_%d/' % self.args.condition
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
    
        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lrt, hrt, filename) in tqdm(list(enumerate(self.loader_test[0]))):
            num += 1
            nmlst.append(filename)
            # print('filename = ', filename)
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
        
            hr = np.squeeze(hrt.cpu().detach().numpy())
            print('hr.shape = ', hr.shape)
            denoiseim = torch.zeros_like(hrt, dtype=hrt.dtype)
        
            batchstep = self.args.n_GPUs * 4
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

            sr = np.float32(denoiseim.cpu().detach().numpy())
            # 3.3D norm 2 998 tiff save；
            sr = np.squeeze(self.normalizer.after(sr))
            hr = np.squeeze(self.normalizerhr.after(hr))
        
            # 4.3D norm0100 psnr
            sr255 = np.squeeze(np.float32(normalize(sr, datamin, datamax, clip=True))) * 255  # [0, 1]
            hr255 = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255  # [0, 1]
        
            cpsnrlst = []
            cssimlst = []

            step = 1
            if self.args.test_only:
                imsave(self.testsave + name + '.tif', sr)
                if 'Planaria' in self.args.save:
                    if condition == 1:
                        randcs = 10
                        randce = hr.shape[0] - 10
                        step = (hr.shape[0] - 20) // 5
                    else:
                        randcs = 85
                        randce = 87
                        if randce >= hr.shape[0]:
                            randcs = hr.shape[0] - 3
                            randce = hr.shape[0]
    
                    for dp in range(randcs, randce, step):
                        savecolorim(self.testsave + name + '-dfnoNormC%d.png' % dp, sr[dp] - hr[dp], norm=False)
                        savecolorim(self.testsave + name + '-C%d.png' % dp, sr[dp])
                        savecolorim(self.testsave + name + '-HRC%d.png' % dp, hr[dp])
                        srpatch = sr255[dp, :patchsize, :patchsize]
                        hrpatch = hr255[dp, :patchsize, :patchsize]
        
                        ##  PSNR/SSIM
                        psm, ssmm = utility.compute_psnr_and_ssim(srpatch, hrpatch)
                        print('SR Image %s - C%d- PSNR/SSIM/MSE = %f/%f' % (name, dp, psm, ssmm))  # /%f, mse
                        cpsnrlst.append(psm)
                        cssimlst.append(ssmm)
                elif 'Tribolium' in self.args.save:
                    if condition == 1:
                        randcs = 2
                        randce = hr.shape[0] - 2
                        step = (hr.shape[0] - 4) // 6
                    else:
                        randcs = hr.shape[0] // 2 - 1
                        randce = randcs + 3
                        
                    for dp in range(randcs, randce, step):
                        hrpatch = normalize(hr255[dp, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        srpatch = normalize(sr255[dp, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        psm, ssmm = utility.compute_psnr_and_ssim(srpatch, hrpatch)
                        cpsnrlst.append(psm)
                        cssimlst.append(ssmm)
                        
                        savecolorim(self.testsave + name + '-dfnoNormC%d.png' % dp, sr[dp] - hr[dp], norm=False)
                        savecolorim(self.testsave + name + '-C%d.png' % dp, sr[dp])
                        savecolorim(self.testsave + name + '-HRC%d.png' % dp, hr[dp])
                        
                file.write('Image \"%s\" Channel = %d-%d \n' % (name, randcs, randce) + 'PSNR = ' + str(
                    cpsnrlst) + '\n SSIM = ' + str(cssimlst))
            else:
                randcs = hr.shape[0] // 2 - 1
                randce = randcs + 3

                for dp in range(randcs, randce, step):
                    hrpatch = normalize(hr255[dp, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                    srpatchour = normalize(sr255[dp, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(srpatchour, hrpatch)
                    cpsnrlst.append(psm)
                    cssimlst.append(ssmm)
                    print('Normalized Image %s - C%d- PSNR/SSIM = %f/%f' % (name, dp, psm, ssmm))
                   
            psnr1, ssim = np.mean(np.array(cpsnrlst)), np.mean(np.array(cssimlst))
            print('SR im:', psnr1, ssim)
            sslst.append(ssim)
            pslst.append(psnr1)
    
        psnrm = np.mean(np.array(pslst))
        ssimm = np.mean(np.array(sslst))
        print('psnr, ssim, num = ', psnrm, ssimm, num)
    
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
    def testiso_rotate(self, batch=0, epoch=None):
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

        self.testsave = self.dir + '/results/'
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
        for idx_data, (lrt, hrt, filename) in tqdm(list(enumerate(self.loader_test[0]))):
            num += 1
            nmlst.append(filename)
            name = '{}'.format(filename[0])
        
            # 1.3D norm 2 998
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
            
            batchstep = self.args.n_GPUs * 4
            if batchstep > 0:
                for wp in tqdm(range(0, hr.shape[2], batchstep)):
                    if wp + batchstep >= hr.shape[2]:
                        wp = hr.shape[2] - batchstep
                    # [d, h, w=batchstep, 1]-> [w=batchstep, h, d, 1]# [360,768,768,2] -> [768,768,360,2]
                    x_rot1 = _rotate(lr[:, :, wp:wp + batchstep, :], axis=1, copy=False)
                    # [w=batchstep, h, d, 1]-> [w=batchstep, h, d]-> [w=batchstep, 1, h, d]
                    x_rot1 = np.expand_dims(np.squeeze(x_rot1), 1)
        
                    x_rot1 = torch.from_numpy(np.ascontiguousarray(x_rot1)).float()
                    x_rot1 = self.prepare(x_rot1)[0]
                    a1 = self.model(x_rot1, 0)
        
                    # [w=batchstep, 1, h, d] -> [w=batchstep, h, d] -> [w=batchstep, h, d, 1]
                    a1 = np.expand_dims(np.squeeze(a1.cpu().detach().numpy()), -1)
                    # [w=batchstep, h, d, 1] -> [d, h, w=batchstep, 1]  # [360,768,768,2]
                    u1 = _rotate(a1, -1, axis=1, copy=False)
                    isoim1[:, :, wp:wp + batchstep, :] = u1
                for hp in tqdm(range(0, hr.shape[1], batchstep)):
                    if hp + batchstep >= hr.shape[1]:
                        hp = hr.shape[1] - batchstep
        
                    # [d, h=batchstep, w, 1]-> [h=batchstep, w, d, 1] # [768,768,360,2]
                    x_rot2 = _rotate(_rotate(lr[:, hp:hp + batchstep, :, :], axis=2, copy=False), axis=0, copy=False)
                    # [h=batchstep, w, d, 1]-> [h=batchstep, w, d]-> [h=batchstep, 1, w, d]
                    x_rot2 = np.expand_dims(np.squeeze(x_rot2), 1)
        
                    x_rot2 = torch.from_numpy(np.ascontiguousarray(x_rot2)).float()
                    a2 = self.model(self.prepare(x_rot2)[0], 0)
        
                    # [h=batchstep, 1, w, d] -> [h=batchstep, w, d] -> [h=batchstep, w, d, 1]
                    a2 = np.expand_dims(np.squeeze(a2.cpu().detach().numpy()), -1)
                    # [h=batchstep, w, d, 1] -> [d, h=batchstep, w, 1]  # [360,768,768,2]
                    u2 = _rotate(_rotate(a2, -1, axis=0, copy=False), -1, axis=2, copy=False)
                    isoim2[:, hp:hp + batchstep, :, :] = u2
            else:
                # [d, h, w=batchstep, 1]-> [w=batchstep, h, d, 1]# [360,768,768,2] -> [768,768,360,2]
                x_rot1 = _rotate(lr[:, :, :, :], axis=1, copy=False)
                # [w=batchstep, h, d, 1]-> [w=batchstep, h, d]-> [w=batchstep, 1, h, d]
                x_rot1 = np.expand_dims(np.squeeze(x_rot1), 1)
    
                x_rot1 = torch.from_numpy(np.ascontiguousarray(x_rot1)).float()
                x_rot1 = self.prepare(x_rot1)[0]
                a1 = self.model(x_rot1, 0)
    
                # [w=batchstep, 1, h, d] -> [w=batchstep, h, d] -> [w=batchstep, h, d, 1]
                a1 = np.expand_dims(np.squeeze(a1.cpu().detach().numpy()), -1)
                # [w=batchstep, h, d, 1] -> [d, h, w=batchstep, 1]  # [360,768,768,2]
                u1 = _rotate(a1, -1, axis=1, copy=False)
                isoim1[:, :, :, :] = u1

                # [d, h=batchstep, w, 1]-> [h=batchstep, w, d, 1] # [768,768,360,2]
                x_rot2 = _rotate(_rotate(lr[:, :, :, :], axis=2, copy=False), axis=0, copy=False)
                # [h=batchstep, w, d, 1]-> [h=batchstep, w, d]-> [h=batchstep, 1, w, d]
                x_rot2 = np.expand_dims(np.squeeze(x_rot2), 1)

                x_rot2 = torch.from_numpy(np.ascontiguousarray(x_rot2)).float()
                a2 = self.model(self.prepare(x_rot2)[0], 0)

                # [h=batchstep, 1, w, d] -> [h=batchstep, w, d] -> [h=batchstep, w, d, 1]
                a2 = np.expand_dims(np.squeeze(a2.cpu().detach().numpy()), -1)
                # [h=batchstep, w, d, 1] -> [d, h=batchstep, w, 1]  # [360,768,768,2]
                u2 = _rotate(_rotate(a2, -1, axis=0, copy=False), -1, axis=2, copy=False)
                isoim2[:, :, :, :] = u2
                
            sr = np.sqrt(np.maximum(isoim1, 0) * np.maximum(isoim2, 0))
            print('sr.shape = ', sr.shape)
            
            # 4.3D norm 2 998 tiff save
            sr = np.squeeze(self.normalizer.after(sr))
            lr = np.squeeze(self.normalizer.after(lr))
            imsave(self.testsave + name + '.tif', sr)

            hr = np.squeeze(self.normalizerhr.after(hr))
            c, h, w = hr.shape
        
            cpsnrlst = []
            cssimlst = []
            for dp in range(1, h, h // 5):
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
            self.model.save(self.dir + '/model/', epoch, is_best=(self.bestep == epoch))
    
        print('+++++++++ meanSR++++++++++++', sum(pslst) / len(pslst), sum(sslst) / len(sslst))
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)

    # # -------------------------- Isotropic Reconstruction --------------------------
    def testproj(self, batch=0, epoch=None, condition=0):
        if self.args.test_only:
            self.testsave = self.dir + 'c%d/' % condition
        else:
            self.testsave = self.dir + '/valid/'
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
        for idx_data, (lrt, hrt, filename) in tqdm(list(enumerate(self.loader_test[0]))):
            num += 1
            nmlst.append(filename)
            name = '{}'.format(filename[0])
        
            # 1.3D norm 2 998
            lrt, hrt = self.prepare(lrt, hrt)

            if (not self.args.test_only):
                agvt, a, _ = self.model(lrt, 0)
            else:
                a = self.model(lrt, 0)
                agvt = np.zeros(1)
            sr = np.float32(np.squeeze(a.cpu().detach().numpy()))

            # 3D norm 2 998 tiff save
            # srtf = np.squeeze(self.normalizer.after(sr))
            srtf = np.squeeze(sr)
            axes_restored = 'YX'
            utility.save_tiff_imagej_compatible(self.testsave + name + '.tif', srtf, axes_restored)
            
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))  # [1, 1, h, w]
            
            ##  PSNR/SSIM
            # 2.(2D norm 0100 PSnr color save)
            hr2dim = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255  # [0, 1]
            # sr2dim = np.float32(normalize(sr, datamin, datamax, clip=True)) * 255  # norm_sr
            sr2dim = np.float32(normalize(np.float32(srtf), datamin, datamax, clip=True)) * 255  # norm_srtf
            psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
            print('2D img Norm-%s - PSNR/SSIM = %f/%f' % (name, psm, ssmm))
            if len(agvt.shape) != 1:
                srgvt = np.float32(np.squeeze(agvt.cpu().detach().numpy()))
                sr2dim = np.float32(normalize(srgvt, datamin, datamax, clip=True)) * 255  # norm_srtf
                psm0, ssmm0 = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print('2D img Norm- !_(*@^@*)_! [Stage 1 output] -%s - PSNR/SSIM = %f/%f' % (name, psm0, ssmm0))

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
    
        self.testsave = self.dir + '/%s/' % (subtestset)
        os.makedirs(self.testsave, exist_ok=True)
        print('make save path', self.testsave)
        
        datamin, datamax = self.args.datamin, self.args.datamax

        torch.set_grad_enabled(False)
        if epoch == None:
            epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\n Evaluation: Batch%d/EPOCH%d' % (batch, epoch))
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        psnrall, ssimall = 0, 0
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in tqdm(list(enumerate(self.loader_test[0]))):
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            if name == '':
                name = 'im%d' % idx_data
            print('image %s ' % (name + '.tif'))
            lrt, hrt = self.prepare(lrt, hrt)  # [1, 121, h//11, w//11]

            if self.args.test_only:
                a = self.model(lrt, 0)
            else:
                au, a = self.model(lrt, 0)
            
            sr = np.float32(a.cpu().detach().numpy())
            print('sr.max(), sr.min() ', sr.max(), sr.min())
            write3d(np.transpose(sr, [0, 2, 3, 1]), self.testsave + name + '.tif', bitdepth=16, norm_max=True)
            sr = (np.clip(np.squeeze(sr), -1, 1) + 1) / 2
            print('norm sr.max(), sr.min() ', sr.max(), sr.min())
            imsave(self.testsave + name + 'norm.tif', np.squeeze(sr))
            print('Save TIF image \' %s \' ' % (self.testsave + name + '.tif'))
            
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
                    for i in range(0, len(hr), 1):
                        savecolorim(self.testsave + 'OriIm' + name + '-Result%d.png' % i, sr[i])
                        num += 1
                        hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255  # [0, 1]
                        sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                        psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                        psnrall += psm
                        ssimall += ssmm
                        print('2D img Norm-%s - PSNR/SSIM/MSE = %f/%f' % (name, psm, ssmm))
            else:
                savecolorim(self.testsave + 'OriIm' + name + '-Result.png', sr[0])
                savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr[0])

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
            # if self.args.precision == 'half':
            #     tensor = tensor.half()
            return tensor.to(self.device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            return False
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch <= self.args.epochs
