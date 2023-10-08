import os, time
import torch
import torch.nn as nn
from model.enlcn import ProjectionUpdater
import model.swinir as module


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale[0]
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = not (torch.cuda.is_available())
        self.device = torch.device('cpu' if self.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        self.args = args
        
        if 'proj' in args.model:
            print('********** %s ***********' % args.model.lower())
            self.model = module.make_modelproj(args).to(self.device)
        elif 'SwinIR2t3' in args.model:
            print('********** %s ***********' % args.model.lower())
            self.model = module.make_model2t3(args).to(self.device)
        else:
            print('********** %s ***********' % args.model.lower())
            self.model = module.make_model(args).to(self.device)
                
        self.proj_updater = ProjectionUpdater(self.model, feature_redraw_interval=640)
        if args.precision == 'half':
            self.model.half()

        self.load(os.path.join('experiment', args.save), modelpath=args.modelpath, resume=args.resume)

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        st = time.time()
        self.proj_updater.redraw_projections()
        self.idx_scale = idx_scale
        
        print('image size', x.shape)
        if self.chop and not self.training and self.args.test_only:
            if 'proj' in self.args.model:
                result = self.forward_chopProj(x)
            elif '2t3' in self.args.model:
                result = self.forward_chop2to3(x)
            else:
                result = self.forward_chop(x, min_size=600000)
        else:
            result = self.model(x)
        
        print('forward time = ', time.time() - st)
        return result

    def get_model(self):
        return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, modelpath='.', resume=-1):
        if self.cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            print('Load Model from ', os.path.join(apath, 'model_latest.pt'))
            self.model.load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_latest.pt'),
                    **kwargs
                ),
                strict=True
            )
        elif resume == -2:
            m = os.path.join(apath, 'model_best.pt')
            print('Load Model from ', m)
            self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
        elif resume < -2:
            m = os.path.join(apath, 'model_best%d.pt' % -resume)
            print('Load Model from ', m)
            self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
        elif resume == 0 and modelpath != '.':
            print('Loading UNet model from {}'.format(modelpath))
            if ('2stg_enlcn' in self.args.model) or ('2stg_proj_care' in self.args.model):
                self.model.project.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
            elif ('stage2' in self.args.model) or ('_unfix' in self.args.model):
                self.model.conv_first0.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
            else:
                self.get_model().load_state_dict(torch.load(modelpath, **kwargs), strict=True)
        elif (resume > 0) and os.path.exists(os.path.join(apath, 'model_{}.pt'.format(resume))):
            print('Load Model from ', os.path.join(apath, 'model_{}.pt'.format(resume)))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=True
            )
        else:
            print('!!!!!!!!  Not Load Model  !!!!!!')
        
    def load_network(self, load_path, strict=True, param_key=None):  # 'params'params_ema
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(load_net, strict=strict)
        print(f'Loading {self.model.__class__.__name__} model from {load_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        # logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print('warning', f'  {v}')
            print('warning', 'Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print('warning', f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print('warning', f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def forward_chop(self, x, shave=10, min_size=6000):
        scale = self.scale
        n_GPUs = min(self.n_GPUs, 4)
        b, _, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        h_size, w_size = h_half + 16, w_half + 16
        h_size += 8
        w_size += 8

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
        # print('lr_list[0].size = ', lr_list[0].size())

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size)\
                for patch in lr_list]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, 1, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_chopProj(self, x, shave=10, min_size=1e60):
        scale = self.scale
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()

        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size += 16-h_size % 16  # 272
        w_size += 16-w_size % 16  #
        
        lr_list = [
                x[:, :, 0:h_size, 0:w_size],  # 272 360
                x[:, :, 0:h_size, (w - w_size):w],  # 272 360
                x[:, :, (h - h_size):h, 0:w_size],  # 272 360
                x[:, :, (h - h_size):h, (w - w_size):w]]  # 272 360
            
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                # print('Proj Output')
                if '2stg_enlcn' in self.args.model:
                    sr_batchgvt, sr_batch = self.model(lr_batch)
                
                # print('sr_batch.size() = ', sr_batch.size())
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self.forward_chopProj(patch, shave=shave, min_size=min_size) for patch in lr_list]
    
        # print('1  sr_list[0].size ', sr_list[0].size())
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
    
        # print('1 x.size', x.size())
        output = x.new(b, 1, h, w)
        # print('1 output', output.shape)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    
        return output

    def forward_chop2to3(self, x, shave=2, min_size=120000):
        scale = 11
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        h_size, w_size = h_half + shave, w_half + shave
      
        h_size += 8-h_size % 8
        w_size += 8-w_size % 8
        
        print('0 x.size = ', x.size(), 'h/w_size = ', h_size, w_size)
        
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
    
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if self.idx_scale < 0:
                    print('UNet 2to3 output')
                    sr_batch, _ = self.model(lr_batch)
                else:
                    print('SWinIR 2to3 output')
                    sr_batchu, sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self.forward_chop2to3(patch, shave=shave, min_size=min_size) for patch in lr_list]
    
        # print('1  sr_list[0].size ', sr_list[0].size())  # [1, 61, 176, 176]
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half  # 88, 88
        h_size, w_size = scale * h_size, scale * w_size  # 352 352
        
        output = x.new(b, 61, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    
        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            # if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output
