import torch
import utility
from utility import IS_TF_1
import loss
from trainer import Trainer
torch.backends.cudnn.enabled = False
import argparse
import template
from data import SR, FlouresceneVCD, Flouresceneproj, Flouresceneiso, Flourescenedenoise
from torch.utils.data import dataloader
import model


def options():
    parser = argparse.ArgumentParser(description='EDSR and MDSR')
    parser.add_argument('--model', default=modelname, help='model name')
    parser.add_argument('--test_only', action='store_true', default=test_only, help='set this option to test the model')
    parser.add_argument('--task', type=int, default=task)
    parser.add_argument('--resume', type=int, default=resume, help='-2:best;-1:latest.ptb; 0:pretrain; >0: resume')
    parser.add_argument('--pre_train', type=str, default=pre_train, help='ENLCAx4.pt  pre-trained model directory')
    parser.add_argument('--save', type=str, default='%s%s/' % (modelname, testset), help='file name to save')
    if task == 2:
        parser.add_argument('--inputchannel', type=int, default=5, help='')
    if (task == 4) or (task == 2):
        parser.add_argument('--condition', type=int, default=condition)
    if task == 1:
        parser.add_argument('--inch', type=int, default=1, help='')

    # Data specifications
    parser.add_argument('--data_test', type=str, default=testset, help='demo image directory')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--patch_size', type=int, default=patch_size, help='input batch size for training')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)

    parser.add_argument('--cpu', action='store_true', default=iscpu, help='')
    parser.add_argument('--print_every', type=int, default=print_every, help='')
    parser.add_argument('--test_every', type=int, default=test_every)
    parser.add_argument('--load', type=str, default='', help='file name to load')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument('--n_resblocks', type=int, default=64, help='number of residual blocks')  # 32,  #
    parser.add_argument('--n_feats', type=int, default=256, help='number of feature maps')

    parser.add_argument('--save_models', action='store_true', default=True, help='save all intermediate models')

    parser.add_argument('--template', default='.', help='You can set various templates in option.py')
    parser.add_argument('--scale', type=str, default='%d' % scale, help='super resolution scale')
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
    parser.add_argument('--loss', type=str, default='1*L1', help='loss function configuration')
    
    args = parser.parse_args()
    template.set_template(args)
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    
    return args


def train():
    if task == 1:
        loader_train = dataloader.DataLoader(
            SR(args, name=testset, train=True, benchmark=False),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=not args.cpu,
            num_workers=0)
        loader_test = [dataloader.DataLoader(
            SR(args, name=testset, train=False, benchmark=False),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
    elif task == 2:
        loader_train = dataloader.DataLoader(
                Flourescenedenoise(args, istrain=True),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=0,
            )
        loader_test = [dataloader.DataLoader(
            Flourescenedenoise(args, istrain=False, c=condition),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0,
        )]
    elif task == 3:
        loader_test = [dataloader.DataLoader(
            Flouresceneiso(args, istrain=False),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
        loader_train = dataloader.DataLoader(
                Flouresceneiso(args, istrain=True),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=0,
            )
    elif task == 4:
        loader_test = [dataloader.DataLoader(
            Flouresceneproj(args, istrain=False, condition=condition),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
        loader_train = dataloader.DataLoader(
                Flouresceneproj(args, istrain=True),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=0)
    elif task == 5:
        loader_test = [dataloader.DataLoader(
            FlouresceneVCD(args, istrain=False, subtestset=subtestset),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
        loader_train = dataloader.DataLoader(
            FlouresceneVCD(args, istrain=True),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)
            
    _model = model.Model(args, checkpoint, unimodel)
    print('Total params: %.4fM' % (sum(p.numel() for p in _model.parameters()) / 1000000.0))
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader_train, loader_test, args.data_test, _model, _loss, checkpoint)

    while t.terminate():
        t.trainUni(tsk=task)
        
    checkpoint.done()


def test():
    loader_train = None
    if task == 1:
        loader_test = [dataloader.DataLoader(
            SR(args, name=testset, train=False, benchmark=False),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
    elif task == 2:
        loader_test = [dataloader.DataLoader(
            Flourescenedenoise(args, istrain=False, c=condition),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
    elif task == 3:
        loader_test = [dataloader.DataLoader(
            Flouresceneiso(args, istrain=False),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
    elif task == 4:
        loader_test = [dataloader.DataLoader(
            Flouresceneproj(args, istrain=False, condition=condition),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
    elif task == 5:
        loader_test = [dataloader.DataLoader(
            FlouresceneVCD(args, istrain=False, subtestset=subtestset),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=0)]
        
    _model = model.Model(args, checkpoint, unimodel)
    print('Total params: %.4fM' % (sum(p.numel() for p in _model.parameters()) / 1000000.0))
    t = Trainer(args, loader_train, loader_test, args.data_test, _model, None, checkpoint)
    
    if task == 1:
        t.test()
    elif task == 2:
        t.test3Ddenoise(condition=condition)
    elif task == 3:
        t.testiso()
    elif task == 4:
        t.testproj(condition=condition)
    elif task == 5:
        t.test2to3(subtestset=subtestset)
        
        
if __name__ == '__main__':
    task = 5
    test_only = True  # False  #
    patch_size = 64  # 128  # 32  #  LR
    resume = 0
    modelname = 'Uni-SwinIR'
    scale = 1
    if IS_TF_1:
        iscpu = True
        print_every = 2
        test_every = 2
    else:
        iscpu = False
        print_every = 100
        test_every = 500

    if task == 1:  # SR
        testset = 'F-actin'  # 'ER'  # 'Microtubules'  # 'CCPs'  #
        pre_train = './ENLCN/SwinIR/%s/model/model_best.pt' % (testset)
        scale = 2
    elif task == 2:  # denoise
        condition = 1
        testset = 'Denoising_Tribolium'  # 'Denoising_Planaria'  #'Synthetic_tubulin_granules', 'Synthetic_tubulin_gfp']  #,,
        pre_train = './experiment/SwinIRmto1Denoising_Tribolium/model/model_best97.pt'
    elif task == 3:  # isotropic
        testset = 'Isotropic_Liver'  # 'Isotropic_Retina'  # 'Isotropic_Drosophila'
        pre_train = './experiment/SwinIR%s/model/model_best465.pt' % testset
    elif task == 4:  # projection
        condition = 1
        testset = 'Projection_Flywing'
        pre_train = './experiment/SwinIRproj2stg_enlcn_2npz%s/model/model_best.pt' % testset
    elif task == 5:  # 2D to 3D
        testset = 'VCD'
        subtestset = 'to_predict'  # 'traindata'  # 'worm'  # 'beads'  #
        pre_train = './experiment/SwinIR2t3_stage2%s/model/model_best.pt' % testset

    args = options()
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    assert checkpoint.ok

    unimodel = model.UniModel(args, tsk=task)
    if test_only:
        test()
    else:
        train()
