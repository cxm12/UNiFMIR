# from torchsummary import summary
from thop import profile
from model.swinir import swinir, swinir2dto3d, swinirup, swinirProj_stage2, UNetA, swinirhugecnn, torch
import argparse
from model.Unimodel import UniModel


def Unimodel():
    model = UniModel(args)  # FLOPs: 1305.23647G, Params:174.83787M
    
    model.task = 1
    input = torch.randn(1, 1, 128, 128)
    flops, params = profile(model, inputs=(input,))
    # # summary(model, (1, 128, 128))
    
    print(f'FLOPs: {flops / 1e9:.5f}G, Params:{params / 1e6:.5f}M', params)
    exit()


def SR():
    model = swinir(upscale=2, in_chans=1)  # FLOPs: 44.07G, Params:1.54067M 1540673.0
    input = torch.randn(1, 1, 128, 128)
    flops, params = profile(model, inputs=(input,))
    # # summary(model, (1, 128, 128))
    
    print(f'FLOPs: {flops / 1e9:.5f}G, Params:{params / 1e6:.5f}M', params)
    exit()


def denoise():
    # inch = 5  # 10.87559G, Params:1.50692M
    inch = 1  # 10.86232G, Params:1.50368M
    model = swinir(upscale=1, in_chans=inch)
    input = torch.randn(1, inch, 64, 64)
    flops, params = profile(model, inputs=(input,))
    # # summary(model, (inch, 64, 64))
    
    print(f'FLOPs: {flops / 1e9:.5f}G, Params:{params / 1e6:.5f}M', params)
    exit()


def ISO():
    inch = 1  # 10.86232G, Params:1.50368M
    model = swinir(upscale=1, in_chans=inch)
    input = torch.randn(1, inch, 64, 64)
    flops, params = profile(model, inputs=(input,))
    # # summary(model, (inch, 64, 64))
    
    print(f'FLOPs: {flops / 1e9:.5f}G, Params:{params / 1e6:.5f}M', params)
    exit()


def Proj():
    # model = swinir(upscale=1, in_chans=50, out_chans=1)  # 11.02489G, Params:1.54337M
    # model = swinirProj_stage2(upscale=1, in_chans=50, out_chans=1, gvt=True)  # 11.14906G, Params:1.67306M
    # model = swinirProj_stage2(upscale=1, in_chans=50, out_chans=1, gvt=False)  # 21.88721G, Params:3.04705M
    # model = swinir(img_size=64, num_feat=256, upscale=1, in_chans=50, embed_dim=256,
    #                depths=[16 for _ in range(6)], num_heads=[8 for _ in range(6)])  #
    gvt = 2
    
    model = swinirProj_stage2(upscale=1, in_chans=50, out_chans=1, gvt=gvt, args=args)


    input = torch.randn(1, 50, 64, 64)
    flops, params = profile(model, inputs=(input,))
    # # summary(model, (50, 64, 64))
    
    print(f'FLOPs: {flops / 1e9:.5f}G, Params:{params / 1e6:.5f}M', params)
    exit()


def Volumerec():
    # model = swinir(upscale=11, in_chans=121, out_chans=61)  # 5.88822G, Params:2.40740M
    # model = UNetA(121, 61)  # 44.57558, Params:19.01660M
    # model = swinirup(upscale=11, in_chans=121, out_chans=61)  # 85.87570G, Params:1.62411M
    model = swinir2dto3d(upscale=11, in_chans=121, out_chans=61)  # 131.20955G, Params:20.66510M

    input = torch.randn(1, 121, 16, 16)
    flops, params = profile(model, inputs=(input,))
    # # summary(model, (121, 176, 176))

    print(f'FLOPs: {flops / 1e9:.5f}G, Params:{params / 1e6:.5f}M', params)
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDSR and MDSR')
    parser.add_argument('--n_resblocks', type=int, default=64, help='')
    parser.add_argument('--n_feats', type=int, default=256, help='')
    parser.add_argument('--inch', type=int, default=50, help='')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--scale', type=str, default='1', help='')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    Unimodel()
    # Volumerec()
    Proj()
    ISO()
    denoise()
    SR()
