from model.attention import *
import torch.nn as nn


def make_model(args, parent=False):
    if args.dilation:
        import dilated
        return ENLCN(args, dilated.dilated_conv)
    else:
        return ENLCN(args)


class ENLCN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ENLCN, self).__init__()
        
        n_resblock = args.n_resblocks
        inch = args.inch
        outch = args.n_colors
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        self.sub_mean = common.MeanShiftC1(args.rgb_range)
        self.add_mean = common.MeanShiftC1(args.rgb_range, sign=1)
        m_head = [conv(inch, n_feats, kernel_size)]
        
        m_body = [ENLCA(
            channel=n_feats, reduction=4,
            res_scale=args.res_scale)]
        for i in range(n_resblock):
            m_body.append(common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ))
            if (i + 1) % 8 == 0:
                m_body.append(ENLCA(
                    channel=n_feats, reduction=4,
                    res_scale=args.res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, outch, kernel_size,
                padding=(kernel_size // 2)
            )
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)
        self.tl = nn.Sequential(*m_tail)
    
    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = x
        comparative_loss = []
        for i in range(len(self.body)):
            if i % 9 == 0:
                res, loss = self.body[i](res)
                comparative_loss.append(loss)
            else:
                res = self.body[i](res)
        res += x
        
        x = self.tl(res)
        # x = self.add_mean(x)

        # return x, comparative_loss
        if self.training:
            return x, comparative_loss
        else:
            return x
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
