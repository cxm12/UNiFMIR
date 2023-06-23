# unify all IR models

from model.swinir import *


def make_model(args):
    args.n_resblocks = 64
    args.n_feats = 256
    return UniModel(args=args)
    

class UniModel(nn.Module):
    def __init__(self, args, tsk=1, img_size=64, patch_size=1,
                 embed_dim=180 // 2, depths=[6, 6, 6], num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, num_feat=32):
        super(UniModel, self).__init__()
        self.img_range = 1
        self.mean = torch.zeros(1, 1, 1, 1)
        self.window_size = window_size
        self.task = tsk
        
        # 1 SR
        self.conv_firstsr = nn.Conv2d(1, embed_dim, 3, 1, 1)
        self.upsamplesr = Upsample(2, num_feat)
        # 2 denoise
        self.conv_firstdT = nn.Conv2d(5, embed_dim, 3, 1, 1)
        # 3 iso
        self.conv_firstiso = nn.Conv2d(1, embed_dim, 3, 1, 1)
        # 4 Projection
        args.n_resblocks = 64
        args.n_feats = 256
        args.inch = 50
        self.project = ENLCN(args=args)
        self.conv_firstproj = nn.Conv2d(1, embed_dim, 3, 1, 1)
        
        # 5 2D to 3D
        self.conv_first0 = UNetA(121, 61)
        self.conv_firstv = nn.Conv2d(61, embed_dim, 3, 1, 1)
        self.conv_before_upsamplev = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_lastv = nn.Conv2d(embed_dim, 61, 3, 1, 1)

        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(self.patch_embed.patches_resolution[0],
                                           self.patch_embed.patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection='1conv'
                         )
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        self.conv_before_upsample0 = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(1, num_feat)
        self.conv_last0 = nn.Conv2d(num_feat, 1, 3, 1, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, x):
        if self.task == 1:
            x = self.check_image_size(x)
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            x = self.conv_firstsr(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample0(x)
            x = self.upsamplesr(x)
            x = self.conv_last0(x)
        elif self.task == 2:
            x = self.check_image_size(x)
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            x = self.conv_firstdT(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample0(x)
            x = self.upsample(x)
            x = self.conv_last0(x)
        elif self.task == 3:
            x = self.check_image_size(x)
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            x = self.conv_firstiso(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample0(x)
            x = self.upsample(x)
            x = self.conv_last0(x)
        elif self.task == 4:
            x2d, closs = self.project(x)

            x2d = self.check_image_size(x2d)
            self.mean = self.mean.type_as(x2d)
            x2d = (x2d - self.mean) * self.img_range
            
            x = self.conv_firstproj(x2d)
            x = self.conv_after_body(self.forward_features(x))
            x = self.conv_before_upsample0(x)
            x = self.conv_last0(x)
            return x2d, x / self.img_range + self.mean + x2d, closs
        elif self.task == 5:
            x = self.check_image_size(x)
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            xunet = self.conv_first0(x)
            
            x = self.conv_firstv(xunet)
            x = self.conv_after_body(self.forward_features(x))  # + x
            x = self.conv_before_upsamplev(x)
            x = self.conv_lastv(x)
            return xunet, x / self.img_range + self.mean

        x = x / self.img_range + self.mean

        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x
