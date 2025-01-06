from .Transformer import *

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 dropout=None, joint=False, emb_dropout=0., T=8, side_dim=384, config=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.num_classes = config.data.num_classes

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        self.joint = joint
        self.T = T
        if joint:
            print('=====using space-time attention====')
            self.time_embedding = nn.Parameter(scale * torch.randn(T, width))  # pos emb
        if emb_dropout > 0:
            print('emb_dropout:{}'.format(emb_dropout))

        self.side_dim = side_dim
        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, dropout=dropout, side_dim=side_dim, T=T, patch_num=(input_resolution // patch_size) ** 2, config=config)

        side_scale = self.side_dim ** -0.5

        self.side_dims = config.network.pyramid.n_embs
        self.side_post_bn = bn_3d(self.side_dims[-1])

        self.side_conv1 = conv_3xnxn_std(3, self.side_dims[0], kernel_size=patch_size, stride=patch_size)

        self.side_pre_bn3d = nn.BatchNorm3d(self.side_dims[0])

        nn.init.ones_(self.side_pre_bn3d.weight)
        nn.init.zeros_(self.side_pre_bn3d.bias)
        nn.init.ones_(self.side_post_bn.weight)
        nn.init.zeros_(self.side_post_bn.bias)

    def forward(self, x: torch.Tensor):
        from einops import rearrange
        x_side = rearrange(x, '(b t) c h w -> b c t h w', t=self.T)

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.T)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        bs = x.shape[0] // self.T

        x_side = self.side_pre_bn3d(self.side_conv1(x_side))
        x_side = rearrange(x_side, 'b c t h w -> (b t) (h w) c')
        # x_side = torch.cat([self.temporal_class_embedding.to(x_side.dtype) + torch.zeros(x_side.shape[0], 1, x_side.shape[-1], dtype=x_side.dtype, device=x_side.device), x_side], dim=1)  # temporal_class

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.emb_dropout > 0:
            x = self.dropout(x)

        x = self.ln_pre.float()(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x_side = x_side.permute(1, 0, 2)

        x_side = self.transformer(x, x_side)

        x_side = x_side.permute(1, 0, 2)

        h = int(x_side.shape[1] ** 0.5)
        x_side = rearrange(x_side, '(b t) (h w) d -> b d t h w', t=self.T, h=h)
        x_side = self.side_post_bn(x_side)

        x_side = x_side.flatten(2).mean(-1)

        return x_side
