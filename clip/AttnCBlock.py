from kan.KAN import *
from efficient_kan.kan import *
from .adapter import *


class AttnCBlock(nn.Module):
    def __init__(self, dim, side_dim, mlp_ratio=1., drop=0., drop_path=0., act_layer=nn.GELU, T=8, config=None):
        super().__init__()
        self.dim = dim
        self.side_dim = side_dim

        self.td4v = TD4V(dim=dim, adapter_scalar=0., T=T, scale=config.network.DA_ratio,sample_mode=config.network.sample_mode)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP
        self.bn_2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # MLP'''

        self.attn = nn.MultiheadAttention(dim, dim // 64, dropout=0.)
        self.ln_1 = LayerNorm(dim)

        self.T = T
        side_attn_std = dim ** -0.5
        side_fc_std = (2 * dim) ** -0.5
        side_proj_std = (dim ** -0.5) * ((2 * 12) ** -0.5)
        for name, p in self.named_parameters():
            if 'mlp.fc1.weight' in name:
                nn.init.normal_(p, std=side_fc_std)
            elif 'mlp.fc2.weight' in name:
                nn.init.normal_(p, std=side_proj_std)
            elif 'pw_conv1.weight' in name:
                nn.init.normal_(p, std=0.02)
            elif 'pw_conv2.weight' in name:
                nn.init.normal_(p, std=0.02)
            elif 'dw_conv1.weight' in name:
                nn.init.normal_(p, std=side_attn_std)
            elif 'attn.in_proj_weight' in name:
                nn.init.normal_(p, std=side_attn_std)
            elif 'attn.out_proj.weight' in name:
                nn.init.normal_(p, std=side_proj_std)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.BatchNorm3d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = None  # self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x[1:], x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def shift_token_all(self, x_token, T):  # [1, bt, d]
        random_num = np.random.uniform()
        c = x_token.shape[-1]
        fold = c // T
        x_token = rearrange(x_token, 'n (b t) d -> n b t d', t=self.T)
        out = torch.zeros_like(x_token)
        for i in range(T):
            for j in range(T):
                out[:, :, i, j * fold:(j + 1) * fold] = x_token[:, :, j, i * fold:(i + 1) * fold]
        out = rearrange(out, 'n b t d -> n (b t) d')
        return out

    def forward(self, x, x_token=None, side_position_embeddings=None, layer_id=None, use_ckpt=False):
        h = int(x.shape[0] ** 0.5)
        x = x + self.td4v(x)

        if x_token is not None:
            ## shift class token
            x_token = self.shift_token_all(x_token, self.T)
            xt = torch.cat([x_token, x], dim=0)
            xt = xt.permute(1, 0, 2)
            if side_position_embeddings is not None:
                xt[:, 1:, :] = xt[:, 1:, :].cuda() + side_position_embeddings.cuda()
            xt = xt.permute(1, 0, 2)
            ## shift class token'''
        else:
            xt = x

        # attn
        xt = self.drop_path(self.attention(self.ln_1(xt)))
        x = x + xt
        # attn'''

        # mlp
        x_ = x
        x = rearrange(x, '(h w) (b t) d -> b d t h w', h=h, t=self.T)
        x = self.bn_2(x)
        x = rearrange(x, 'b d t h w -> (h w) (b t) d', h=h, t=self.T)
        x = x_ + self.drop_path(self.mlp(x))
        # mlp'''

        return x
