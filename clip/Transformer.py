from .ResidualAttentionBlock import *
from .AttnCBlock import *


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None, side_dim=384, T=8, patch_num=49, config=None):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)]
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers
        self.T = T
        self.side_dim = side_dim
        side_scale = self.side_dim ** -0.5
        self.config = config

        if self.config.network.fine_tuning:
            self.fine_tuning_mode = []

        self.side_upsample = []
        self.side_upsample_ln = []
        self.side_upsample_layers = []
        self.side_dims = config.network.pyramid.n_embs
        self.side_temporal_embeddings = nn.ParameterList()
        self.side_temporal_embeddings_layers = []

        self.resblocks = []
        self.side_transformer = []
        self.side_linears = []
        self.side_lns = []
        self.drop_layer_mode = 'fix random'  # random or fix or False or fix random
        self.side_start_layer = 0
        if self.drop_layer_mode == 'interval':
            self.drop_layer = [i for i in range(0, layers, 2)]
        else:
            self.drop_layer = [i for i in range(self.side_start_layer)]
        self.temporal_ratio = 1
        for i in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]))
            if i not in self.drop_layer:
                self.side_transformer.append(AttnCBlock(self.side_dims[i], int(self.side_dims[i] * self.temporal_ratio), mlp_ratio=config.network.pyramid.mlp_ratio, kernel_size=1, T=self.T, config=config))
                self.side_linears.append(nn.Linear(width, self.side_dims[i]))  # ori_fc_clip

                # fc_side_for_different_dim
                if self.side_dims[i] != self.side_dims[i - 1] and i != 0:
                    self.side_upsample.append(nn.Linear(self.side_dims[i - 1], self.side_dims[i]))
                    self.side_upsample_ln.append(LayerNorm(self.side_dims[i - 1]))
                    self.side_upsample_layers.append(i)
                # fc_side_for_different_dim'''

                # side_temporal_embeddings
                if self.side_dims[i] != self.side_dims[i - 1]:
                    self.side_temporal_embeddings.append(side_scale * torch.randn((self.T, self.side_dims[i])))
                    nn.init.normal_(list(self.side_temporal_embeddings.parameters())[-1], std=0.01)
                    self.side_temporal_embeddings_layers.append(i)
                # side_temporal_embeddings'''

                self.side_lns.append(LayerNorm(width))

        self.side_upsample = nn.ModuleList(self.side_upsample)
        self.side_upsample_ln = nn.ModuleList(self.side_upsample_ln)

        self.side_lns = nn.ModuleList(self.side_lns)  # ori_fc_clip
        self.side_linears = nn.ModuleList(self.side_linears)  # ori_fc_clip
        self.resblocks = nn.ModuleList(self.resblocks)
        self.side_transformer = nn.ModuleList(self.side_transformer)

    def forward(self, x: torch.Tensor, x_side: torch.Tensor, side_spatial_position_embeddings: torch.Tensor = None):
        k = 0
        h = int(x.shape[0] ** 0.5)
        j = 0
        l = 0
        for i in range(len(self.resblocks)):
            x = self.resblocks[i](x)  # transformer_block
            if i in self.drop_layer:
                if i >= self.side_start_layer and self.drop_layer_mode != 'interval':
                    k += 1
                continue

            xs2xt = self.side_linears[k](self.side_lns[k](x))

            x_token = xs2xt[:1, :, :]
            xs2xt = xs2xt[1:, :, :]
            if (x_side.shape[0] != xs2xt.shape[0]):
                xs2xt = rearrange(xs2xt, '(n t) b d -> n (b t) d', t=self.T)

            if i in self.side_upsample_layers:
                # if self.side_upsample[j].weight.shape[0] == self.side_upsample[j].weight.shape[1]:
                #     x_side = x_side + self.side_upsample[j](self.side_upsample_ln[j](x_side))
                # else:
                x_side = self.side_upsample[j](self.side_upsample_ln[j](x_side))
                j += 1

            x_side = x_side + self.side_attn[i](self.side_attn_ln[i](x_side), self.side_attn_ln[i](x_side), self.side_attn_ln[i](x_side), need_weights=False, attn_mask=None)[0]

            x_side = 0.5 * x_side[:x_side.shape[0] // 2] + 0.5 * x_side[x_side.shape[0] // 2:]
            x_side = torch.cat((x_token, x_side), dim=0)
            x_side = self.side_upsample[i](self.side_upsample_ln[i](x_side))
            x_token = x_side[:1]
            x_side = x_side[1:]

            x_side = 0.5 * x_side + 0.5 * xs2xt

            # side_temporal_embeddings
            if i in self.side_temporal_embeddings_layers:
                side_temporal_embeddings = list(self.side_temporal_embeddings.parameters())[l]
                x_side = rearrange(x_side, 'n (b t) d -> (n b) t d', t=self.T)
                x_side = x_side + side_temporal_embeddings.to(x_side.dtype).to(x_side.device)
                x_side = rearrange(x_side, '(n b) t d -> n (b t) d', n=xs2xt.shape[0])
                l += 1
            # side_temporal_embeddings'''

            x_side = self.side_transformer[k](x_side, x_token, None, i)
            k += 1

        return x_side
