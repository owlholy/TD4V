import torch.utils.checkpoint as checkpoint
from .base_function import *


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.control_point1 = AfterReconstruction(d_model)
        self.control_point2 = AfterReconstruction(d_model)
        self.control_atm = AfterReconstruction(d_model)

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, use_checkpoint=False):
        x = self.control_atm(x)
        x = self.control_point1(x)
        # MHSA
        if use_checkpoint:
            attn_out = checkpoint.checkpoint(self.attention, self.ln_1.float()(x))
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attention(self.ln_1.float()(x)))

        x = self.control_point2(x)
        # FFN
        if use_checkpoint:
            mlp_out = checkpoint.checkpoint(self.mlp, self.ln_2.float()(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2.float()(x)))
        return x


class ResidualAttentionBlock_text(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, config=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.prompt_tx = None
        self.prompt_tx_t = None

        # prompt_tx
        if config.text_encoder.enable:
            self.prompt_lenth = 4
            self.prompt_tx = nn.Parameter(torch.empty(self.prompt_lenth, 1, d_model))
            torch.nn.init.xavier_uniform_(self.prompt_tx.data)
        else:
            self.prompt_tx = None
        # prompt_tx'''

        # prompt_tx_t
        if config.text_encoder.text_t:
            self.prompt_t_lenth = 4
            self.prompt_tx_t = nn.Parameter(torch.empty(self.prompt_t_lenth, 1, d_model))
            torch.nn.init.xavier_uniform_(self.prompt_tx_t.data)
        else:
            self.prompt_tx_t = None
        # prompt_tx_t'''

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        try:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        except:
            mask = torch.empty(x.shape[0], x.shape[0]).to(dtype=x.dtype, device=x.device)
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
            return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor):
        N, B, D = x.shape

        if self.prompt_tx is not None and B != 1:
            prompt_tx = self.prompt_tx.expand(-1, B, D)
            prompt_lenth_half = self.prompt_lenth // 2
            x = torch.cat((prompt_tx[:prompt_lenth_half, :, :], x, prompt_tx[-prompt_lenth_half:, :, :]), dim=0)

        if self.prompt_tx_t is not None and B == 1:
            prompt_tx_t = self.prompt_tx_t.expand(-1, B, D)
            prompt_lenth_t_half = self.prompt_t_lenth // 2
            x = torch.cat((prompt_tx_t[:prompt_lenth_t_half, :, :], x, prompt_tx_t[-prompt_lenth_t_half:, :, :]), dim=0)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        if self.prompt_tx is not None and B != 1:
            x = x[prompt_lenth_half:-prompt_lenth_half, :, :]

        if self.prompt_tx_t is not None and B == 1:
            x = x[prompt_lenth_t_half:-prompt_lenth_t_half, :, :]

        return x


class Transformer_text(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, config=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock_text(width, heads, attn_mask, config) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)