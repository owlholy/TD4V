from einops import rearrange
from fractions import  Fraction
from .base_function import *


class TD4V(nn.Module):
    def __init__(self, dim, adapter_scalar=0., drop_path=0., T=8, scale=1., sample_mode="FC"):
        super().__init__()
        self.dim = dim
        scale = Fraction(scale)
        self.side_dim = int(self.dim * scale)

        self.bn_down = bn_3d(dim)

        if sample_mode == "FC":
            self.conv_down = nn.Conv3d(self.dim, self.side_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=1)
        else:
            self.conv_down = nn.Conv3d(self.dim, self.side_dim, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=1)

        self.conv_1 = nn.Conv3d(self.side_dim, self.side_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=1)
        self.conv_2 = nn.Conv3d(self.side_dim, self.side_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=1)

        if sample_mode == "FC":
            self.conv_up = nn.Conv3d(self.side_dim, self.dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=1)
        else:
            self.conv_up = nn.Conv3d(self.side_dim, self.dim, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=1)

        self.act = QuickGELU()

        if adapter_scalar == "learnable_scalar" or adapter_scalar == 0.:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.T = T

    def forward(self, x):
        N, BT, D = x.shape
        H = int(N ** 0.5)
        T = self.T
        B = BT // T

        x = rearrange(x, '(h w) (b t) d -> b d t h w', h=H, t=T)
        x = self.conv_down(self.bn_down(x))

        x_t2 = torch.diff(x, dim=2)
        x_t1 = -torch.diff(x, dim=2)

        x_0 = torch.zeros((B, self.side_dim, 1, H, H)).to(x.dtype).to(x.device)
        x_t1 = torch.cat((x_t1, x_0), dim=2)
        x_t2 = torch.cat((x_0, x_t2), dim=2)

        x_t1 = self.conv_1(self.act(x_t1))
        x_t2 = self.conv_2(self.act(x_t2))

        x = x + x_t1 + x_t2

        x = self.conv_up(x)
        x = rearrange(x, 'b d t h w -> (h w) (b t) d')

        x = x * self.scale

        return x
