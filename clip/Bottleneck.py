from .base_function import *
from .adapter import *


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        # side_mid
        # side_inplanes = inplanes // 2
        # side_planes = planes // 2
        # self.side_bn = nn.BatchNorm3d(side_inplanes)
        # self.side_conv = nn.Sequential(*[
        #     nn.Conv3d(side_inplanes, side_inplanes, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=1),
        #     nn.Conv3d(side_inplanes, side_inplanes, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=side_inplanes),
        #     nn.Conv3d(side_inplanes, side_inplanes, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=1),
        # ])
        # self.side_conv1 = nn.Conv2d(side_inplanes, side_planes, 1, bias=False)
        # self.side_bn1 = nn.BatchNorm2d(side_planes)
        # self.side_conv2 = nn.Conv2d(side_planes, side_planes, 3, padding=1, bias=False)
        # self.side_bn2 = nn.BatchNorm2d(side_planes)
        # self.side_conv3 = nn.Conv2d(side_planes, side_planes * self.expansion, 1, bias=False)
        # self.side_bn3 = nn.BatchNorm2d(side_planes * self.expansion)

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

            # side_mid
            # self.side_downsample = nn.Sequential(OrderedDict([
            #     ("-1", nn.AvgPool2d(stride)),
            #     ("0", nn.Conv2d(side_inplanes, side_planes * self.expansion, 1, stride=1, bias=False)),
            #     ("1", nn.BatchNorm2d(side_planes * self.expansion))
            # ]))
            # self.bone2side = nn.Linear(planes, side_planes)

    def forward(self, x):
        # x, x_side = x

        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        # side_mid
        # identity_side = x_side
        # out_side = rearrange(x_side, '(b t) c w h -> b c t w h', t=8)
        # out_side = self.relu(self.side_bn(self.side_conv(out_side)))
        # out_side = rearrange(out_side, 'b c t w h -> (b t) c w h')
        # out_side = self.relu(self.side_bn1(self.side_conv1(out_side)))
        # out_side = self.relu(self.side_bn2(self.side_conv2(out_side)))

        # side_mid
        # if self.downsample is not None:
        #     x2s = rearrange(out, 'bt c h w -> bt h w c')
        #     x2s = self.bone2side(x2s)
        #     x2s = rearrange(x2s, 'bt h w c -> bt c h w')
        #     out_side = out_side * 0.5 + x2s * 0.5

        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        # side_mid
        # out_side = self.avgpool(out_side)
        # out_side = self.side_bn3(self.side_conv3(out_side))

        if self.downsample is not None:
            identity = self.downsample(x)
            # identity_side = self.side_downsample(x_side)  # side_mid

        out += identity
        out = self.relu(out)

        # side_mid
        # out_side += identity_side
        # out_side = self.relu(out_side)

        return out  # ori
        # return out, out_side  # side


class side_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, T=8, config=None):
        super().__init__()
        self.T = T
        self.config = config

        if config.resnet_mode.s4v == True:
            self.bn = nn.BatchNorm3d(inplanes)
            scale = Fraction(config.resnet_mode.s4v_ratio)
            bottleneck = int(inplanes * scale)
            self.conv = nn.Sequential(*[
                nn.Conv3d(inplanes, bottleneck, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=1),
                nn.Conv3d(bottleneck, bottleneck, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=bottleneck),
                nn.Conv3d(bottleneck, inplanes, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=1),
            ])
        if config.resnet_mode.td4v == True:
            self.dadapter = DAdapter(dim=inplanes, adapter_scalar=0., T=T, scale=config.resnet_mode.da_ratio, sample_mode=config.resnet_mode.sample_mode)

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        # with_residual
        if self.config.resnet_mode.s4v:
            x = rearrange(x, '(b t) c w h -> b c t w h', t=self.T)
            x = self.relu(x + self.bn(self.conv(x)))
            x = rearrange(x, 'b c t w h -> (b t) c w h')

        if self.config.resnet_mode.td4v:
            h = x.shape[2]
            x = rearrange(x, 'bt c h w -> (h w) bt c')
            x = x + self.dadapter(x)
            x = rearrange(x, '(h w) bt c -> bt c h w', h=h)

        out = x
        identity = x

        # without_residual
        # if self.config.resnet_mode.s4v:
        #     out = rearrange(x, '(b t) c w h -> b c t w h', t=self.T)
        #     out = self.relu(self.bn(self.conv(out)))
        #     out = rearrange(out, 'b c t w h -> (b t) c w h')

        # if self.config.resnet_mode.s4v:
        #     h = x.shape[2]
        #     out = rearrange(x, 'bt c h w -> (h w) bt c')
        #     out = self.dadapter(out)
        #     out = rearrange(out, '(h w) bt c -> bt c h w', h=h)

        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]
