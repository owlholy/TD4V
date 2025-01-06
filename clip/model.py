from typing import Tuple, Union
import numpy as np
from .VisualTransformer import *
from .Transformer import *
from .Bottleneck import *


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, T=8, config=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.T = T
        self.config = config

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 joint=False,
                 tm=None, T=8,dropout = 0., emb_dropout = 0.,side_dim = 384,
                 config=None
                 ):
        super().__init__()

        self.side_dim = side_dim

        self.context_length = context_length
        if dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, dropout, vision_layers)]  # stochastic depth decay rule
        else:
            dpr = None

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                T=T,
                config=config,
            )
            if tm:
                print('=========using Temporal Shift Module==========')
                from Recognition.modules.temporal_modeling import make_temporal_shift
                make_temporal_shift(self.visual, T)

        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                joint=joint, dropout=dpr,
                emb_dropout=emb_dropout,
                T=T,
                side_dim=side_dim,
                config=config,
            )
            if tm == 'tsm':
                print('=========using Temporal Shift Module==========')
                from Recognition.modules.temporal_modeling import make_temporal_shift_vit
                make_temporal_shift_vit(self.visual, T)
            elif tm == 'tokenshift':
                print('=========using TokenShift =========={} layers'.format(vision_layers))
                from Recognition.modules.temporal_modeling import make_tokenshift
                make_tokenshift(
                    self.visual, T, n_div=4,
                    locations_list=[x for x in range(vision_layers)]
                )
            elif tm == "tokent1d":
                print('=========using TokenT1D ==========')
                from Recognition.modules.temporal_modeling import make_tokenT1D
                make_tokenT1D(
                    self.visual, T, n_div=4,
                    locations_list=[x for x in range(vision_layers)]
                )                
            elif tm == 'dividedST':
                print('=========using DividedST ==========')
                from Recognition.modules.temporal_modeling import make_DividedST
                make_DividedST(
                    self.visual, T, vision_heads, emb_dropout, None,
                    locations_list=[8,9,10,11]
                )

            elif tm == 'localuni':
                print('=========using LocalUni ==========')
                from Recognition.modules.temporal_modeling import make_LocalUni
                if vision_layers == 12:
                    start = int(vision_layers * 1/3)
                else:
                    start = int(vision_layers * 1/3)
                make_LocalUni(
                    self.visual, T, vision_heads, emb_dropout, None,
                    locations_list=[x for x in range(start, vision_layers)]
                )                
            elif tm == 't1d':
                print('=========using T1D ==========')
                from Recognition.modules.temporal_modeling import make_T1D4VIT
                if vision_layers == 12:
                    start = int(vision_layers * 1/3)
                else:
                    start = int(vision_layers * 1/3)
                make_T1D4VIT(
                    self.visual, T,
                    locations_list=[x for x in range(start, vision_layers)]
                )    

            elif tm == 'atm':
                print('=========using ATM ==========')
                from Recognition.modules.ATM import make_ATM
                if vision_layers == 12:
                    start = 10 # int(vision_layers * 1/3)
                else:
                    start = 22 # int(vision_layers * 1/3)
                make_ATM(
                    self.visual, T,
                    locations_list=[x for x in range(start, vision_layers)]
                )  

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout=dpr,
            config=config,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        
        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.T = T

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
                        
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        side_fc_std = (2 * self.side_dim) ** -0.5

        for block in self.transformer.side_linears:
            nn.init.normal_(block.weight, std=side_fc_std)

        for block in self.transformer.side_lns:
            nn.init.zeros_(block.bias)
            nn.init.ones_(block.weight)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, images):
        return self.visual(images.type(self.dtype))


    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features, self.logit_scale.exp()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, tm=None,T=8,dropout=0., joint=False,emb_dropout=0.,pretrain=True, side_dim=384, config=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        tm=tm, T=T, joint=joint,
        dropout=dropout, emb_dropout=emb_dropout,side_dim=side_dim, config=config
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    if tm == True or tm in ["tsm", "tokenshift"]:
        # old dict for new model, rename some keys
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in model_dict.items():
            if k not in state_dict and k.replace('.net', '') in state_dict:
                replace_dict.append((k.replace('.net', ''), k))
        for k, k_new in replace_dict:
            state_dict[k_new] = state_dict.pop(k)


    convert_weights(model)
    if pretrain:
        print('loading clip pretrained model!')
        if joint and tm != "dividedST":  #or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict,strict=False)
        else:
            if tm == "tokent1d":
                model.load_state_dict(state_dict, strict=False)
            elif tm == "localuni":
                model.load_state_dict(state_dict, strict=False)
            elif tm == "t1d":
                model.load_state_dict(state_dict, strict=False)
            elif tm == "dividedST":
                # model.load_state_dict(state_dict, strict=False)
                model_dict = model.state_dict()
                new_state_dict = state_dict.copy()
                for key in state_dict:
                    if 'visual.transformer.resblocks' in key and 'attn' in key:
                        new_key1 = key.replace('attn','control_point1.temporal_attn')
                        new_key2 = key.replace('attn','control_point2.temporal_attn')
                        if new_key1 in model_dict:
                            new_state_dict[new_key1] = state_dict[key]
                        if new_key2 in model_dict:
                                new_state_dict[new_key2] = state_dict[key]
                    if 'visual.transformer.resblocks' in key and 'ln' in key:
                        new_key1 = key.replace('ln_1', 'control_point1.temporal_ln')
                        new_key2 = key.replace('ln_1', 'control_point2.temporal_ln')
                        if new_key1 in model_dict:
                            new_state_dict[new_key1] = state_dict[key]
                        if new_key2 in model_dict:
                            new_state_dict[new_key2] = state_dict[key]
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict, strict=False)


    else:
        print('not using full clip pretrained model, only visual!')

        for k in list(state_dict.keys()):
            if not k.find("visual")>-1:
                state_dict.pop(k)

        model.load_state_dict(state_dict,strict=False)

    return model.eval()
