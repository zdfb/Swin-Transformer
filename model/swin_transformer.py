import torch
import numpy as np
import torch.nn as nn
from model.utils import load_pretrained
from model.layers import PatchEmbed, PatchMerging, BasicLayer


###### 构建Swin Transformer 主体结构 ######


# Swin Transformer主体架构
class SwinTransformer(nn.Module):
    def __init__(self, patch_size = 4, in_chans = 3, num_classes = 1000,
                 embed_dim = 96, depths = (2, 2, 6, 2), num_heads = (3, 6, 12, 24),
                 window_size = 7, mlp_ratio = 4., qkv_bias = True,
                 drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1,
                 norm_layer = nn.LayerNorm, patch_norm = True,):
        super().__init__()

        self.num_classes = num_classes  # 输出类别
        self.num_layers = len(depths)  # stage数量, 4

        self.embed_dim = embed_dim  # 编码维度数
        self.patch_norm = patch_norm

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 最后一个stage的输出维度  96 * 2 ** 3 = 96 * 8
        self.mlp_ratio = mlp_ratio  # mlp维度放大倍数
        
        # 对patch进行编码，划分为(224 / 4, 224/ 4)个patch，（B, 3, 224, 224） -> (B, 56 * 56, 96)
        self.patch_embed = PatchEmbed(
            patch_size = patch_size, in_c = in_chans, embed_dim = embed_dim,
            norm_layer = norm_layer if self.patch_norm else None)
        
        self.pos_drop = nn.Dropout(p = drop_rate)  # dropout

        # 在0 ~ 0.1间均匀生成(2 + 2 + 6 + 2)个点, 每层的dropout_rate逐渐递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        # 建立基础stage
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim = int(embed_dim * 2 ** i_layer),  # 输出维度为 96 * 2 ** layer
                                depth = depths[i_layer],  # 该stage重复次数
                                num_heads = num_heads[i_layer],  # 多头注意力机制头数
                                window_size = window_size,  # 窗口包含patch数
                                mlp_ratio = self.mlp_ratio,  # mlp层维度放大倍数 
                                qkv_bias = qkv_bias,
                                drop = drop_rate,
                                attn_drop = attn_drop_rate,
                                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer = norm_layer,
                                downsample = PatchMerging if (i_layer < self.num_layers - 1) else None,)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]

        print(x.shape)
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
def swin_small_patch4_window7_224(num_classes: int = 1000, pretrained = True):
    model = SwinTransformer(in_chans = 3,
                            patch_size = 4,
                            window_size = 7,
                            embed_dim = 96,
                            depths = (2, 2, 18, 2),
                            num_heads = (3, 6, 12, 24),
                            num_classes = num_classes,)

    if pretrained == True:
        load_pretrained('swin_small_patch4_window7_224.pth', model)

    return model
