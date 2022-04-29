import torch
import numpy as np
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from model.utils import window_partition, window_reverse,  DropPath




# patchembedding层
class PatchEmbed(nn.Module):
    def __init__(self, patch_size = 4, in_c = 3, embed_dim = 96, norm_layer = None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape  # (batch, C, H, W)

        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)

        if pad_input:
            # (W_left, W_right, H_top, H_bottom, C_front, C_back)
            # 将图片补齐，在右方及下方补零，使其能够被patch_size整除
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1], 0, self.patch_size[0] - H % self.patch_size[0], 0, 0))

        x = self.proj(x)  # 进行编码，(B, 3, 224, 224) -> (B, 96, 56, 56)
        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)  # (B, 96, 56, 56) -> (B, 96, 56 * 56) -> (B, 56 * 56, 96)
        x = self.norm(x)  # Linear norm
        return x, H, W


# PatchMerging操作, 类似于yolov5的Focus操作
class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer = nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 输入维度
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 降维
        self.norm = norm_layer(4 * dim)  # layer norm

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)  # (B, H * W, C) -> (B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 1::2, 0::2, :] 
        x2 = x[:, 0::2, 1::2, :]  
        x3 = x[:, 1::2, 1::2, :] 
        x = torch.cat([x0, x1, x2, x3], -1)  
        x = x.view(B, -1, 4 * C)  

        x = self.norm(x) 
        x = self.reduction(x)  # 降维

        return x


# 构建mlp层 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# 构建windowattention机制，仅在window内进行attention
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias = True, attn_drop = 0., proj_drop = 0.):

        super().__init__()
        self.dim = dim  # 输出特征尺度
        self.window_size = window_size  # windows尺寸，内部包含的patch数
        self.num_heads = num_heads  # 多头注意力机制头数
        head_dim = dim // num_heads  # 每个头的dim数
        self.scale = head_dim ** -0.5  # scale

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2 * widows_size - 1 * 2 * windows_size - 1, head], (13 * 13)

        coords_h = torch.arange(self.window_size[0])  # (0 ~ window_size)
        coords_w = torch.arange(self.window_size[1])  # (0 ~ window_size)

        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, window_size, window_size)
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size * window_size]
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, window_size * window_size, window_size * window_size)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (window_size * window_size, window_size * window_size, 2)

        relative_coords[:, :, 0] += self.window_size[0] - 1  # 将最小值转变为0
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 将最小值转变为0 
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 乘2 * 7 - 1
        relative_position_index = relative_coords.sum(-1)  # [window_size * window_size, window_size * window_size]

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 定义q, k, v
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 线性映射
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std =.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape  # (B * num_windowa, window_size * window_size, 编码维度)
      
        # (B * num_windows, window_size * window_size, 3 * embed_dim)
        # (B * num_windows, window_size * window_size, 3, head_num, head_dim) -> # (3, B * num_windows, head_num, window_size * window_size,  head_dim)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  #
        
        # 拆分q, k, v
        q, k, v = qkv.unbind(0)  # (B * num_windows, head_num, window_size * window_size,  head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # 计算余弦相似度

        # 添加相关位置编码，每个patch与其余所有patch的相对位置信息， （window_size * window_size, window_size * window_size, head_dim）
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        
        # 转换维度顺序，(head_dim, window_size * window_size, window_size * window_size)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        # 增加一维后，加到原来生成的余弦相似度中
        attn = attn + relative_position_bias.unsqueeze(0)  # (B * num_windows, head_dim, window_size * window_size, window_size * window_size)

        if mask is not None:
            nW = mask.shape[0]  # 窗口数

            # (B, num_windows, head_dim, window_size * window_size, window_size * window_size)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  # attention_drop_out

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 计算加权和

        x = self.proj(x)  # 线性空间映射
        x = self.proj_drop(x)  # proj_drop_out
        return x


# 构建stage内的每一个block
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size = 7, shift_size = 0,
                 mlp_ratio = 4., qkv_bias = True, drop = 0., attn_drop = 0., drop_path = 0.,
                 act_layer = nn.GELU, norm_layer = nn.LayerNorm):

        super().__init__()
        self.dim = dim  # 输出维度
        self.num_heads = num_heads  # 多头注意力机制头数
        self.window_size = window_size  # windows宽度包含的patch数
        self.shift_size = shift_size  # 滑动数
        self.mlp_ratio = mlp_ratio  # mlp全连接层放大倍数
 
        self.norm1 = norm_layer(dim)  # layer norm

        
        self.attn = WindowAttention(
            dim, window_size = (self.window_size, self.window_size), num_heads = num_heads, qkv_bias = qkv_bias,
            attn_drop = attn_drop, proj_drop = drop)  # 在窗口内进行self-attention操作

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)  # layer_norm层
        mlp_hidden_dim = int(dim * mlp_ratio)  # mlp隐藏层维度
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop=drop)  # 构建mlp层

    def forward(self, x, attn_mask):
        H, W = self.H, self.W  # 输入特征图的高与宽
        B, L, C = x.shape  # 输入形状 (batch_size, H * W,  embed_dim)
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # (batch_sizem, H, W, embed_dim)

        # 把特征图padding为window_size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape  

        # 特征图移位操作，先向下移，再向右移动
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts = (-self.shift_size, -self.shift_size), dims = (1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # 划分windows
        x_windows = window_partition(shifted_x, self.window_size)  # (B * num_windows, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (B * num_windows, window_size * window_size, C)

        # 在窗口内计算attention
        attn_windows = self.attn(x_windows, mask = attn_mask)  # (B * num_windows, window_size * window_size, C)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # (B * num_windows, window_size, window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # (B, map_size, map_size, C)

        # 还原特征图移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts = (self.shift_size, self.shift_size), dims = (1, 2))
        else:
            x = shifted_x

        # 若在处理时添加了padding，进行还原
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # (B, H, W, C) -> (B, H * W, C)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# 建立swin_transformer每个stage
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio = 4., qkv_bias = True, drop = 0., attn_drop = 0.,
                 drop_path = 0., norm_layer = nn.LayerNorm, downsample = None):
        super().__init__()

        self.dim = dim  # 输出维度
        self.depth = depth  # 该stage重复次数
        self.window_size = window_size  # 每一个windows内存在的patch数

        self.shift_size = window_size // 2  # 滑动窗口步长

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim = dim,
                num_heads = num_heads,
                window_size = window_size,
                shift_size = 0 if (i % 2 == 0) else self.shift_size,  # WA与SWA交替使用
                mlp_ratio = mlp_ratio, 
                qkv_bias = qkv_bias,
                drop = drop,
                attn_drop = attn_drop,
                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer = norm_layer)
            for i in range(depth)])

        # patch merging层，降采样
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):

        # 保证特征图的尺寸为window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size  # 更新之后的特征图尺寸
        Wp = int(np.ceil(W / self.window_size)) * self.window_size  # 更新之后的特征图尺寸
        
        img_mask = torch.zeros((1, Hp, Wp, 1), device = x.device)  # (1, Hp, Wp, 1)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (window_num, window_size, window_size, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # (window_num, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (window_num, 1, window_size * window_size) - (window_num, window_size * window_size, 1)
        
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # (window_num, window_size * window_size, window_size * window_size)
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # (window_num, window_size * window_size, window_size * window_size)


        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2  # 输入特征图长宽变为原来的一半

        return x, H, W
