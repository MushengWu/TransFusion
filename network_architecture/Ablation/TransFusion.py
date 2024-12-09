import torch
import torch.nn as nn
import torch.nn.functional as F

from network_architecture.window_function import window_partition, window_reverse, computer_mask

from timm.models.layers import trunc_normal_


class Conv3d_BN(nn.Sequential):
    def __init__(self, a, b,
                 ks=(1, 1, 1), stride=(1, 1, 1), pad=(0, 0, 0), dilation=(1, 1, 1), groups=1, bn_weight_init=1.):
        super().__init__()
        self.add_module('c', nn.Conv3d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm3d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv3d(w.size(1) * self.c.groups, w.size(0), w.shape[2:],
                      stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0.:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_channels=1, embed_dim=(16, 32), norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj_1 = nn.Sequential(Conv3d_BN(in_channels, embed_dim[0], 3, 2, 1), nn.ReLU(),
                                    Conv3d_BN(embed_dim[0], embed_dim[0], 3, 1, 1), nn.ReLU())
        self.proj_2 = Conv3d_BN(embed_dim[0], embed_dim[1], 3, 2, 1, bn_weight_init=0.)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, D, H, W = x.shape

        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x1 = self.proj_1(x)
        x2 = self.proj_2(x1)
        if self.norm is not None:
            D, H, W = x2.shape[2:]
            x2 = x2.flatten(2).transpose(1, 2)  # B,C,D,H,W -> B,C,DHW -> B,DHW,C
            x2 = self.norm(x2)
            x2 = x2.transpose(1, 2).view(-1, self.embed_dim, D, H, W)  # B,DHW,C -> # B,C,D,H,W
        return x1, x2


class SqueezeExcite3D(nn.Module):
    def __init__(self, dim, rd_ratio=0.25, norm_layer=None):
        super().__init__()
        hid_dim = int(dim * rd_ratio)
        self.fc1 = nn.Conv3d(dim, hid_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.bn = nn.BatchNorm3d(hid_dim) if norm_layer is not None else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(hid_dim, dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.sigmoid(x_se)


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, kernel=3, stride=2, padding=1):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv3d_BN(dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = Conv3d_BN(hid_dim, hid_dim, kernel, stride, padding, groups=hid_dim)
        self.se = SqueezeExcite3D(hid_dim, .25)
        self.conv3 = Conv3d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class PatchMerging_(nn.Module):
    def __init__(self, dim, out_dim, kernel=3, stride=2, padding=1, res=False):
        super().__init__()
        hid_dim = int(dim * 4)
        self.res = res
        self.conv1 = Conv3d_BN(dim, dim, ks=kernel, stride=stride, pad=padding, groups=dim)
        self.conv2 = nn.Conv3d(dim, hid_dim, kernel_size=(1, 1, 1))
        self.conv3 = nn.Conv3d(hid_dim, out_dim, kernel_size=(1, 1, 1))
        # self.norm = nn.GroupNorm(num_groups=dim, num_channels=dim)
        self.act = nn.ReLU(inplace=True)
        if res:
            self.res_conv = nn.Conv3d(dim, out_dim, kernel_size=(1, 1, 1), stride=(2, 2, 2))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.act(self.conv2(x1))
        x1 = self.conv3(x1)
        if self.res:
            res = self.res_conv(x)
            x1 = x1 + res
        return x1


class DwConvBlock(nn.Module):
    def __init__(self, dim, kernel, stride, pad):
        super().__init__()
        self.conv_1 = nn.Sequential(
            Conv3d_BN(dim, int(2 * dim)),
            nn.ReLU(inplace=True)
        )
        self.dw_conv = Conv3d_BN(int(2 * dim), int(2 * dim), ks=kernel, stride=stride, pad=pad, groups=int(2 * dim))
        self.conv_2 = nn.Sequential(
            Conv3d_BN(int(2 * dim), dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        x = self.conv_1(x)
        x = self.dw_conv(x)
        x = self.conv_2(x)
        return x + residual


class MultiScaleConv(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.conv = nn.ModuleList(
            [
                DwConvBlock(dim=dim // num_heads,
                            kernel=3 + i * 2,
                            stride=1,
                            pad=1 + i)
                for i in range(num_heads)
            ]
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        feats = x.chunk(self.num_heads, dim=1)
        out = []
        for i, conv in enumerate(self.conv):
            out.append(conv(feats[i]))
        x = torch.cat(feats, dim=1)

        # shuffle channels
        x = x.view(B, self.num_heads, C // self.num_heads, D, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, -1, D, H, W)
        return x


class CascadeAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio, window_size):
        super().__init__()
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv3d_BN(dim // num_heads, self.key_dim * 2 + self.d))
            dws.append(Conv3d_BN(self.key_dim, self.key_dim, ks=3, pad=1, groups=self.key_dim))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)

        self.proj = nn.Sequential(nn.ReLU(), Conv3d_BN(self.d * num_heads, dim, bn_weight_init=0))

        # Using the relative position embedding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        coord_d = torch.arange(window_size[0])
        coord_h = torch.arange(window_size[1])
        coord_w = torch.arange(window_size[2])
        coord = torch.stack(torch.meshgrid(coord_d, coord_h, coord_w))
        coord_flatten = torch.flatten(coord, 1)
        relative_coord = coord_flatten[:, :, None] - coord_flatten[:, None, :]
        relative_coord = relative_coord.permute(1, 2, 0).contiguous()
        relative_coord[:, :, 0] += window_size[0] - 1
        relative_coord[:, :, 1] += window_size[1] - 1
        relative_coord[:, :, 2] += window_size[2] - 1

        relative_coord[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coord[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coord.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, C, Wd, Wh, Ww = x.shape
        N = Wd * Wh * Ww
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]

        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, Wd, Wh, Ww).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B*nD*nH*nW, C/h, N
            attn = (
                    (q.transpose(-2, -1) @ k) * self.scale
                    +
                    relative_position_bias[i]
            )
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B // nW, -1, N, N) + mask.unsqueeze(0)
                attn = attn.view(-1, N, N)
                attn = attn.softmax(dim=-1)  # BNN
            else:
                attn = attn.softmax(dim=-1)  # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, Wd, Wh, Ww)  # B*nD*nH*nW,C/h,d,h,w
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, dim=1))
        self.vis = x
        return x


class CascadeAttentionBlock(nn.Module):
    """
        input  -> [B, C, D, H, W]
        output -> [B, C, D, H, W]

    Args:


    """
    def __init__(self, dim, key_dim, num_heads, attn_ratio, window_size=(7, 7, 7), shifted=False):
        super().__init__()
        self.shifted = shifted
        if shifted:
            self.shift_size = (3, 3, 3)
        else:
            self.shift_size = (0, 0, 0)
        self.window_size = window_size
        self.attn = CascadeAttention(dim, key_dim, num_heads, attn_ratio, window_size)

    def forward(self, x):
        B, C, D, H, W = x.shape

        if D <= self.window_size[0] and H <= self.window_size[1] and W <= self.window_size[2]:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)

            pad_d1 = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
            pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
            pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

            padding = pad_d1 > 0 or pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))

            if self.shifted:
                Dp, Hp, Wp = x.shape[1:4]
                attn_mask = computer_mask(Dp, Hp, Wp, self.window_size, self.shift_size, x.device)
            else:
                attn_mask = None
            # cyclic shift
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))

            x = window_partition(x, self.window_size)
            x = self.attn(x, attn_mask)
            x = window_reverse(x, self.window_size, B, C, D + pad_d1, H + pad_b, W + pad_r)
            # reverse cyclic shift
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(2, 3, 4))
            if padding:
                x = x[:, :D, :H, :W, :].contiguous()
            x = x.permute(0, 4, 1, 2, 3)
        return x


class FFN(nn.Module):
    def __init__(self, dim, hid_dim=None):
        super().__init__()
        if hid_dim is None:
            hid_dim = int(dim * 2)
        self.pw1 = Conv3d_BN(dim, hid_dim)
        self.act = nn.ReLU()
        self.pw2 = Conv3d_BN(hid_dim, dim, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class EfficientViTBlock(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio, window_size):
        super().__init__()
        self.dw0 = Residual(Conv3d_BN(dim, dim, ks=3, stride=1, pad=1, bn_weight_init=0.))
        self.attn_1 = Residual(CascadeAttentionBlock(dim, key_dim, num_heads, attn_ratio, window_size))
        self.ffn_1 = Residual(FFN(dim=dim, hid_dim=int(dim * 2)))
        self.attn_2 = Residual(CascadeAttentionBlock(dim, key_dim, num_heads, attn_ratio, window_size, shifted=True))
        self.ffn_2 = Residual(FFN(dim=dim, hid_dim=int(dim * 2)))

    def forward(self, x):
        x = self.dw0(x)
        x = self.attn_1(x)
        x = self.ffn_1(x)
        x = self.attn_2(x)
        x = self.ffn_2(x)
        return x


class Cascade_stage(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio, window_size, depths, down=False):
        super().__init__()
        if down:
            self.down = PatchMerging_(dim=dim // 2, out_dim=dim, res=True)
        else:
            self.down = None

        self.multi_conv = MultiScaleConv(dim=dim, num_heads=num_heads)
        attn_blocks = []
        for i in range(depths):
            attn_blocks.append(
                nn.Sequential(
                    EfficientViTBlock(dim, key_dim, num_heads, attn_ratio, window_size)
                )
            )
        self.attn_blocks = nn.ModuleList(attn_blocks)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.multi_conv(x)

        for blk in self.attn_blocks:
            x = blk(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, dim=[64, 128, 256]):
        super().__init__()
        self.se2 = SqueezeExcite3D(dim[0])
        self.se3 = SqueezeExcite3D(dim[1])
        self.se4 = SqueezeExcite3D(dim[2])

        self.upx2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.upx4 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')

        self.downx2 = nn.Conv3d(dim[0], dim[1], (2, 2, 2), (2, 2, 2))
        self.downx4 = nn.Conv3d(dim[1], dim[2], (2, 2, 2), (2, 2, 2))

        self.fc1 = nn.Conv3d(dim[1], dim[0], (1, 1, 1), bias=False)
        self.fc2 = nn.Conv3d(dim[2], dim[0], (1, 1, 1), bias=False)
        self.fc3 = nn.Conv3d(dim[2], dim[1], (1, 1, 1), bias=False)

        self.dw2 = Residual(Conv3d_BN(dim[0], dim[0], 3, 1, 1, groups=dim[0], bn_weight_init=0.))
        self.dw3 = Residual(Conv3d_BN(dim[1], dim[1], 3, 1, 1, groups=dim[1], bn_weight_init=0.))
        self.dw4 = Residual(Conv3d_BN(dim[2], dim[2], 3, 1, 1, groups=dim[2], bn_weight_init=0.))

        self.conv = Conv3d_BN(2, 1, ks=3, pad=1, bn_weight_init=0.)
        self.sigmoid = nn.Sigmoid()

    def spatial_attention(self, x):
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.conv(torch.cat((max_out, avg_out), dim=1))
        spatial_out = self.sigmoid(spatial_out)
        return x * spatial_out

    def forward(self, x1, x2, x3):
        # x1 = self.se2(x1)
        # x2 = self.se3(x2)
        # x3 = self.se4(x3)

        # x1_fused = x1 + self.upx2(self.fc1(x2)) + self.upx4(self.fc2(x3))
        # x2_fused = self.downx2(x1) + x2 + self.upx2(self.fc3(x3))
        # x3_fused = self.downx4(self.downx2(x1)) + self.downx4(x2) + x3

        x1_fused = x1 + self.fc1(self.upx2(x2)) + self.fc2(self.upx4(x3))
        x2_fused = self.downx2(x1) + x2 + self.fc3(self.upx2(x3))
        x3_fused = self.downx4(self.downx2(x1)) + self.downx4(x2) + x3

        # x1 = self.spatial_attention(x1_fused)
        # x2 = self.spatial_attention(x2_fused)
        # x3 = self.spatial_attention(x3_fused)
        return x1_fused, x2_fused, x3_fused


class NewSkipConnection(nn.Module):
    def __init__(self, dim=[64, 128, 256]):
        super().__init__()
        # stage 1
        self.stage1 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Sequential(
                    nn.Conv3d(
                        dim[0],
                        dim[0],
                        kernel_size=(3, 3, 3),
                        stride=(2, 2, 2),
                        padding=(1, 1, 1),
                        groups=dim[0],
                        bias=False),
                    nn.BatchNorm3d(dim[0]),
                    nn.Conv3d(
                        dim[0],
                        dim[1],
                        kernel_size=(1, 1, 1)),
                    nn.BatchNorm3d(dim[1])
                ),
                nn.Sequential(
                    nn.Conv3d(
                        dim[0],
                        dim[0],
                        kernel_size=(5, 5, 5),
                        stride=(4, 4, 4),
                        padding=(2, 2, 2),
                        groups=dim[0],
                        bias=False),
                    nn.BatchNorm3d(dim[0]),
                    nn.Conv3d(
                        dim[0],
                        dim[2],
                        kernel_size=(1, 1, 1)),
                    nn.BatchNorm3d(dim[2])
                )
            ]
        )
        # stage 2
        self.stage2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        dim[1],
                        dim[0],
                        kernel_size=(1, 1, 1)
                    ),
                    nn.BatchNorm3d(dim[0]),
                    nn.Upsample(scale_factor=2, mode='trilinear')
                ),
                nn.Identity(),
                nn.Sequential(
                    nn.Conv3d(
                        dim[1],
                        dim[1],
                        kernel_size=(3, 3, 3),
                        stride=(2, 2, 2),
                        padding=(1, 1, 1),
                        groups=dim[1],
                        bias=False
                    ),
                    nn.BatchNorm3d(dim[1]),
                    nn.Conv3d(
                        dim[1],
                        dim[2],
                        kernel_size=(1, 1, 1)
                    ),
                    nn.BatchNorm3d(dim[2])
                )
            ]
        )
        # stage 3
        self.stage3 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        dim[2],
                        dim[0],
                        kernel_size=(1, 1, 1)
                    ),
                    nn.BatchNorm3d(dim[0]),
                    nn.Upsample(scale_factor=4, mode='trilinear')
                ),
                nn.Sequential(
                    nn.Conv3d(
                        dim[2],
                        dim[1],
                        kernel_size=(1, 1, 1)
                    ),
                    nn.BatchNorm3d(dim[1]),
                    nn.Upsample(scale_factor=2, mode='trilinear')
                ),
                nn.Identity()
            ]
        )
        # act
        self.act = nn.GELU()

        self.se1 = SqueezeExcite3D(dim[0])
        self.se2 = SqueezeExcite3D(dim[1])
        self.se3 = SqueezeExcite3D(dim[2])

    def forward(self, x1, x2, x3):
        x1 = self.se1(x1)
        x2 = self.se2(x2)
        x3 = self.se3(x3)

        x1_fused = self.act(self.stage1[0](x1) + self.stage2[0](x2) + self.stage3[0](x3))
        x2_fused = self.act(self.stage1[1](x1) + self.stage2[1](x2) + self.stage3[1](x3))
        x3_fused = self.act(self.stage1[2](x1) + self.stage2[2](x2) + self.stage3[2](x3))
        return x1_fused, x2_fused, x3_fused


class DecoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv_1 = Conv3d_BN(dim, dim // 2, ks=3, pad=1)
        self.conv_2 = Conv3d_BN(dim // 2, dim // 2, ks=3, pad=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = self.act(self.conv_1(torch.cat((x1, x2), dim=1)))
        x = self.act(self.conv_2(x))
        return x


class MultiScaleCascadeTrans(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 patch_size,
                 window_size,
                 embed_dim=None,
                 key_dim=None,
                 depths=[2, 2, 2, 2],
                 num_heads=None,
                 norm_layer=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = [64, 128, 256, 512]
        if key_dim is None:
            key_dim = [16, 16, 16, 16]
        if num_heads is None:
            num_heads = [4, 4, 4, 4]
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]

        # Encoder
        self.patch_embed = PatchEmbed3D(patch_size=patch_size,
                                        in_channels=in_channels,
                                        embed_dim=(32, 64),
                                        norm_layer=norm_layer)

        self.encoder_1 = Cascade_stage(dim=embed_dim[0], key_dim=key_dim[0], num_heads=num_heads[0],
                                       attn_ratio=attn_ratio[0], window_size=window_size, depths=depths[0])
        self.encoder_2 = Cascade_stage(dim=embed_dim[1], key_dim=key_dim[1], num_heads=num_heads[1],
                                       attn_ratio=attn_ratio[1], window_size=window_size, depths=depths[1], down=True)
        self.encoder_3 = Cascade_stage(dim=embed_dim[2], key_dim=key_dim[2], num_heads=num_heads[2],
                                       attn_ratio=attn_ratio[2], window_size=window_size, depths=depths[2], down=True)

        # Bottleneck
        self.bottleneck = Cascade_stage(dim=embed_dim[3], key_dim=key_dim[3], num_heads=num_heads[3],
                                        attn_ratio=attn_ratio[3], window_size=window_size, depths=depths[3], down=True)

        # Skip-connection
        self.skip = NewSkipConnection(dim=[64, 128, 256])

        # Decoder
        self.decoder_0 = DecoderBlock(dim=embed_dim[0])
        self.decoder_1 = DecoderBlock(dim=embed_dim[1])
        self.decoder_2 = DecoderBlock(dim=embed_dim[2])
        self.decoder_3 = DecoderBlock(dim=embed_dim[3])

        self.seg_head = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            Conv3d_BN(32, 32, ks=3, pad=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=(1, 1, 1))
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x0, x1 = self.patch_embed(x)

        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)

        x4 = self.bottleneck(x3)

        x1, x2, x3 = self.skip(x1, x2, x3)

        x = self.decoder_3(x4, x3)
        x = self.decoder_2(x, x2)
        x = self.decoder_1(x, x1)
        x = self.decoder_0(x, x0)

        x = self.seg_head(x)
        return x


if __name__ == '__main__':
    from thop import profile, clever_format
    model = MultiScaleCascadeTrans(in_channels=1, num_classes=1, patch_size=(4, 4, 4), window_size=(7, 7, 7), depths=[1, 1, 1, 1])
    in_ = torch.rand(1, 1, 128, 128, 128)
    macs, params = profile(model, inputs=(in_, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('macs:{}'.format(macs))
    print('params:{}'.format(params))