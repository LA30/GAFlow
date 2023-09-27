

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
import einops
from itertools import repeat
import collections.abc
from natten.functional import natten2dav, natten2dqkrpb

from utils.utils import coords_grid, bilinear_sampler
from model.cuda import Crop_Rev_cu


class GCL(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 depth=6,
                 num_head=8,
                 window_size=7,
                 neig_win_num=1,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 args=None):
        super().__init__()

        self.num_feature = embed_dim
        self.num_head = num_head

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            GCLBlock(
                dim=self.num_feature,
                num_heads=self.num_head,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                args=args)
            for i in range(depth)])

        self.norm = norm_layer(self.num_feature)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        x = inputs.permute(0, 2, 3, 1).contiguous()
        
        x = x.view(B, H*W, C)
        for f in self.blocks:
            x = f(x, inputs.shape)

        x = self.norm(x)
        out = x.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous()
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GCLBlock(nn.Module):
    def __init__(self, dim, num_heads, sc=1.,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 args=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.drop_path = nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.nat = GA(dim=dim, kernel_size=13, dilation=3, num_heads=num_heads)
        
    def _func_attn(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # B H W C
        x = self.nat(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, *inputs):
        """ Forward function.
            input tensor size (B, H*W, C).
        """
        x, feat_shape = inputs
        B, C, H, W = feat_shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # B C H W
        
        x = self._func_attn(x)
        x = x.contiguous().view(B, C, H*W).permute(0, 2, 1).contiguous()  # B HW C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class GA(nn.Module):
    '''
        Based on Neighborhood Attention 2D Module (https://github.com/SHI-Labs/NATTEN) 
    '''
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.proj = nn.Linear(dim, dim)

        k = kernel_size
        self.h_k = (k - 1) // 2
        self.k = k
        self.sigma = 9
        self.sc = 0.1 
        self.lr_m = nn.Parameter(torch.ones(k, k) * self.sc * 0.5)

    def _func_gauss(self, attn, feat_x):
        b, h, w, c = feat_x.shape
        device = attn.device

        k = self.k
        crd_ker = torch.linspace(0, k-1, k).to(device)
        x = crd_ker.view(1, k)
        y = crd_ker.view(k, 1)
        idx_x = self.h_k
        idx_y = self.h_k
        gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * self.sigma**2))
        ker_S = gauss_kernel

        ker_S = ker_S + self.lr_m / self.sc
        attn_gauss = ker_S.view(1, 1, 1, 1, self.k**2).repeat(b, self.num_heads, h, w, 1) * attn
        return attn_gauss

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)

        attn = self._func_gauss(attn, x)
        attn = attn.softmax(dim=-1)

        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )


class GGAM(nn.Module):
    def __init__(self, args, chnn, heads=1):
        super().__init__()
        self.scale = chnn ** -0.5
        self.to_qk = nn.Conv2d(chnn, chnn * 2, 1, bias=False)
        self.to_v = nn.Conv2d(chnn, chnn, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigma = 20 
        self.dm = DGM(chnn)
        self.h_p = [0.1, 2.5, 10]
        self.h_k = self.sigma
        self.k = self.h_k * 2 + 1

    def _func_gauss(self, attn, feat_ctx, feat_mo, sigma=5): 
        attn_shape = attn.shape
        b, _, h, w = feat_ctx.shape
        device = attn.device

        crd_w = torch.linspace(0, w-1, w).to(device)
        crd_h = torch.linspace(0, h-1, h).to(device)
        x = crd_w.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        y = crd_h.view(1, h, 1, 1, 1).expand(b, h, 1, h, w) 
        idx_x = crd_w.view(1, 1, 1, 1, w).expand(b, 1, 1, h, w)
        idx_y = crd_h.view(1, 1, 1, h, 1).expand(b, 1, 1, h, w)

        thr, sc, bi = self.h_p
        gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2)) * sc + bi
        gauss_kernel[gauss_kernel < (bi+thr*sc)] = 0
        
        gauss_kernel = self.dm(feat_mo, gauss_kernel)
        gauss_kernel = einops.rearrange(gauss_kernel, 'b h w u v -> b u v h w')
        attn = gauss_kernel.view(attn_shape) * attn
        return attn

    def _func_crop(self, attn, feat_shape):
        b, c, h, w = feat_shape
        attn = attn.permute(0, 1, 3, 2).contiguous()

        r = self.h_k
        coords = coords_grid(b, h, w, device=attn.device)
        coords = coords.permute(0, 2, 3, 1).contiguous()
        batch, h1, w1, _ = coords.shape

        corr = attn.view(b*h*w, 1, h, w)
        dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
        dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

        centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        corr = corr.permute(0, 1, 3, 2).contiguous()
        corr = corr.view(batch, h1, w1, -1)
        attn = torch.einsum('b h w k -> b k h w', corr).contiguous()
        attn_crop = attn
        attn_crop = attn_crop.softmax(dim=1)  # -1

        coords = coords_grid(b, h, w, device=attn.device)
        coords = torch.einsum('b p n -> b n p', coords.view(b, 2, h*w)).contiguous()

        attn_rev = torch.einsum('b k h w -> b h w k', attn_crop).contiguous().view(b, h*w, self.k, self.k)
        attn_rev = Crop_Rev_cu(attn_rev, coords, [self.h_k, self.h_k], [h, w])
        attn_rev = attn_rev.contiguous().view(b, h, w, h*w) 
        attn_hw = attn_rev
        attn_hw = attn_hw.view(b, h*w, h*w) 
        return attn_hw

    def forward(self, *inputs):
        feat_ctx, feat_mo, itr = inputs
        feat_shape = feat_mo.shape
        b, c, h, w = feat_shape
        
        if itr == 0:
            feat_q, feat_k = self.to_qk(feat_ctx).chunk(2, dim=1)
            feat_q = self.scale * feat_q.view(b, c, h*w)
            feat_k = feat_k.view(b, c, h*w)
            attn = torch.einsum('b c n, b c m -> b m n', feat_q, feat_k)
            attn = attn.view(b, 1, h*w, h*w)

            attn = self._func_gauss(attn, feat_ctx, feat_mo, self.sigma)
            self.attn = self._func_crop(attn, feat_shape)

        feat_v = self.to_v(feat_mo).view(b, c, h*w)
        feat_o = torch.einsum('b n m, b c m -> b c n', self.attn, feat_v).contiguous().view(b, c, h, w)
        feat_o = feat_mo + feat_o * self.gamma
        return feat_o


class DGM(nn.Module):
    def __init__(self, chnn_in, chnn = 32, sc=100):
        super().__init__()
        self.sc = sc
        self.C_in = nn.Conv2d(1, chnn, kernel_size=1, stride=1, padding=0)
        self.C_ofs = nn.Sequential(
            nn.Conv2d(chnn, chnn, 5, 1, 2, groups=chnn, padding_mode='replicate'),
            nn.ReLU(inplace=True), 
            nn.Conv2d(chnn, 2, 1, 1, 0, bias=False))
        self.C_a = nn.Conv2d(chnn, 1, kernel_size=1)
        self.lam = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def _func_ref(self, h, w, b, dtype, device):
        x_h = torch.linspace(0.5, h-0.5, h, dtype=dtype, device=device)
        x_w = torch.linspace(0.5, w-0.5, w, dtype=dtype, device=device)
        ref_y, ref_x = torch.meshgrid(x_h, x_w)
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(w).mul_(2).sub_(1)
        ref[..., 0].div_(h).mul_(2).sub_(1)
        return ref.unsqueeze(0).repeat(b, 1, 1, 1)

    def forward(self, feat, gauss):
        gauss = einops.rearrange(gauss, 'b u v h w -> b h w u v')
        b, _, h, w = feat.shape
        gauss_shape = gauss.shape
        dtype, device = feat.dtype, feat.device
        
        feat = feat.mean(1, True)
        feat_in = self.C_in(feat)
        feat_a = feat_in
        feat_in = einops.rearrange(feat_in, 'b (g c) h w -> (b g) c h w', g=1, c=feat_in.shape[1])
        ofs = self.C_ofs(feat_in) 
        
        ofs = ofs.tanh().mul(torch.tensor([1.0/h, 1.0/w], device=device).reshape(1, 2, 1, 1))
        ofs = einops.rearrange(ofs, 'b p h w -> b h w p')
        ref = self._func_ref(h, w, b, dtype, device)
        ofs = self.sc * ofs
        pos = ofs + ref
        pos[pos < -1] = -1
        pos[pos > 1] = 1
        gauss_d = F.grid_sample(input=gauss.view(b, -1, h, w), grid=pos[..., (1, 0)], mode='bilinear', align_corners=True)

        feat_a = self.C_a(feat_a)
        gauss_a = 1 + self.lam * feat_a
        gauss_a = gauss_a.view(b, 1, 1, h, w).expand(b, h, w, h, w)
        return gauss_d.view(gauss_shape) * gauss_a


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
