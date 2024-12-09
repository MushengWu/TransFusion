from operator import mul
from functools import reduce, lru_cache

import torch


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    nD = D // window_size[0]
    nH = H // window_size[1]
    nW = W // window_size[2]

    # B,D,H,W,C --> B,nD,d,nH,h,nW,w,C --> B,nD,nH,nW,d,h,w,C --> B*nD*nH*nW,d,h,w,C --> B*nD*nH*nW,C,d,h,w
    x = x.view(B, nD, window_size[0], nH, window_size[1], nW, window_size[2], C).permute(
        0, 1, 3, 5, 2, 4, 6, 7).reshape(B*nD*nH*nW, *window_size, C).permute(0, 4, 1, 2, 3)
    return x


def window_reverse(x, window_size, B, C, D, H, W):
    nD = D // window_size[0]
    nH = H // window_size[1]
    nW = W // window_size[2]

    # B*nD*nH*nW,C,d,h,w --> B*nD*nH*nW,d,h,w,C --> B,nD,nH,nW,d,h,w,C --> B,nD,d,nH,h,nW,w,C --> B,D,H,W,C
    x = x.permute(0, 2, 3, 4, 1).view(B, nD, nH, nW, *window_size, C).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(
        B, nD*window_size[0], nH*window_size[1], nW*window_size[2], C)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def computer_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, W, H, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, reduce(mul, window_size))
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
