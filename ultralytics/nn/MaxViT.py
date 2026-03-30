"""MaxViT block — native implementation for YOLO integration.

Implements the core MaxViT architecture from:
  'MaxViT: Multi-Axis Vision Transformer' (Tu et al., 2022)
  https://arxiv.org/abs/2204.01697

Structure per block:
  1. MBConv (depthwise ConvNeXt stem)   — local feature extraction
  2. Window self-attention + FFN        — local attention
  3. Grid (dilated) self-attention + FFN — sparse global attention

Adaptive padding ensures any input spatial size works regardless of window size.
No timm dependency — fully self-contained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _PreNormMHSA(nn.Module):
    """Pre-norm multi-head self-attention with residual connection."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.norm      = nn.LayerNorm(dim)
        self.qkv       = nn.Linear(dim, dim * 3, bias=False)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, L, C)  →  (N, L, C)  with residual."""
        skip = x
        x = self.norm(x)
        N, L, C = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(N, L, C)
        return skip + self.proj(x)


class _PreNormFFN(nn.Module):
    """Pre-norm feed-forward network with residual connection."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Partition helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pad_to_window(x: torch.Tensor, wh: int, ww: int):
    """Pad spatial dims so H % wh == 0 and W % ww == 0."""
    _, _, H, W = x.shape
    ph = (wh - H % wh) % wh
    pw = (ww - W % ww) % ww
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, H, W


def _window_partition(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    """(B, H, W, C) → (B * nW, wh*ww, C)  — local windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh * ww, C)


def _window_unpartition(x: torch.Tensor, wh: int, ww: int, B: int, H: int, W: int) -> torch.Tensor:
    x = x.view(B, H // wh, W // ww, wh, ww, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _grid_partition(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    """(B, H, W, C) → (B * nG, wh*ww, C)  — dilated grid (sparse global)."""
    B, H, W, C = x.shape
    x = x.view(B, wh, H // wh, ww, W // ww, C)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, wh * ww, C)


def _grid_unpartition(x: torch.Tensor, wh: int, ww: int, B: int, H: int, W: int) -> torch.Tensor:
    x = x.view(B, H // wh, W // ww, wh, ww, -1)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)


# ─────────────────────────────────────────────────────────────────────────────
# MaxViT block
# ─────────────────────────────────────────────────────────────────────────────

class MaxViTCNNBlock(nn.Module):
    """MaxViT block: MBConv stem + window attention + grid attention.

    Args:
        c1          : number of input channels  (from YOLO parse_model)
        c2          : number of output channels (from YOLO parse_model, equals c1)
        window_size : (wh, ww) attention window size (default [8, 8])
        stride      : reserved for API compatibility, always 1

    Note:
        parse_model rewrites the YAML ``[512, [16,16]]`` args into
        ``[c1, c2_scaled, [16,16]]`` before calling the constructor, so the
        constructor must accept **(c1, c2, window_size)** — not (ch, window_size).
    """

    def __init__(self, c1: int, c2: int, window_size=None, stride: int = 1):
        super().__init__()

        ch = c2  # output channels (c1 == c2 for this block in all standard configs)

        if window_size is None:
            window_size = [8, 8]
        if isinstance(window_size, int):
            window_size = [window_size, window_size]
        self.wh, self.ww = int(window_size[0]), int(window_size[1])

        # optional 1×1 projection when c1 ≠ c2
        self.proj_in = nn.Conv2d(c1, ch, 1) if c1 != ch else nn.Identity()

        # 1 attention head per 64 channels (min 1)
        num_heads = max(1, ch // 64)

        # ── MBConv (ConvNeXt-style depthwise stem) ────────────────────────
        self.mbconv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch),   # depthwise 7×7
            nn.GroupNorm(1, ch),
            nn.Conv2d(ch, ch * 4, kernel_size=1),                      # expand
            nn.GELU(),
            nn.Conv2d(ch * 4, ch, kernel_size=1),                      # squeeze
        )

        # ── Window attention (local) ──────────────────────────────────────
        self.window_attn = _PreNormMHSA(ch, num_heads)
        self.window_ffn  = _PreNormFFN(ch)

        # ── Grid attention (sparse global) ────────────────────────────────
        self.grid_attn = _PreNormMHSA(ch, num_heads)
        self.grid_ffn  = _PreNormFFN(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 0. Optional input projection (when c1 != c2) ─────────────────
        x = self.proj_in(x)

        # ── 1. MBConv with residual ───────────────────────────────────────
        x = x + self.mbconv(x)

        # Pad so spatial dims are divisible by window size
        x, H0, W0 = _pad_to_window(x, self.wh, self.ww)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)          # → (B, H, W, C)

        # ── 2. Window self-attention (local) ─────────────────────────────
        tokens = _window_partition(x, self.wh, self.ww)
        tokens = self.window_attn(tokens)
        tokens = self.window_ffn(tokens)
        x = _window_unpartition(tokens, self.wh, self.ww, B, H, W)

        # ── 3. Grid self-attention (sparse global) ────────────────────────
        tokens = _grid_partition(x, self.wh, self.ww)
        tokens = self.grid_attn(tokens)
        tokens = self.grid_ffn(tokens)
        x = _grid_unpartition(tokens, self.wh, self.ww, B, H, W)

        # Unpad and restore (B, C, H, W)
        return x[:, :H0, :W0, :].permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def test_maxvit_block():
    """Validate MaxViTCNNBlock across common YOLO feature map sizes."""
    configs = [
        # (channels, window_size, spatial_size,  description)
        (128,  [8,  8],  (40, 40), "640px P4  scale-n"),
        (256,  [8,  8],  (20, 20), "640px P5  scale-n"),
        (512,  [8,  8],  (40, 40), "640px P4  scale-l"),
        (1024, [8,  8],  (20, 20), "640px P5  scale-l"),
        (128,  [16, 16], (16, 16), "stride-detect 256px"),
        (256,  [8,  8],  (8,  8),  "stride-detect 256px"),
        (512,  [16, 16], (21, 21), "odd spatial (tests padding)"),
    ]

    for ch, ws, hw, desc in configs:
        print(f"  {desc}: ch={ch}, window={ws}, spatial={hw}")
        block = MaxViTCNNBlock(ch, ch, ws)  # c1=c2=ch  (as parse_model provides)
        x = torch.randn(1, ch, hw[0], hw[1])
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print(f"    {tuple(x.shape)} → {tuple(out.shape)} ✅")

    print("\n✅ All MaxViT tests passed!")


if __name__ == "__main__":
    test_maxvit_block()
