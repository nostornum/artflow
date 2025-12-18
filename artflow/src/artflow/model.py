import math

from abc import ABC, abstractmethod
from functools import cache
from typing import Dict, Literal, Tuple, cast

import einops
import einx
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as attention


@cache
def div(n: int, m: int) -> int:
    for d in range(m, n + 1):
        if n % d == 0:
            return d
    return n


def basic_init(module: nn.Module) -> None:
    if not isinstance(module, nn.Linear):
        return
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    torch.nn.init.xavier_uniform_(module.weight)


def patch_init(module: nn.Module) -> None:
    if not isinstance(module, nn.Conv2d):
        return
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    nn.init.xavier_uniform_(module.weight.data.view([module.weight.data.shape[0], -1]))


def adaln_init(module: nn.Module) -> None:
    if not isinstance(module, AdaLN):
        return
    if module.adaLN[-1].bias is not None:
        nn.init.zeros_(cast(nn.Linear, module.adaLN[-1]).bias)
    nn.init.zeros_(cast(nn.Linear, module.adaLN[-1]).weight)


def precompute_freqs_cis_2d(d: int, h: int, w: int, theta: float = 10000.0, scale=16.0):
    x_pos = torch.linspace(0, scale, w)  # type: ignore
    y_pos = torch.linspace(0, scale, h)  # type: ignore
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, d, 4)[: (d // 4)].float() / d))  # Hc/4
    x_frq = torch.outer(x_pos, freqs).float()  # N Hc/4
    y_frq = torch.outer(y_pos, freqs).float()  # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_frq), x_frq)
    y_cis = torch.polar(torch.ones_like(y_frq), y_frq)
    freqx = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)  # N,Hc/4,2
    freqx = freqx.reshape(h * w, -1)
    return freqx


def apply_rotary_emb_cross(xq: torch.Tensor, xk: torch.Tensor, freq_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    freq_cis = freq_cis[None, None, :, :] if freq_cis.ndim < 4 else freq_cis
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freq_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freq_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Router(nn.Module, ABC):
    Kind = Literal["PASS", "TREAD", "SPRINT"]

    def __init__(
        self,
        *,
        depth_init: int,
        depth_term: int,
        rate: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.depth_init: int = depth_init
        self.depth_term: int = depth_term
        self.rate: float = rate

    def skip(self, *, i: int) -> bool:
        # Skip computation if drop rate is 100%
        return not (i < self.depth_init or i > self.depth_term or self.rate != 1)

    @abstractmethod
    def mask(self, *, x: Tensor, h: int, w: int) -> Tensor:
        # x: [B, S, D] -> m: [B, S]
        raise NotImplementedError()

    @abstractmethod
    def drop(self, *, x: Tensor, m: Tensor) -> Tensor:
        # x: [B, S0, D], m: [B, S1, D] -> x': [B, S1, D]
        raise NotImplementedError()

    @abstractmethod
    def fuse(self, *, x: Tensor, m: Tensor, x_k: Tensor) -> Tensor:
        # x: [B, S1, D], m: [B, S1, D], x_o: [B, S0, D] -> x': [B, S0, D]
        raise NotImplementedError()

    @classmethod
    def create(cls, kind: Kind, **kwargs) -> "Router":
        match kind:
            case "PASS":
                return PASSRouter(**kwargs)
            case "TREAD":
                return TREADRouter(**kwargs)
            case "SPRINT":
                return SPRINTRouter(**kwargs)


class PASSRouter(Router):
    """
    PASS: Router that just passes its' input to the output
    """

    def __init__(
        self,
        *,
        depth_init: int = 2,
        depth_term: int = 10,
        rate: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            depth_init=depth_init,
            depth_term=depth_term,
            rate=rate,
        )

    def mask(self, *, x: Tensor, h: int, w: int) -> Tensor:
        return torch.arange(h * w).expand(x.size(0), -1)

    def drop(self, *, x: Tensor, m: Tensor) -> Tensor:
        m = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = x.gather(dim=1, index=m)
        return x

    def fuse(self, *, x: Tensor, m: Tensor, x_k: Tensor) -> Tensor:
        m = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = x_k.scatter(dim=1, index=m, src=x)
        return x


class TREADRouter(Router):
    """
    TREAD: Token Routing for Efficient DiT Training
    Paper: https://arxiv.org/abs/2501.04765
    """

    def __init__(
        self,
        *,
        depth_init: int = 2,
        depth_term: int = 10,
        rate: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            depth_init=depth_init,
            depth_term=depth_term,
            rate=rate,
        )

    def mask(self, *, x: Tensor, h: int, w: int) -> Tensor:
        B, S, _ = x.size()
        mask = torch.rand((B, S), device=x.device)
        mask = torch.argsort(mask, dim=1)
        num_mask = math.floor(S * self.rate)
        num_keep = S - num_mask
        ids_keep = mask[:, :num_keep]
        ids_keep = torch.sort(ids_keep, dim=1).values
        ids_keep = ids_keep
        return ids_keep

    def drop(self, *, x: Tensor, m: Tensor) -> Tensor:
        m = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = x.gather(dim=1, index=m)
        return x

    def fuse(self, *, x: Tensor, m: Tensor, x_k: Tensor) -> Tensor:
        m = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = x_k.scatter(dim=1, index=m, src=x)
        return x


class SPRINTRouter(Router):
    """
    SPRINT: Sparse-Dense REsidual Fusion for Efficient Diffusion Transformers
    Paper: https://arxiv.org/abs/2510.21986
    """

    def __init__(
        self,
        *,
        n: int,
        k: int,
        dim: int,
        dim_s: int,
        dim_d: int,
        depth_init: int = 2,
        depth_term: int = 10,
        **kwargs,
    ) -> None:
        assert 1 <= n, f"n must be greater or equal to 1, got instead n={n}"
        assert 0 <= k <= n**2, f"k must be greater than zero and at most {n**2}, got instead k={k}"
        super().__init__(depth_init=depth_init, depth_term=depth_term, rate=min(1, max(0, 1 - k / n**2)))
        self.projection = nn.Linear(dim_d + dim_s, dim, bias=False)
        self.mask_token = nn.Parameter(torch.randn(dim_s))
        self.n: int = n

    def mask(self, *, x: Tensor, h: int, w: int) -> Tensor:
        B, S, _ = x.size()
        H, W = h, w

        # Adjust split based on aspect ratio
        r: float = W / H
        M: int = div(W, math.ceil(self.n * max(1, r)))
        N: int = div(H, math.ceil(self.n * max(1, r**-1)))

        # Group partition
        H_G: int = H // N
        W_G: int = W // M
        S_G: int = H_G * W_G

        # Amount of tokens per-partition
        num_mask: int = math.floor(self.rate * S_G)
        num_keep: int = S_G - num_mask

        # Keep indices of random tokens per-partition
        mask = torch.rand((B, N * M, S_G), device=x.device)
        mask = torch.argsort(mask, dim=-1)[..., :num_keep]

        # Compute absolute indices per-sequence of tokens
        ids_keep = torch.arange(S, device=x.device).expand(B, -1)
        ids_keep = cast(Tensor, einx.rearrange("b ((h_n h_g) (w_m w_g)) -> b (h_n w_m) (h_g w_g)", ids_keep, h_n=N, h_g=H_G, w_m=M, w_g=W_G))
        ids_keep = torch.gather(ids_keep, dim=-1, index=mask)
        ids_keep = ids_keep.flatten(1).sort(dim=-1).values
        return ids_keep

    def drop(self, *, x: Tensor, m: Tensor) -> Tensor:
        m = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = x.gather(dim=1, index=m)
        return x

    def fuse(self, *, x: Tensor, m: Tensor, x_k: Tensor) -> Tensor:
        B, S, _ = x_k.size()
        m = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x_sd = self.mask_token.expand(B, S, -1).type_as(x).scatter(dim=1, index=m, src=x)
        x_ds = x_k
        x_fuse = torch.cat([x_sd, x_ds], dim=-1)
        x_fuse = self.projection.forward(x_fuse)
        return x_fuse


class SwiGLU(nn.Module):
    """
    SwiGLU from "GLU Variants Improve Transformer"
    Paper: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim: int, dim_hidden: int, dim_out: int | None = None, bias: bool = True) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, dim_hidden, bias=bias)
        self.w_silu = nn.Linear(dim, dim_hidden, bias=bias)
        self.w_proj = nn.Linear(dim_hidden, dim_out if dim_out else dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        h1: Tensor = self.w_silu(x)
        h2: Tensor = self.w_gate(x)
        h3: Tensor = self.w_proj(F.silu(h1) * h2)
        return h3


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, bias: bool = True, scale: float = 2 / 3) -> None:
        super().__init__()
        self.mlp = SwiGLU(dim, dim_hidden=int(scale * dim_hidden), bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class Embed(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_embed: int,
        bias: bool = True,
        norm: Literal["RMSNorm"] | None = None,
    ) -> None:
        super().__init__()
        self.proj: nn.Module = nn.Linear(dim_input, dim_embed, bias=bias)
        self.norm: nn.Module = nn.RMSNorm(dim_embed) if norm == "RMSNorm" else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_embed: int,
        patch_size: int,
        bias: bool = True,
        norm: Literal["RMSNorm"] | None = None,
    ) -> None:
        super().__init__()
        self.proj: nn.Module = nn.Conv2d(dim_input, dim_embed, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm: nn.Module = nn.RMSNorm(dim_embed) if norm == "RMSNorm" else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        return x


class TimestepEmbedder(nn.Module):
    """
    Positional Encodings from "Attention Is All You Need"
    Paper: https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        dim_freq: int,
        dim_hidden: int,
        *,
        cycle: int = 10,
        bias: bool = True,
        theta: float = 1e4,
    ) -> None:
        super().__init__()
        self.scale: float = 2 * math.pi * cycle
        self.dim_freq: int = dim_freq // 2
        self.theta: float = theta
        self.mlp = nn.Sequential(
            *[
                nn.Linear(dim_freq, dim_hidden, bias=bias),
                nn.SiLU(),
                nn.Linear(dim_hidden, dim_hidden, bias=bias),
            ]
        )

    @staticmethod
    def timestep_embedding(t: Tensor, theta: float, scale: float, dim: int) -> Tensor:
        freqs: Tensor = torch.exp(-math.log(theta) * torch.arange(dim, dtype=torch.float32, device=t.device) / dim)
        freqs = torch.outer(t, freqs) * scale
        freqs_sin, freqs_cos = torch.sin(freqs), torch.cos(freqs)
        freqs_embeds: Tensor = einx.rearrange("... (d 1), ... (d 1) -> ... (d 1+1)", freqs_sin, freqs_cos)  # type: ignore
        return freqs_embeds

    def forward(self, t: Tensor) -> Tensor:
        t_freq: Tensor = self.timestep_embedding(t.flatten(), self.theta, self.scale, self.dim_freq)
        t_emb: Tensor = self.mlp(t_freq)
        return t_emb


class NerfEmbedder(nn.Module):
    """
    PixNerd: Pixel Neural Field Diffusion
    Paper: https://arxiv.org/abs/2507.23268
    """

    def __init__(self, dim_input: int, dim_hidden: int, max_freq: int, bias: bool = True) -> None:
        super().__init__()
        self.max_freq: int = max_freq
        self.posdicts: Dict[Tuple[int, int], Tensor] = dict()
        self.embedder = nn.Linear(dim_input + max_freq**2, dim_hidden, bias=bias)

    def fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos: Tensor = self.posdicts.setdefault((patch_size, patch_size), precompute_freqs_cis_2d(self.max_freq**2 * 2, patch_size, patch_size)).to(device) if (patch_size, patch_size) not in self.posdicts else self.posdicts[(patch_size, patch_size)].to(device)
        pos: Tensor = pos[None, :, :].to(device=device, dtype=dtype)
        return pos

    def forward(self, x: Tensor) -> Tensor:
        B, P2, _ = x.size()
        x_dct: Tensor = self.fetch_pos(patch_size=int(P2**0.5), device=x.device, dtype=x.dtype)
        x = torch.cat([x, x_dct.repeat(B, 1, 1)], dim=-1)
        x = self.embedder(x)
        return x


class TAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int, qk_norm: bool = True, qkv_bias: bool = False) -> None:
        super().__init__()
        self.q_norm: nn.RMSNorm | nn.Identity = nn.RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.k_norm: nn.RMSNorm | nn.Identity = nn.RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.W_qkv = nn.Linear(dim, 3 * dim_head * heads, qkv_bias)
        self.W_out = nn.Linear(dim_head * heads, dim)
        self.heads: int = heads

    def forward(self, x: Tensor) -> Tensor:
        qkv: Tuple[Tensor, ...] = torch.chunk(self.W_qkv(x), chunks=3, dim=-1)
        q: Tensor = qkv[0]
        k: Tensor = qkv[1]
        v: Tensor = qkv[2]

        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.heads)
        k = einops.rearrange(k, "b s (h d) -> b h s d", h=self.heads)
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.heads)

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())

        A = attention(q, k, v, attn_mask=None, dropout_p=0.0)
        A = einops.rearrange(A, "b h s d -> b s (h d)")
        o: Tensor = self.W_out(A)
        return o


class RAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int, qk_norm: bool = True, qkv_bias: bool = False) -> None:
        super().__init__()
        self.q_norm: nn.Module = nn.RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = nn.RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.W_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=qkv_bias)
        self.W_out = nn.Linear(heads * dim_head, dim)
        self.heads: int = heads

    def forward(self, x: Tensor, freq_cis: Tensor) -> Tensor:
        qkv: Tuple[Tensor, ...] = torch.chunk(self.W_qkv(x), chunks=3, dim=-1)
        q: Tensor = qkv[0]
        k: Tensor = qkv[1]
        v: Tensor = qkv[2]

        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.heads)
        k = einops.rearrange(k, "b s (h d) -> b h s d", h=self.heads)
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.heads)

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())
        q_rot, k_rot = apply_rotary_emb_cross(q, k, freq_cis)

        A = attention(q_rot, k_rot, v, attn_mask=None, dropout_p=0.0)
        A = einops.rearrange(A, "b h s d -> b s (h d)")
        o: Tensor = self.W_out(A)
        return o


class CAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int, qk_norm: bool = True, qkv_bias: bool = False) -> None:
        super().__init__()
        self.q_norm: nn.Module = nn.RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = nn.RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.W_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=qkv_bias)
        self.W_kvy = nn.Linear(dim, 2 * heads * dim_head, bias=qkv_bias)
        self.W_out = nn.Linear(heads * dim_head, dim)
        self.heads: int = heads

    def forward(self, x: Tensor, c: Tensor, freq_cis: Tensor) -> Tensor:
        qx, kx, vx = torch.chunk(self.W_qkv(x), 3, dim=-1)
        kx = einops.rearrange(kx, "b s (h d) -> b h s d", h=self.heads)
        vx = einops.rearrange(vx, "b s (h d) -> b h s d", h=self.heads)
        qx = einops.rearrange(qx, "b s (h d) -> b h s d", h=self.heads)

        qx = self.q_norm(qx.contiguous())
        kx = self.k_norm(kx.contiguous())
        qx, kx = apply_rotary_emb_cross(qx, kx, freq_cis)

        ky, vy = torch.chunk(self.W_kvy(c), 2, dim=-1)
        ky = einops.rearrange(ky, "b s (h d) -> b h s d", h=self.heads)
        vy = einops.rearrange(vy, "b s (h d) -> b h s d", h=self.heads)
        ky = self.k_norm(ky.contiguous())

        q = qx
        k = torch.cat([kx, ky], dim=2)
        v = torch.cat([vx, vy], dim=2)

        A = attention(q, k, v, attn_mask=None, dropout_p=0.0)
        A = einops.rearrange(A, "b h s d -> b s (h d)")
        o = self.W_out(A)
        return o


class AdaLN(nn.Module):
    """
    AdaLN from "Scalable Diffusion Models with Transformers"
    Paper: "https://arxiv.org/abs/2212.09748"
    """

    def __init__(self, dim_hidden: int, num_chunks: int, bias: bool = True, activation: bool = True) -> None:
        super().__init__()
        self.num_chunks: int = num_chunks
        self.adaLN = nn.Sequential(
            *[
                nn.SiLU() if activation else nn.Identity(),
                nn.Linear(dim_hidden, self.num_chunks * dim_hidden, bias=bias),
            ]
        )

    @staticmethod
    def modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        return x * (1.0 + scale) + shift

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return torch.chunk(self.adaLN(x), chunks=self.num_chunks, dim=-1)


class TextRefineBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, adaln_activ: bool = False) -> None:
        super().__init__()
        self.adaLN_modulation = AdaLN(dim_hidden=dim, num_chunks=6, activation=adaln_activ)
        self.msa_norm = nn.RMSNorm(dim)
        self.attn = TAttention(dim, dim // heads, heads)
        self.mlp_norm = nn.RMSNorm(dim)
        self.ffwd = FeedForward(dim, dim_hidden=int(dim * mlp_ratio))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.adaLN_modulation(c)
        x = x + gate_msa * self.attn(AdaLN.modulate(self.msa_norm(x), scale=scale_msa, shift=shift_msa))
        x = x + gate_mlp * self.ffwd(AdaLN.modulate(self.mlp_norm(x), scale=scale_mlp, shift=shift_mlp))
        return x


class DitCrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, adaln_activ: bool = False) -> None:
        super().__init__()
        self.adaLN_modulation = AdaLN(dim, num_chunks=6, activation=adaln_activ)
        self.msa_norm = nn.RMSNorm(dim)
        self.attn = CAttention(dim, dim_head=dim // heads, heads=heads)
        self.mlp_norm = nn.RMSNorm(dim)
        self.ffwd = FeedForward(dim, dim_hidden=int(dim * mlp_ratio))

    def forward(self, x: Tensor, c: Tensor, t: Tensor, r: Tensor) -> Tensor:
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.adaLN_modulation(t)
        x = x + gate_msa * self.attn(AdaLN.modulate(self.msa_norm(x), scale=scale_msa, shift=shift_msa), c=c, freq_cis=r)
        x = x + gate_mlp * self.ffwd(AdaLN.modulate(self.mlp_norm(x), scale=scale_mlp, shift=shift_mlp))
        return x


class DitOutput(nn.Module):
    def __init__(self, dim: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(dim, dim_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.proj(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim: int, bias: bool = True) -> None:
        super().__init__()
        self.adaln = AdaLN(dim, num_chunks=3, bias=bias)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffwd = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaln.forward(c)
        x = x + gate_mlp * self.ffwd(AdaLN.modulate(self.norm(x), scale=scale_mlp, shift=shift_mlp))
        return x


class PIXHead(nn.Module):
    def __init__(self, dim_input: int, dim_hidden: int, dim_condt: int, dim_output: int, num_res_blocks: int, patch_size: int) -> None:
        super().__init__()
        self.p_size: int = patch_size
        self.x_proj = nn.Linear(dim_input, dim_hidden)
        self.c_proj = nn.Linear(dim_condt, dim_hidden * self.p_size**2)
        self.layers = nn.ModuleList([ResBlock(dim_hidden) for _ in range(num_res_blocks)])
        self.output = DitOutput(dim_hidden, dim_output)
        self.init_weights()

    def init_weights(self) -> None:
        self.apply(basic_init)

        for layer in self.layers:
            layer.apply(adaln_init)

        nn.init.zeros_(self.output.proj.weight)
        nn.init.zeros_(self.output.proj.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.x_proj(x)
        c = self.c_proj(c)
        c = c.reshape(c.size(0), self.p_size**2, -1)

        for layer in self.layers:
            x = layer(x=x, c=c)

        x = self.output(x)
        return x


class DeCo(nn.Module):
    """
    DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation
    Paper: https://arxiv.org/abs/2511.19365
    """

    def __init__(
        self,
        *,
        patch_size: int = 32,
        num_heads: int = 8,
        dim_input: int = 384,
        dim_hidden_enc: int = 768,
        dim_hidden_dec: int = 768,
        dim_timestep: int = 256,
        dim_size: int = 256,
        dim_txt_emb: int = 1024,
        num_embedder_layers: int = 4,
        num_encoder_layers: int = 16,
        num_decoder_layers: int = 2,
        mlp_ratio_txt: float = 4.0,
        mlp_ratio_enc: float = 4.0,
        path_drop: float = 0.05,
        router_n: int = 2,
        router_k: int = 1,
    ) -> None:
        super().__init__()

        self.path_drop: float = path_drop
        self.patch_size: int = patch_size
        self.dim_hidden: int = dim_hidden_enc
        self.dim_size_i: int = dim_size // 2
        self.dim_head_e: int = dim_hidden_enc // num_heads

        self.b_h_embedder = TimestepEmbedder(dim_freq=self.dim_size_i, dim_hidden=self.dim_size_i)
        self.b_w_embedder = TimestepEmbedder(dim_freq=self.dim_size_i, dim_hidden=self.dim_size_i)
        self.s_embedder = nn.Linear(self.dim_size_i * 2, dim_hidden_enc)
        self.router: SPRINTRouter = SPRINTRouter(n=router_n, k=router_k, depth_init=2, dim=self.dim_hidden, dim_s=self.dim_hidden, dim_d=self.dim_hidden, depth_term=num_encoder_layers - 3)

        self.t_embedder = TimestepEmbedder(dim_freq=dim_timestep, dim_hidden=dim_hidden_enc)
        self.c_embedder = Embed(dim_input=dim_txt_emb, dim_embed=dim_hidden_enc, norm="RMSNorm")
        self.p_embedder = PatchEmbed(dim_input=dim_input, dim_embed=dim_hidden_enc, patch_size=patch_size)
        self.x_embedder = NerfEmbedder(dim_input=dim_input, dim_hidden=dim_hidden_dec, max_freq=8)

        self.embedder = nn.ModuleList([TextRefineBlock(dim_hidden_enc, num_heads, mlp_ratio_txt) for _ in range(num_embedder_layers)])
        self.encoder = nn.ModuleList([DitCrossBlock(dim_hidden_enc, num_heads, mlp_ratio_enc) for _ in range(num_encoder_layers)])
        self.decoder = PIXHead(dim_input=dim_hidden_dec, dim_hidden=dim_hidden_dec, dim_condt=dim_hidden_enc, dim_output=dim_input, num_res_blocks=num_decoder_layers, patch_size=patch_size)
        self.init_weights()

    def init_weights(self) -> None:
        # Initialize transformer layers
        self.apply(basic_init)

        # Initialize embedding MLPs
        for module in [self.t_embedder, self.b_h_embedder, self.b_w_embedder]:
            nn.init.normal_(cast(nn.Linear, module.mlp[0]).weight, std=0.02)
            nn.init.normal_(cast(nn.Linear, module.mlp[2]).weight, std=0.02)

        # Use AdaLN-Zero initialization
        for module in [self.embedder, self.encoder]:
            module.apply(adaln_init)

        # Initialize Conv2d like Linear
        for module in [self.p_embedder]:
            module.apply(patch_init)

        # Initialize decoder head
        self.decoder.init_weights()

        # Persist RoPE2d tensors
        self.posdict: Dict[Tuple[int, int], Tensor] = dict()

    def fetch_pos(self, h: int, w: int, device: torch.device) -> Tensor:
        return self.posdict.setdefault((h, w), precompute_freqs_cis_2d(self.dim_head_e, h, w)).to(device) if (h, w) not in self.posdict else self.posdict[(h, w)].to(device)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        B, C, H, W = x.size()
        P: int = self.patch_size
        E_Y: int = H // P
        E_X: int = W // P
        S: int = E_Y * E_X

        # Embed bucket target size
        b_e: Tensor = torch.tensor([H, W], device=x.device, dtype=torch.float32).expand(B, -1)
        b_s: Tuple[Tensor, Tensor] = cast(Tuple[Tensor, Tensor], einx.rearrange("b (h + w) -> (b h), (b w)", b_e, h=1, w=1))
        b_e: Tensor = self.s_embedder(torch.cat([self.b_h_embedder(b_s[0]), self.b_w_embedder(b_s[1])], dim=-1))

        # Add time and size contitions
        t_e: Tensor = self.t_embedder(t)
        t_e = t_e + b_e
        t_e = einops.rearrange(t_e, "b h -> b 1 h")

        # Activate for AdaLN
        t_c: Tensor = F.silu(t_e)

        # Text embedding
        c_s: Tensor = self.c_embedder(c)

        # Text refinement
        for layer in self.embedder:
            c_s = layer(c_s, t_c)

        # Activate for AdaLN
        c_s = F.silu(c_s)

        # Embed patches for encoder
        x_r: Tensor = self.fetch_pos(E_Y, E_X, x.device)
        x_s: Tensor = self.p_embedder(x)

        # Router state
        x_m: Tensor = torch.empty(size=[0])
        x_k: Tensor = torch.empty(size=[0])
        x_o: Tensor = x_r.clone()

        # Perform DiT layers for condition
        for i, layer in enumerate(self.encoder):
            # Select tokens for sparse-deep branch
            if i == self.router.depth_init:
                x_k: Tensor = x_s.clone()
                x_m: Tensor = self.router.mask(x=x_s, h=E_Y, w=E_X)
                x_s: Tensor = self.router.drop(x=x_s, m=x_m)
                x_r: Tensor = x_r.expand(B, -1, -1).gather(dim=1, index=x_m.unsqueeze(-1).expand(-1, -1, x_r.size(-1))).unsqueeze(1)

            # Perform DiT layer for condition
            if not self.router.skip(i=i):
                x_s: Tensor = layer(x=x_s, c=c_s, t=t_c, r=x_r)

            # Fuse sparse-deep & deep-shallow branches
            if i == self.router.depth_term:
                # Path-Drop Learning: mask random image tokens
                if self.training:
                    m_rand: Tensor = torch.rand(size=[B, 1, 1], device=x_s.device)
                    m_tokn: Tensor = self.router.mask_token.expand(B, 1, -1)
                    m_drop: Tensor = m_rand <= self.path_drop
                    x_s: Tensor = torch.where(m_drop, m_tokn, x_s)

                # Path-Drop Guidance: mask sparse-deep branch
                x_s: Tensor = self.router.mask_token.expand(B, x_s.size(1), -1) if self.router.rate == 1 else x_s

                # Fuse sd and ds branches along with text tokens
                x_s: Tensor = self.router.fuse(x=x_s, m=x_m, x_k=x_k)
                x_r: Tensor = x_o

        # Inject time and project
        x_s: Tensor = F.silu(x_s + t_e)

        # Reshape before decoder
        x_p: Tensor = x.reshape(B * S, C, P**2).transpose(1, 2)
        x_s: Tensor = x_s.reshape(B * S, self.dim_hidden)

        # Forward through decoder
        x_p: Tensor = self.x_embedder(x_p)
        x_p: Tensor = self.decoder(x_p, x_s)

        # Reshape before output
        x_p: Tensor = x_p.transpose(1, 2).reshape(B, S, -1)
        x_p: Tensor = cast(Tensor, einx.rearrange("b (h w) (c py px) -> b c (h py) (w px)", x_p, h=E_Y, w=E_X, py=P, px=P))
        return x_p
