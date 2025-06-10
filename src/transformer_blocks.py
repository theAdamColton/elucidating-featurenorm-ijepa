import math
from typing import Literal

import einx
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch import nn
from dataclasses import dataclass, field
from diffmoe.diffmoe import DiffMoeMLP


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Adapted from diffusers/src/diffusers/models/embeddings.py L697
    https://github.com/huggingface/diffusers/blob/58431f102cf39c3c8a569f32d71b2ea8caa461e1/src/diffusers/models/embeddings.py#L1176


    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([(B), S, D], [(B), S, D],)
            Where (B) dim is optional

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos, sin = freqs_cis  # [(B) S, D]

    x_real, x_imag = einx.rearrange("b h s (hd two) -> two b h s hd", x, two=2)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1)
    x_rotated = einx.rearrange("b h s hd two -> b h s (hd two)", x_rotated)

    og_dtype = x.dtype

    x = x.float()
    x_rotated = x_rotated.float()

    # (B) S D -> (B) H S D
    is_batched = cos.ndim == 3
    if is_batched:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    x = x * cos + x_rotated * sin
    x = x.to(og_dtype)

    return x


class DynTanh(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True):
        super().__init__()

        self.alpha = nn.Parameter(torch.full((hidden_size,), 0.5))

        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(hidden_size))
            self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        x = F.tanh(einx.multiply("... d, d", x, self.alpha))
        if self.elementwise_affine:
            x = einx.multiply("... d, d", x, self.gamma)
            x = einx.add("... d, d", x, self.beta)

        return x


@dataclass
class AttentionConfig:
    embed_dim: int = 64
    head_dim: int = 64
    num_attention_heads: int = 1
    should_use_qk_norm: bool = True

    qk_norm_mode: Literal["layernorm", "dyntanh"] = "layernorm"


class Attention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.proj_qkv = nn.Linear(
            config.embed_dim,
            config.head_dim * config.num_attention_heads * 3,
            bias=False,
        )

        def get_head_norm():
            if config.qk_norm_mode == "layernorm":
                return nn.LayerNorm(config.head_dim)
            elif config.qk_norm_mode == "dyntanh":
                return DynTanh(config.head_dim)
            else:
                raise ValueError(config.qk_norm_mode)

        if config.should_use_qk_norm:
            self.q_norm = get_head_norm()
            self.k_norm = get_head_norm()

        self.proj_out = nn.Linear(
            config.num_attention_heads * config.head_dim, config.embed_dim, bias=False
        )

    def forward(self, x, block_mask=None, attn_mask=None, rotary_embeds=None):
        assert not ((attn_mask is not None) and block_mask is not None)

        q, k, v = einx.rearrange(
            "b n (three h d) -> three b h n d",
            self.proj_qkv(x),
            three=3,
            h=self.config.num_attention_heads,
        )

        if self.config.should_use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rotary_embeds is not None:
            q = apply_rotary_emb(q, rotary_embeds)
            k = apply_rotary_emb(k, rotary_embeds)

        if block_mask is not None:
            # Hack for pytorch 2.6
            # Because flex attention doesn't currently work with autocast
            smallest_dtype = torch.float32
            for tensor in q, k, v:
                if tensor.dtype.itemsize < smallest_dtype.itemsize:
                    smallest_dtype = tensor.dtype
            q = q.to(smallest_dtype)
            k = k.to(smallest_dtype)
            v = v.to(smallest_dtype)

            out = flex_attention(q, k, v, block_mask=block_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = einx.rearrange("b h n d -> b n (h d)", out)
        return (self.proj_out(out), k)


class MLP(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@dataclass
class TransformerBlockConfig:
    embed_dim: int = 64
    attention_config: AttentionConfig = field(default_factory=lambda: AttentionConfig())

    diffmoe_num_experts: int = 8
    mlp_mode: Literal["vanilla", "diffmoe"] = "vanilla"
    norm_mode: Literal["layernorm", "dyntanh"] = "layernorm"


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config

        def get_norm():
            if config.norm_mode == "layernorm":
                return nn.LayerNorm(config.embed_dim)
            elif config.norm_mode == "dyntanh":
                return DynTanh(config.embed_dim)
            else:
                raise ValueError(config.norm_mode)

        self.norm1 = get_norm()
        self.attention = Attention(self.config.attention_config)

        if config.mlp_mode == "vanilla":
            self.norm2 = get_norm()
            self.mlp = MLP(config.embed_dim)
        elif config.mlp_mode == "diffmoe":
            norm2 = get_norm()
            self.mlp = DiffMoeMLP(
                embed_dim=config.embed_dim,
                norm_module=norm2,
                num_experts=config.diffmoe_num_experts,
            )
        else:
            raise ValueError(config.mlp_mode)

    def forward(
        self,
        x,
        key_pad_mask=None,
        attn_block_mask=None,
        attn_mask=None,
        rotary_embeds=None,
    ):
        norm_x = self.norm1(x)

        x = (
            x
            + self.attention(
                norm_x,
                block_mask=attn_block_mask,
                attn_mask=attn_mask,
                rotary_embeds=rotary_embeds,
            )[0]
        )

        if self.config.mlp_mode == "vanilla":
            x = x + self.mlp(self.norm2(x))
            diffmoe_outputs = tuple()
        elif self.config.mlp_mode == "diffmoe":
            # set the dynamic padding mult to a large number
            # so that torch.compile doesn't have to deal with
            # many dynamic shapes
            b, s, _ = x.shape
            bs = b * s
            k = bs // self.config.diffmoe_num_experts
            dynamic_padding_mult = 2 ** math.floor(math.log2(k))
            dynamic_padding_mult = max(8, dynamic_padding_mult)

            # integrated norm and residual
            x, *diffmoe_outputs = self.mlp(
                x, padding_mask=key_pad_mask, dynamic_padding_mult=dynamic_padding_mult
            )
        else:
            raise ValueError(self.config.norm_mode)

        return x, *diffmoe_outputs
