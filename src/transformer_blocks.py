from typing import Literal
import einx
import math
import torch
from torch.nn import init
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch import nn
from dataclasses import dataclass, field


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


@dataclass
class DiffMoeMLPConfig:
    embed_dim: int = 64
    mlp_ratio: int = 4
    use_bias: bool = True

    norm_mode: Literal["layernorm", "dyntanh"] = "layernorm"

    num_experts: int = 8

    capacity: float = 1.0
    should_clip_capacity_to_pow2: bool = False


class DiffMoeMLP(nn.Module):
    """
    DiffMOE as in:
    DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers
    https://arxiv.org/pdf/2503.14487
    """

    def __init__(self, config: DiffMoeMLPConfig):
        super().__init__()
        self.config = config

        if config.norm_mode == "layernorm":
            self.norm = nn.LayerNorm(config.embed_dim)
        elif config.norm_mode == "dyntanh":
            self.norm = DynTanh(config.embed_dim)
        else:
            raise ValueError(config.norm_mode)

        self.gate_proj = nn.Linear(config.embed_dim, config.num_experts, bias=False)

        self.fc1s = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.embed_dim * config.mlp_ratio,
                config.embed_dim,
            )
        )
        init.kaiming_uniform_(self.fc1s, a=math.sqrt(5))

        self.activation_fn = nn.GELU(approximate="tanh")

        self.fc2s = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.embed_dim,
                config.embed_dim * config.mlp_ratio,
            )
        )
        init.kaiming_uniform_(self.fc2s, a=math.sqrt(5))

        if config.use_bias:
            self.b1s = nn.Parameter(
                torch.empty(config.num_experts, config.embed_dim * config.mlp_ratio)
            )

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc1s)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.b1s, -bound, bound)

            self.b2s = nn.Parameter(torch.empty(config.num_experts, config.embed_dim))

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc2s)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.b2s, -bound, bound)

    def forward(
        self,
        x,
        mask=None,
    ):
        """
        x: Shape: (... d)
        mask: Optional, shape: (...)
        Notation:
        k: the total number of selected tokens
        m: the total number of dropped tokens
        d: input hidden channel size
        dd: mlp hidden channel size
        """
        config = self.config

        og_shape = x.shape
        x = einx.rearrange("... d -> (...) d", x)
        bs = x.shape[0]

        # TODO
        # this differs from the official diff-moe implementation
        # I derive gate scores from unnormalized x, instead of
        # normalized x
        # And I use tanh scaled to [0,1], instead of softmax
        scores = self.gate_proj(x)
        # scores = scores.softmax(-1)
        scores = (F.tanh(scores) + 1) / 2

        if mask is not None:
            mask = einx.rearrange("... -> (...) one", mask, one=1)
            assert bs == mask.shape[0]
            mask_value = -200
            scores = scores.masked_fill(mask, mask_value)

        # k is the total number of MLP forward passes over all experts
        k = int(bs * self.config.capacity) // self.config.num_experts
        if config.should_clip_capacity_to_pow2:
            k = 2 ** int(math.log2(k))
            k = max(k, 1)

        sorted_expert_weights, sorted_expert_idx = scores.sort(0, descending=True)
        kept_expert_weights, dropped_expert_weights = (
            sorted_expert_weights[:k],
            sorted_expert_weights[k:],
        )
        kept_expert_idx, dropped_expert_idx = (
            sorted_expert_idx[:k],
            sorted_expert_idx[k:],
        )

        kept_expert_idx = einx.rearrange("k n -> (k n)", kept_expert_idx)

        # [b] d, (k n) -> (k n) d
        y = torch.index_select(x, 0, kept_expert_idx)

        y = self.norm(y)

        y = einx.dot("(k n) d, n dd d -> (k n) dd", y, self.fc1s)
        if self.config.use_bias:
            y = einx.add("(k n) dd, n dd -> (k n) dd", y, self.b1s)

        y = F.gelu(y, approximate="tanh")

        y = einx.dot("(k n) dd, n d dd -> (k n) d", y, self.fc2s)
        if self.config.use_bias:
            y = einx.add("(k n) d, n d -> (k n) d", y, self.b2s)

        y = einx.multiply("(k n) d, k n -> (k n) d", y, kept_expert_weights)
        y = y.to(x.dtype)

        x = torch.index_add(x, 0, kept_expert_idx, y)

        x = x.reshape(og_shape)

        return x


@dataclass
class TransformerBlockConfig:
    mlp_config: DiffMoeMLPConfig = field(default_factory=lambda: DiffMoeMLPConfig())
    attention_config: AttentionConfig = field(default_factory=lambda: AttentionConfig())

    norm_mode: Literal["layernorm", "dyntanh"] = "layernorm"


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config

        self.embed_dim = config.mlp_config.embed_dim

        if config.norm_mode == "layernorm":
            self.norm1 = nn.LayerNorm(self.embed_dim)
        elif config.norm_mode == "dyntanh":
            self.norm1 = DynTanh(self.embed_dim)
        else:
            raise ValueError(config.norm_mode)

        self.attention = Attention(self.config.attention_config)
        self.mlp = DiffMoeMLP(self.config.mlp_config)

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
        x = self.mlp(x, key_pad_mask)
        return x
