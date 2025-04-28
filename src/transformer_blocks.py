import einx
import math
import torch
from torch.nn import init
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch import nn
from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    embed_dim: int = 64
    head_dim: int = 64
    num_attention_heads: int = 1
    should_use_qk_norm: bool = True


class Attention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.proj_qkv = nn.Linear(
            config.embed_dim,
            config.head_dim * config.num_attention_heads * 3,
            bias=False,
        )
        if config.should_use_qk_norm:
            self.q_norm = nn.LayerNorm(config.head_dim)
            self.k_norm = nn.LayerNorm(config.head_dim)
        self.proj_out = nn.Linear(
            config.num_attention_heads * config.head_dim, config.embed_dim, bias=False
        )

    def forward(self, x, block_mask=None, attn_mask=None):
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

        # Hack for pytorch 2.6
        # Because flex attention doesn't currently work with autocast
        smallest_dtype = torch.float32
        for tensor in q, k, v:
            if tensor.dtype.itemsize < smallest_dtype.itemsize:
                smallest_dtype = tensor.dtype
        q = q.to(smallest_dtype)
        k = k.to(smallest_dtype)
        v = v.to(smallest_dtype)

        if block_mask is not None:
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

        self.norm = nn.LayerNorm(config.embed_dim)

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
    ):
        """
        x: Shape: (... d)
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
        # And I use tanh instead of softmax
        scores = self.gate_proj(x)
        scores = scores.softmax(-1)
        # scores = (F.tanh(scores) + 1) / 2

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

        x = torch.index_add(x, 0, kept_expert_idx, y)

        x = x.reshape(og_shape)

        return x


class AdaLayerNormShift(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, emb):
        shift = self.linear(self.silu(emb))
        x = self.norm(x) + shift
        return x


@dataclass
class TransformerBlockConfig:
    mlp_config: DiffMoeMLPConfig = field(default_factory=lambda: DiffMoeMLPConfig())
    attention_config: AttentionConfig = field(default_factory=lambda: AttentionConfig())


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config

        self.embed_dim = config.mlp_config.embed_dim

        self.norm1 = AdaLayerNormShift(self.embed_dim)
        self.attention = Attention(self.config.attention_config)
        self.mlp = DiffMoeMLP(self.config.mlp_config)

    def forward(self, x, t, block_mask=None, attn_mask=None):
        x = (
            x
            + self.attention(
                self.norm1(x, t), block_mask=block_mask, attn_mask=attn_mask
            )[0]
        )
        x = self.mlp(x)
        return x
