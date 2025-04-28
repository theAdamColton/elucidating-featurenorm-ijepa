import math
import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import einx
from dataclasses import dataclass, field

from src.transformer_blocks import TransformerBlock, TransformerBlockConfig


class WindowedPatchFlattener(nn.Module):
    def __init__(self, window_size: int = 4, patch_size: int = 4, dtype=torch.bfloat16):
        self.window_size = window_size
        self.patch_size = patch_size
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = einx.rearrange(
            "... (np ps)... c -> ... np... (ps... c)", x, ps=self.patch_size
        )
        x = einx.rearrange(
            "... (nw ws)... d -> ... (nw... ws...) d", x, ws=self.window_size
        )
        x = (x.to(self.dtype) / 255) * 2 - 1
        return x


class TimestepEmbedder(nn.Module):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    """

    def __init__(self, hidden_size=256, scale=1.0):
        super().__init__()
        downscale_freq_shift: float = 1
        max_period: int = 10000
        half_dim = hidden_size // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0,
            end=half_dim,
            dtype=torch.float32,
        )
        exponent = exponent / (half_dim - downscale_freq_shift)

        self.emb = nn.Parameter(torch.exp(exponent), requires_grad=False)
        self.scale = scale

    def forward(
        self,
        timesteps: torch.Tensor,
    ):
        emb = einx.multiply("..., d -> ... d", timesteps, self.emb)
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class AdaLayerNormShiftScale(nn.Module):
    def __init__(
        self,
        hidden_size,
        cond_size,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_size, hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * ( 1 + scale)
        x = x+ shift
        return x


@dataclass
class EncoderConfig:
    input_size: int = 768

    output_size: int = 256

    sliding_window_size: int = 5
    num_transformer_blocks: int = 2
    block_config: TransformerBlockConfig = field(
        default_factory=lambda: TransformerBlockConfig()
    )

    max_num_height_tokens: int = 16
    max_num_width_tokens: int = 16


class Encoder(nn.Module):
    def __init__(self, config=EncoderConfig()):
        super().__init__()
        self.config = config

        self.hidden_size = config.block_config.mlp_config.embed_dim

        self.proj_in = nn.Linear(config.input_size, self.hidden_size)
        self.temb = TimestepEmbedder(self.hidden_size)

        self.h_emb = nn.Parameter(
            torch.empty(config.max_num_height_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.h_emb, std=0.02)
        self.w_emb = nn.Parameter(
            torch.empty(config.max_num_width_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.w_emb, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(config.block_config)
            for _ in range(config.num_transformer_blocks)
        )

        self.norm_out = AdaLayerNormShiftScale(self.hidden_size, self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, config.output_size)

    def forward(self, x, t, token_ids, return_target_hidden_states=False):
        config = self.config

        b, s, d = x.shape

        assert einx.matches("b s d", x, d=config.input_size)
        assert einx.matches("b s", t, b=b, s=s)
        assert einx.matches("b s three", token_ids, b=b, s=s, three=3)

        device, dtype = x.device, x.dtype

        x = self.proj_in(x)

        sample_ids, position_ids = token_ids[:, 0], token_ids[:, 1:]

        hemb = self.h_emb[position_ids[..., 0]]
        wemb = self.w_emb[position_ids[..., 1]]
        pos_emb = hemb + wemb

        x = x + pos_emb

        sliding_window_reach = (config.sliding_window_size - 1) // 2

        def mask_mod(b, h, q_idx, kv_idx):
            is_same_sample = sample_ids[b, q_idx] == sample_ids[b, kv_idx]
            q_pos = position_ids[q_idx]
            kv_pos = position_ids[kv_idx]
            cheby_dist = (q_pos - kv_pos).abs().amax(-1)
            is_in_reach = cheby_dist <= sliding_window_reach
            return is_same_sample & is_in_reach

        block_mask = create_block_mask(
            mask_mod, B=b, H=None, Q_LEN=s, KV_LEN=s, device=device, BLOCK_SIZE=64
        )

        temb = self.temb(t)

        if return_target_hidden_states:
            target_hidden_states = torch.empty(
                b, s, config.output_size, device=device, dtype=dtype
            )

            mask = t == 0
            target_hidden_states[mask] = x

        for i, block in enumerate(self.blocks):
            x = block(x, temb, block_mask=block_mask)

            if return_target_hidden_states:
                mask = t == (i + 1)
                target_hidden_states[mask] = x

        x = self.norm_out(x, temb)
        x = self.proj_out(x)

        if return_target_hidden_states:
            target_hidden_states = self.norm_out(target_hidden_states, temb)

            return x, target_hidden_states

        return (x,)


@dataclass
class PredictorConfig:
    input_size: int = 256

    num_transformer_blocks: int = 2
    block_config: TransformerBlockConfig = field(
        default_factory=lambda: TransformerBlockConfig()
    )

    max_num_height_tokens: int = 16
    max_num_width_tokens: int = 16


class Predictor(nn.Module):
    def __init__(self, config=PredictorConfig()):
        super().__init__()
        self.config = config

        self.hidden_size = config.block_config.mlp_config.embed_dim

        self.proj_in = nn.Linear(config.input_size, self.hidden_size)
        self.temb = TimestepEmbedder(self.hidden_size)

        self.p_emb = nn.Parameter(torch.empty(self.hidden_size))
        init.trunc_normal_(self.p_emb, std=0.02)

        self.h_emb = nn.Parameter(
            torch.empty(config.max_num_height_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.h_emb, std=0.02)
        self.w_emb = nn.Parameter(
            torch.empty(config.max_num_width_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.w_emb, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(config.block_config)
            for _ in range(config.num_transformer_blocks)
        )

        self.norm_out = AdaLayerNormShiftScale(self.hidden_size, self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, config.input_size)

    def forward(self, x, t, token_ids, prediction_mask):
        config = self.config

        b, s, d = x.shape
        device, dtype = x.device, x.dtype

        assert einx.matches("b s d", x, d=config.input_size)
        assert einx.matches("b s", t, b=b, s=s)
        assert einx.matches("b s three", token_ids, b=b, s=s, three=3)
        assert einx.matches("b s", prediction_mask, b=b, s=s)

        x = self.proj_in(x)

        sample_ids, position_ids = token_ids[:, 0], token_ids[:, 1:]

        hemb = self.h_emb[position_ids[..., 0]]
        wemb = self.w_emb[position_ids[..., 1]]
        pos_emb = hemb + wemb

        p_emb = einx.multiply("d, b s -> b s d", self.p_emb, prediction_mask)

        x = x + pos_emb + p_emb

        temb = self.temb(t)

        attn_mask = einx.equal(
            "b s1, b s2 -> b h s1 s2",
            sample_ids,
            sample_ids,
            h=config.block_config.attention_config.num_attention_heads,
        )

        for block in self.blocks:
            x = block(x, temb, attn_mask=attn_mask)

        x = self.norm_out(x, temb)
        x = self.proj_out(x)

        return x


@dataclass
class IJEPADepthSmartConfig:
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig())
    predictor: PredictorConfig = field(default_factory=lambda: PredictorConfig())

    predictor_batch_repeat: int = 8
    predictor_context_capacity: float = 0.125
    predictor_target_capacity: float = 0.125

    should_tie_predictor_pos_emb: bool = True
    should_tie_predictor_temb: bool = True
    should_tie_predictor_norm_out: bool = True


class IJEPADepthSmart(nn.Module):
    def __init__(self, config=IJEPADepthSmartConfig()):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config.encoder)
        self.ema_encoder = Encoder(config.encoder)
        self.ema_encoder.load_state_dict(self.encoder.state_dict())

        self.predictor = Predictor(config.predictor)

        if config.should_tie_predictor_pos_emb:
            self.predictor.h_emb = self.encoder.h_emb
            self.predictor.w_emb = self.encoder.w_emb

        if config.should_tie_predictor_temb:
            self.predictor.temb = self.encoder.temb

        if config.should_tie_predictor_norm_out:
            self.predictor.norm_out = self.encoder.norm_out

    def forward(self, x, y, x_token_ids, y_token_ids, interp = 0):
        config = self.config
        device, dtype = x.device, x.dtype

        b, xs, d = x.shape
        b, ys, d = y.shape

        t = torch.randint(
            0, config.encoder.num_transformer_blocks + 1, (b,), device=device
        )

        x_t = einx.rearrange("b -> b xs", t, xs=xs)
        y_t = einx.rearrange("b -> b ys", t, ys=ys)

        with torch.no_grad():
            ema_encoder_outputs, target_hidden_states = self.ema_encoder(
                y, y_t, y_token_ids, return_target_hidden_states=True
            )

            target_hidden_states = ema_encoder_outputs * interp + target_hidden_states * (1-interp)

        target_hidden_states = F.layer_norm(target_hidden_states, (d,))

        x, *_ = self.encoder(x, x_t, x_token_ids)

        target_hidden_states = einx.rearrange("b ys d -> (r b) ys d", r = config.predictor_batch_repeat)
        x = einx.rearrange("b xs d -> (r b) xs d", x, r = config.predictor_batch_repeat)

        num_ctx_tokens = int(round(xs * config.predictor_context_capacity))
        ctx_ids = torch.rand(config.predictor_batch_repeat * b, xs, device=device, dtype=dtype).topk(
            num_ctx_tokens, dim=-1, sorted=False
        )

        num_target_tokens = int(round(ys * config.predictor_target_capacity))
        target_ids = torch.rand(config.predictor_batch_repeat * b, ys, device=device, dtype=dtype).topk(
            num_target_tokens, dim=-1, sorted=False
        )

        ctx = einx.get_at("rb [xs] d, rb k -> rb k d", x, ctx_ids)
        targets = einx.get_at(
            "rb [ys] d, rb m -> rb m d", target_hidden_states, target_ids
        )

        x = torch.cat((ctx,targets)), 1)
        prediction_mask = torch.zeros(b, x.shape[1], dtype=torch.bool, device=device)
        prediction_mask[:, num_ctx_tokens:] = 1

        x = self.predictor(x, t, x_token_ids, prediction_mask=prediction_mask)
        predictions = x[:, num_ctx_tokens:]

        loss = F.smooth_l1_loss(predictions, targets)

        return loss
