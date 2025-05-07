import math
from typing import Literal
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import einx
from dataclasses import dataclass, field

from src.transformer_blocks import TransformerBlock, TransformerBlockConfig
from src.dataset import MASK_SEQUENCE_ID


def compute_smooth_rank(x, eps=1e-7):
    """
    x: Batch of representations, shape: (B Z)

    This is a metric studied in the 2023 paper:
    RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank

    Higher smooth rank
    """
    x = x.float()

    s = torch.linalg.svdvals(x)
    s_norm = s.norm(1)
    p = s / s_norm
    log_p = torch.log(p + eps)
    entropy = torch.exp(-(p * log_p).sum())
    return entropy


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


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta=10000,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device)[
                : (dim // 2)
            ]
            / dim
        )
    )  # [D/2]
    freqs = einx.dot("... s, d2 -> ... s d2", pos, freqs)  # [..., s, D/2]
    freqs_cos = freqs.cos().repeat_interleave(2, dim=-1).float()  # [..., s, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=-1).float()  # [..., s, D]
    return freqs_cos, freqs_sin


class RopePosEmbedND(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta=10000, axes_dim=(32, 32)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor):
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[..., i], theta=self.theta
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class AdaLayerNormShiftScale(nn.Module):
    def __init__(
        self,
        hidden_size,
        cond_size,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_size, hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale)
        x = x + shift
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


class RunningBatchNorm(nn.Module):
    def __init__(self, hidden_size, beta=0.99, eps=1e-5):
        super().__init__()
        self.is_initted = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.running_mean = nn.Parameter(torch.empty(hidden_size), requires_grad=False)
        self.running_std = nn.Parameter(torch.empty(hidden_size), requires_grad=False)
        self.beta = beta
        self.eps = eps

    def forward(self, x, mask=None):
        need_init = self.is_initted == 0 and self.training
        need_update = self.is_initted > 0 and self.training

        if need_init or need_update:
            if mask is not None:
                x_tokens = x[mask]
            else:
                x_tokens = x

            with torch.no_grad():
                mean = einx.mean("[...] d", x_tokens)
                std = einx.std("[...] d", x_tokens)

            if need_init:
                self.running_mean.copy_(mean)
                self.running_std.copy_(std)

                self.is_initted.fill_(1)

            elif need_update:
                self.running_mean.lerp_(mean, 1 - self.beta)
                self.running_std.lerp_(std, 1 - self.beta)

        x = einx.subtract("... d, d", x, self.running_mean)
        x = einx.divide("... d, d", x, self.running_std.clamp(self.eps))

        return x


@dataclass
class EncoderConfig:
    input_size: int = 768

    sliding_window_size: int = 5
    num_transformer_blocks: int = 10
    block_config: TransformerBlockConfig = field(
        default_factory=lambda: TransformerBlockConfig()
    )

    use_rope2d: bool = True

    max_num_height_tokens: int = 64
    max_num_width_tokens: int = 64
    max_num_register_tokens: int = 8
    norm_out_mode: Literal[
        "disabled", "adanorm", "layernorm", "batchnorm", "dyntanh"
    ] = "layernorm"


class Encoder(nn.Module):
    def __init__(self, config=EncoderConfig()):
        super().__init__()
        self.config = config

        self.hidden_size = config.block_config.mlp_config.embed_dim
        self.head_dim = config.block_config.attention_config.head_dim

        self.proj_in = nn.Linear(config.input_size, self.hidden_size)
        self.temb = TimestepEmbedder(self.hidden_size)

        self.reg_emb = nn.Parameter(
            torch.empty(config.max_num_register_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.reg_emb, std=0.02)

        self.h_emb = nn.Parameter(
            torch.empty(config.max_num_height_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.h_emb, std=0.02)
        self.w_emb = nn.Parameter(
            torch.empty(config.max_num_width_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.w_emb, std=0.02)

        if config.use_rope2d:
            self.rope_pos_emb = RopePosEmbedND(
                axes_dim=(self.head_dim // 2, self.head_dim // 2)
            )

        self.blocks = nn.ModuleList(
            TransformerBlock(config.block_config)
            for _ in range(config.num_transformer_blocks)
        )

        if config.norm_out_mode == "layernorm":
            self.norm_out = nn.LayerNorm(self.hidden_size)
        elif config.norm_out_mode == "disabled":
            self.norm_out = nn.Identity()
        elif config.norm_out_mode == "batchnorm":
            self.norm_out = RunningBatchNorm(self.hidden_size)
        elif config.norm_out_mode == "dyntanh":
            self.norm_out = DynTanh(self.hidden_size)
        elif config.norm_out_mode == "adanorm":
            self.norm_out = AdaLayerNormShiftScale(self.hidden_size, self.hidden_size)
        else:
            raise ValueError(config.norm_out_mode)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        token_ids: torch.Tensor,
        return_target_hidden_states=False,
        return_all_layer_features=False,
    ):
        config = self.config

        b, s, d = x.shape

        assert einx.matches("b s d", x, d=config.input_size)
        assert einx.matches("b s", t, b=b, s=s)
        assert einx.matches("b s four", token_ids, b=b, s=s, four=4)

        assert not (return_target_hidden_states and return_all_layer_features)

        device, dtype = x.device, x.dtype

        x = self.proj_in(x)

        sample_ids, register_ids, position_ids = (
            token_ids[..., 0],
            token_ids[..., 1],
            token_ids[..., 2:],
        )

        is_register = register_ids != MASK_SEQUENCE_ID
        register_ids = register_ids.masked_fill(~is_register, 0)
        reg_emb = self.reg_emb[register_ids]
        reg_emb = einx.multiply("b s d, b s", reg_emb, is_register)
        x = x + reg_emb

        if config.use_rope2d:
            rotary_embeds = self.rope_pos_emb(position_ids)
        else:
            rotary_embeds = None

        pos_emb = self.h_emb[position_ids[..., 0]] + self.w_emb[position_ids[..., 1]]
        x = x + pos_emb

        attn_mask = einx.equal(
            "b s1, b s2 -> b h s1 s2",
            sample_ids,
            sample_ids,
            h=config.block_config.attention_config.num_attention_heads,
        )

        temb = self.temb(t)

        if return_target_hidden_states:
            target_hidden_states = torch.zeros(
                b, s, self.hidden_size, device=device, dtype=dtype
            )

            mask = t == 0
            # basically masked fill
            target_hidden_states = target_hidden_states + x * mask.unsqueeze(-1)

        elif return_all_layer_features:
            all_layer_features = torch.empty(
                config.num_transformer_blocks + 1,
                b,
                s,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )

            all_layer_features[0] = x

        for i, block in enumerate(self.blocks):
            x = block(x, temb, attn_mask=attn_mask, rotary_embeds=rotary_embeds)

            if return_target_hidden_states:
                mask = t == (i + 1)
                target_hidden_states = target_hidden_states + x * mask.unsqueeze(-1)

            elif return_all_layer_features:
                all_layer_features[i + 1] = x

        if config.norm_out_mode == "batchnorm":
            x = self.norm_out(x, sample_ids != MASK_SEQUENCE_ID)
        elif config.norm_out_mode == "adanorm":
            x = self.norm_out(x, temb)
        else:
            x = self.norm_out(x)

        if return_target_hidden_states:
            return x, target_hidden_states

        elif return_all_layer_features:
            if config.norm_out_mode == "adanorm":
                all_layer_features = self.norm_out(all_layer_features, temb)
            else:
                # TODO
                # Don't want this to contribute to RunningBatchNorm estimations
                all_layer_features = self.norm_out(all_layer_features)

            return x, all_layer_features

        return (x,)


@dataclass
class PredictorConfig:
    input_size: int = 64

    num_transformer_blocks: int = 2
    block_config: TransformerBlockConfig = field(
        default_factory=lambda: TransformerBlockConfig()
    )

    use_rope2d: bool = True
    max_num_height_tokens: int = 64
    max_num_width_tokens: int = 64
    max_num_register_tokens: int = 8

    should_zero_ctx_register_tokens: bool = True


class Predictor(nn.Module):
    def __init__(self, config=PredictorConfig()):
        super().__init__()
        self.config = config

        self.hidden_size = config.block_config.mlp_config.embed_dim
        self.head_dim = config.block_config.attention_config.head_dim

        if config.input_size == self.hidden_size:
            self.proj_in = nn.Identity()
        else:
            self.proj_in = nn.Linear(config.input_size, self.hidden_size)

        self.temb = TimestepEmbedder(self.hidden_size)

        self.reg_emb = nn.Parameter(
            torch.empty(config.max_num_register_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.reg_emb, std=0.02)

        self.h_emb = nn.Parameter(
            torch.empty(config.max_num_height_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.h_emb, std=0.02)
        self.w_emb = nn.Parameter(
            torch.empty(config.max_num_width_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.w_emb, std=0.02)

        self.pred_emb = nn.Parameter(torch.empty(self.hidden_size))
        init.trunc_normal_(self.pred_emb, std=0.02)

        if config.use_rope2d:
            self.pos_emb = RopePosEmbedND(
                axes_dim=(self.head_dim // 2, self.head_dim // 2)
            )

        self.blocks = nn.ModuleList(
            TransformerBlock(config.block_config)
            for _ in range(config.num_transformer_blocks)
        )

        self.norm_out = AdaLayerNormShiftScale(self.hidden_size, self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, config.input_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        token_ids: torch.Tensor,
        prediction_mask: torch.Tensor,
    ):
        config = self.config

        b, s, _ = x.shape

        assert einx.matches("b s d", x, d=config.input_size)
        assert einx.matches("b s", t, b=b, s=s)
        assert einx.matches("b s four", token_ids, b=b, s=s, four=4)

        x = self.proj_in(x)

        sample_ids, register_ids, position_ids = (
            token_ids[..., 0],
            token_ids[..., 1],
            token_ids[..., 2:],
        )

        # zero out tokens to predict
        x = einx.multiply("b s d, b s", x, ~prediction_mask)

        is_register = register_ids != MASK_SEQUENCE_ID
        register_ids = register_ids.masked_fill(~is_register, 0)
        reg_emb = self.reg_emb[register_ids]
        reg_emb = reg_emb * is_register.unsqueeze(-1)

        if config.should_zero_ctx_register_tokens:
            # zero out context register tokens
            x = x * ~is_register.unsqueeze(-1)

        x = x + reg_emb

        pos_emb = self.h_emb[position_ids[..., 0]] + self.w_emb[position_ids[..., 1]]
        x = x + pos_emb

        p_emb = einx.multiply("d, b s -> b s d", self.pred_emb, prediction_mask)

        x = x + p_emb

        temb = self.temb(t)

        attn_mask = einx.equal(
            "b s1, b s2 -> b h s1 s2",
            sample_ids,
            sample_ids,
            h=config.block_config.attention_config.num_attention_heads,
        )

        if config.use_rope2d:
            rotary_embeds = self.pos_emb(position_ids)
        else:
            rotary_embeds = None

        for block in self.blocks:
            x = block(x, temb, attn_mask=attn_mask, rotary_embeds=rotary_embeds)

        x = self.norm_out(x, temb)
        x = self.proj_out(x)

        return x


@dataclass
class IJEPADepthSmartConfig:
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig())
    predictor: PredictorConfig = field(default_factory=lambda: PredictorConfig())

    depthsmart_mode: Literal["random-layers", "disabled", "noise"] = "random-layers"
    num_denoiser_timesteps: int = 1000

    should_predict_register_tokens: bool = False

    target_norm_mode: Literal[
        "layernorm", "disabled", "batchnorm", "running-batchnorm"
    ] = "layernorm"

    predictor_batch_repeat: int = 8
    predictor_context_capacity: float = 0.125
    predictor_target_capacity: float = 0.125


class IJEPADepthSmart(nn.Module):
    def __init__(self, config=IJEPADepthSmartConfig()):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config.encoder)
        self.ema_encoder = Encoder(config.encoder)
        self.ema_encoder.load_state_dict(self.encoder.state_dict())
        self.ema_encoder.eval()
        self.ema_encoder.requires_grad_(False)

        if config.target_norm_mode == "running-batchnorm":
            self.running_batchnorm = RunningBatchNorm(self.encoder.hidden_size)

        self.predictor = Predictor(config.predictor)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_token_ids: torch.Tensor,
        y_token_ids: torch.Tensor,
        interp=0,
        return_smooth_rank=False,
    ):
        config = self.config
        device, dtype = x.device, x.dtype

        b, xs, d = x.shape
        b, ts, d = y.shape

        y = torch.cat((x, y), 1)
        y_token_ids = torch.cat((x_token_ids, y_token_ids), 1)

        b, ys, d = y.shape

        num_feature_depth = config.encoder.num_transformer_blocks + 1

        if config.depthsmart_mode == "random-layers":
            t = torch.randint(0, num_feature_depth, (b,), device=device)
        elif config.depthsmart_mode == "disabled":
            t = torch.full((b,), num_feature_depth - 1, device=device)
        elif config.depthsmart_mode == "noise":
            t = torch.randint(0, config.num_denoiser_timesteps, (b,), device=device)
        else:
            raise ValueError(config.depthsmart_mode)

        x_t = einx.rearrange("b -> b xs", t, xs=xs)
        y_t = einx.rearrange("b -> b ys", t, ys=ys)

        with torch.no_grad():
            ema_encoder_outputs = self.ema_encoder(
                y,
                y_t,
                y_token_ids,
                return_target_hidden_states=config.depthsmart_mode == "random-layers",
            )

            if config.depthsmart_mode == "random-layers":
                ema_encoder_outputs, target_hidden_states, *_ = ema_encoder_outputs
                target_hidden_states = (
                    ema_encoder_outputs * interp + target_hidden_states * (1 - interp)
                )
            else:
                target_hidden_states, *_ = ema_encoder_outputs

        # Keep only the target hidden states from
        # tokens absent from context
        target_hidden_states = target_hidden_states[:, xs:, :]
        y_token_ids = y_token_ids[:, xs:, :]

        if config.target_norm_mode == "layernorm":
            target_hidden_states = F.layer_norm(
                target_hidden_states, (target_hidden_states.shape[-1],)
            )
        elif config.target_norm_mode == "batchnorm":
            mask = y_token_ids[..., 0] != MASK_SEQUENCE_ID
            mean = einx.mean("[n] d", target_hidden_states[mask])
            std = einx.std("[n] d", target_hidden_states[mask])
            target_hidden_states = einx.subtract(
                "b ts d, d", target_hidden_states, mean
            )
            eps = 1e-7
            target_hidden_states = einx.divide(
                "b ts d, d", target_hidden_states, std.clamp(eps)
            )
        elif config.target_norm_mode == "running-batchnorm":
            mask = y_token_ids[..., 0] != MASK_SEQUENCE_ID
            target_hidden_states = self.running_batchnorm(target_hidden_states, mask)

        elif config.target_norm_mode == "disabled":
            pass
        else:
            raise ValueError(config.target_norm_mode)

        smooth_rank = None
        if return_smooth_rank:
            smooth_rank = compute_smooth_rank(
                target_hidden_states.view(-1, target_hidden_states.shape[-1])
            )

        x, *_ = self.encoder(x, x_t, x_token_ids)

        # Repeat tensors for prediction
        target_hidden_states = einx.rearrange(
            "b ts d -> (r b) ts d",
            target_hidden_states,
            r=config.predictor_batch_repeat,
        )
        x = einx.rearrange("b xs d -> (r b) xs d", x, r=config.predictor_batch_repeat)
        y_token_ids = einx.rearrange(
            "b ts nd -> (r b) ts nd", y_token_ids, r=config.predictor_batch_repeat
        )
        x_token_ids = einx.rearrange(
            "b xs nd -> (r b) xs nd", x_token_ids, r=config.predictor_batch_repeat
        )

        # The predictor uses a random subset of the context
        # to predict a random subset of the target
        num_ctx_tokens = int(round(xs * config.predictor_context_capacity))
        num_target_tokens = int(round(ts * config.predictor_target_capacity))

        # Random scores for picking context tokens to be used by the predictor
        ctx_scores = torch.rand(
            config.predictor_batch_repeat * b, xs, device=device, dtype=dtype
        )
        # Try to prevent the predictor from being given padding tokens
        # as part of context
        is_ctx_padding = x_token_ids[..., 0] == MASK_SEQUENCE_ID
        ctx_scores.masked_fill_(is_ctx_padding, -1)
        ctx_ids = ctx_scores.topk(num_ctx_tokens, dim=-1, sorted=False).indices

        target_scores = torch.rand(
            config.predictor_batch_repeat * b, ts, device=device, dtype=dtype
        )
        # Try to prevent the predictor from being given padding tokens
        # as prediction targets
        is_target_padding = y_token_ids[..., 0] == MASK_SEQUENCE_ID
        target_scores.masked_fill_(is_target_padding, -1)
        target_ids = target_scores.topk(num_target_tokens, dim=-1, sorted=False).indices

        ctx = einx.get_at("rb [xs] d, rb k -> rb k d", x, ctx_ids)
        targets = einx.get_at(
            "rb [ts] d, rb m -> rb m d", target_hidden_states, target_ids
        )

        ctx_token_ids = einx.get_at("rb [xs] nd, rb k -> rb k nd", x_token_ids, ctx_ids)
        target_token_ids = einx.get_at(
            "rb [ts] nd, rb m -> rb m nd", y_token_ids, target_ids
        )

        if config.depthsmart_mode == "noise":
            noise_timesteps = einx.rearrange(
                "b -> (r b)", t, r=config.predictor_batch_repeat
            )
            noise = torch.randn_like(target_hidden_states)
            raise NotImplementedError()

        x = torch.cat((ctx, targets), 1)
        token_ids = torch.cat((ctx_token_ids, target_token_ids), 1)

        rb, ps, d = x.shape

        prediction_mask = torch.zeros(
            config.predictor_batch_repeat * b,
            x.shape[1],
            dtype=torch.bool,
            device=device,
        )
        prediction_mask[:, num_ctx_tokens:] = 1

        t = einx.rearrange("b -> (r b) ps", t, r=config.predictor_batch_repeat, ps=ps)

        x = self.predictor(x, t, token_ids, prediction_mask=prediction_mask)
        predictions = x[:, num_ctx_tokens:]

        loss = F.smooth_l1_loss(predictions, targets, reduction="none")

        # TODO
        # The predictor here is predicting the teacher's register tokens
        # Should these registers be upweighted? Or excluded from loss

        target_sequence_ids = token_ids[:, num_ctx_tokens:, 0]
        target_register_ids = token_ids[:, num_ctx_tokens:, 1]
        is_target_mask = target_sequence_ids != MASK_SEQUENCE_ID

        if not config.should_predict_register_tokens:
            is_target_mask = is_target_mask & (target_register_ids == MASK_SEQUENCE_ID)

        loss = loss[is_target_mask].mean()

        return dict(loss=loss, smooth_rank=smooth_rank)
