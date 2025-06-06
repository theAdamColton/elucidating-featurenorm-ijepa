from typing import Literal
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import einx
from dataclasses import dataclass, field

from src.transformer_blocks import TransformerBlock, TransformerBlockConfig, DynTanh
from src.dataset import MASK_SAMPLE_ID


def get_attn_mask(sample_ids, num_heads):
    attn_mask = einx.equal(
        "b s1, b s2 -> b s1 s2",
        sample_ids,
        sample_ids,
    )
    is_not_padding = sample_ids != MASK_SAMPLE_ID
    attn_mask = attn_mask & is_not_padding.unsqueeze(-1) & is_not_padding.unsqueeze(-2)
    attn_mask = einx.rearrange("b s1 s2 -> b h s1 s2", attn_mask, h=num_heads)
    return attn_mask


def expand_trailing(y, x):
    """
    x has more dimensions than y

    unsqueezes y to match the number of dimensions of x
    and then expands (repeats) y along these trailing dimensions
    so that the shapes of x and y match
    """
    y_shape = y.shape
    x_shape = x.shape
    new_y_shape = list(y_shape) + list(x_shape[len(y_shape) :])
    for _ in range(len(new_y_shape) - len(y_shape)):
        y = y.unsqueeze(-1)
    y = y.expand(new_y_shape)
    return y


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
        pos (`torch.Tensor`): Floating point position ids for the frequency tensor. [...]
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [... D]
            Where D==dim
    """
    device = pos.device
    assert dim % 2 == 0

    with torch.autocast(device.type, enabled=False):
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
    """
    Adapted from:
    https://github.com/huggingface/diffusers/blob/54cddc1e127a481ecb20179acf4a57f5421f4626/src/diffusers/models/embeddings.py#L962
    """

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
        freqs_cos = torch.cat(cos_out, dim=-1)
        freqs_sin = torch.cat(sin_out, dim=-1)
        return freqs_cos, freqs_sin


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

                def update(p, w):
                    return p.float().lerp(w.float(), 1 - self.beta).to(p.dtype)

                self.running_mean.copy_(update(self.running_mean, mean))
                self.running_std.copy_(update(self.running_std, std))

        x = einx.subtract("... d, d", x, self.running_mean)
        x = einx.divide("... d, d", x, self.running_std.clamp(self.eps))

        return x


@dataclass
class EncoderConfig:
    input_size: int = 768

    num_transformer_blocks: int = 10
    block_config: TransformerBlockConfig = field(
        default_factory=lambda: TransformerBlockConfig()
    )

    use_rope2d: bool = True

    use_abs_pos_emb: bool = False

    max_num_height_tokens: int = 64
    max_num_width_tokens: int = 64
    max_num_register_tokens: int = 8
    norm_out_mode: Literal["disabled", "layernorm", "batchnorm", "dyntanh"] = (
        "layernorm"
    )
    norm_elementwise_affine: bool = True


class Encoder(nn.Module):
    def __init__(self, config=EncoderConfig()):
        super().__init__()
        self.config = config

        self.hidden_size = config.block_config.embed_dim
        self.head_dim = config.block_config.attention_config.head_dim

        self.proj_in = nn.Linear(config.input_size, self.hidden_size)

        self.reg_emb = nn.Parameter(
            torch.empty(config.max_num_register_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.reg_emb, std=0.02)

        if config.use_abs_pos_emb:
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
            self.norm_out = nn.LayerNorm(
                self.hidden_size, elementwise_affine=config.norm_elementwise_affine
            )
        elif config.norm_out_mode == "disabled":
            self.norm_out = nn.Identity()
        elif config.norm_out_mode == "batchnorm":
            self.norm_out = RunningBatchNorm(self.hidden_size)
        elif config.norm_out_mode == "dyntanh":
            self.norm_out = DynTanh(
                self.hidden_size, self.config.norm_elementwise_affine
            )
        else:
            raise ValueError(config.norm_out_mode)

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        return_all_layer_features: bool = False,
    ):
        config = self.config

        b, s, d = x.shape
        device, dtype = x.device, x.dtype

        if not torch.compiler.is_compiling():
            assert einx.matches("b s d", x, d=config.input_size)
            assert einx.matches("b s four", token_ids, b=b, s=s, four=4)

        x = self.proj_in(x)

        sample_ids, register_ids, position_ids = (
            token_ids[..., 0],
            token_ids[..., 1],
            token_ids[..., 2:],
        )

        is_register = register_ids != MASK_SAMPLE_ID
        register_ids = register_ids.masked_fill(~is_register, 0)
        reg_emb = self.reg_emb[register_ids]
        reg_emb = einx.multiply("b s d, b s", reg_emb, is_register)
        x = x + reg_emb

        if config.use_rope2d:
            rotary_embeds = self.rope_pos_emb(position_ids)
        else:
            rotary_embeds = None

        if config.use_abs_pos_emb:
            pos_emb = (
                self.h_emb[position_ids[..., 0]] + self.w_emb[position_ids[..., 1]]
            )
            x = x + pos_emb

        attn_mask = get_attn_mask(
            sample_ids, config.block_config.attention_config.num_attention_heads
        )

        key_pad_mask = sample_ids == MASK_SAMPLE_ID

        if return_all_layer_features:
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
            x = block(
                x,
                key_pad_mask=key_pad_mask,
                attn_mask=attn_mask,
                rotary_embeds=rotary_embeds,
            )

            if return_all_layer_features:
                all_layer_features[i + 1] = x

        if config.norm_out_mode == "batchnorm":
            x = self.norm_out(x, sample_ids != MASK_SAMPLE_ID)
        else:
            x = self.norm_out(x)

        if return_all_layer_features:
            # Do not normalize layer features
            return x, all_layer_features

        return (x,)


@dataclass
class PredictorConfig:
    input_size: int = 64
    should_mlp_in: bool = False

    num_transformer_blocks: int = 2
    block_config: TransformerBlockConfig = field(
        default_factory=lambda: TransformerBlockConfig()
    )

    use_rope2d: bool = True
    use_abs_pos_emb: bool = False

    max_num_height_tokens: int = 64
    max_num_width_tokens: int = 64
    max_num_register_tokens: int = 8

    should_zero_ctx_register_tokens: bool = True

    norm_out_mode: Literal["layernorm", "dyntanh"] = "layernorm"


class Predictor(nn.Module):
    def __init__(self, config=PredictorConfig()):
        super().__init__()
        self.config = config

        self.hidden_size = config.block_config.embed_dim
        self.head_dim = config.block_config.attention_config.head_dim

        if config.should_mlp_in:
            self.proj_in = nn.Sequential(
                nn.Linear(config.input_size, config.input_size * 4),
                nn.GELU(approximate="tanh"),
                nn.Linear(config.input_size * 4, self.hidden_size),
            )
        else:
            self.proj_in = nn.Linear(config.input_size, self.hidden_size)

        self.reg_emb = nn.Parameter(
            torch.empty(config.max_num_register_tokens, self.hidden_size)
        )
        init.trunc_normal_(self.reg_emb, std=0.02)

        if config.use_abs_pos_emb:
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

        if config.norm_out_mode == "layernorm":
            self.norm_out = nn.LayerNorm(self.hidden_size)
        elif config.norm_out_mode == "dyntanh":
            self.norm_out = DynTanh(self.hidden_size)
        else:
            raise ValueError(config.norm_out_mode)

        self.proj_out = nn.Linear(self.hidden_size, config.input_size)

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        prediction_mask: torch.Tensor,
    ):
        config = self.config

        b, s, _ = x.shape

        if not torch.compiler.is_compiling():
            assert einx.matches("b s d", x, d=config.input_size)
            assert einx.matches("b s four", token_ids, b=b, s=s, four=4)

        x = self.proj_in(x)

        sample_ids, register_ids, position_ids = (
            token_ids[..., 0],
            token_ids[..., 1],
            token_ids[..., 2:],
        )

        # zero out tokens to predict
        x = einx.multiply("b s d, b s", x, ~prediction_mask)

        is_register = register_ids != MASK_SAMPLE_ID
        register_ids = register_ids.masked_fill(~is_register, 0)
        reg_emb = self.reg_emb[register_ids]
        reg_emb = reg_emb * is_register.unsqueeze(-1)

        if config.should_zero_ctx_register_tokens:
            # zero out context register tokens
            x = x * ~is_register.unsqueeze(-1)

        x = x + reg_emb

        if config.use_abs_pos_emb:
            pos_emb = (
                self.h_emb[position_ids[..., 0]] + self.w_emb[position_ids[..., 1]]
            )
            x = x + pos_emb

        p_emb = einx.multiply("d, b s -> b s d", self.pred_emb, prediction_mask)

        x = x + p_emb

        attn_mask = get_attn_mask(
            sample_ids, config.block_config.attention_config.num_attention_heads
        )

        key_pad_mask = sample_ids == MASK_SAMPLE_ID

        if config.use_rope2d:
            rotary_embeds = self.pos_emb(position_ids)
        else:
            rotary_embeds = None

        for block in self.blocks:
            x = block(
                x,
                key_pad_mask=key_pad_mask,
                attn_mask=attn_mask,
                rotary_embeds=rotary_embeds,
            )

        x = self.norm_out(x)
        x = self.proj_out(x)

        return x


def get_random_idx_with_replacement(
    r,
    b,
    s,
    capacity=0.5,
    mask=None,
    device=torch.device("cuda"),
    dtype=torch.float32,
):
    scores = torch.rand(
        r,
        b,
        s,
        device=device,
        dtype=dtype,
    )

    if mask is not None:
        assert b, s == mask.shape
        mask = mask.unsqueeze(0)
        scores.masked_fill_(mask, -1)

    k = int(round(s * capacity))
    idx = torch.topk(scores, k, sorted=False).indices
    idx = einx.rearrange("r b k -> (r b) k", idx)

    return idx


def get_random_idx_without_replacement(
    r,
    b,
    s,
    capacity=0.5,
    mask=None,
    device=torch.device("cpu"),
    dtype=torch.float32,
):
    """

    Sample r sets of independent random indices within the range of s.

    This function generates random indices that can be used to index into a sequence of length s.
    It optionally excludes positions that are marked as True in the provided mask.
    """

    if s * capacity * r > s:
        raise ValueError(
            f"ValueError: The product of s ({s}), capacity ({capacity}), and r ({r}) "
            f"must be less than or equal to s ({s}). However, the calculated product is {s * capacity * r}, "
            f"which is greater than s ({s})"
        )

    ss = int(round(s * capacity))
    if s % ss != 0:
        raise ValueError(
            f"The sequence length {s} with capacity {capacity} "
            f"results in a subsampled sequence length {ss} that is not divisible by {r}. "
            f"Please ensure that the sequence length is divisible by the subsampled capacity."
        )

    scores = torch.rand(
        b,
        s,
        device=device,
        dtype=dtype,
    )

    if mask is not None:
        scores.masked_fill_(mask, -1)

    idx = scores.argsort(descending=True)
    idx = einx.rearrange("b (rr ss) -> rr b ss", idx, ss=ss)
    # rr b xs -> r b ss
    idx = idx[:r]

    idx = einx.rearrange("r b ss -> (r b) ss", idx)

    return idx


@dataclass
class IJEPAConfig:
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig())
    predictor: PredictorConfig = field(default_factory=lambda: PredictorConfig())

    should_predict_register_tokens: bool = False
    should_attempt_mask_dropping: bool = True
    should_predict_from_all_target: bool = False

    target_norm_mode: Literal[
        "layernorm", "disabled", "batchnorm", "running-batchnorm"
    ] = "layernorm"

    predictor_batch_repeat: int = 8
    predictor_context_capacity: float = 0.125
    predictor_target_capacity: float = 0.125

    sample_predictor_context_with_replacement: bool = True
    sample_predictor_targets_with_replacement: bool = True


class IJEPAModel(nn.Module):
    def __init__(self, config=IJEPAConfig()):
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

        self.did_forward_once = False

    def forward(
        self,
        patches: torch.Tensor,
        token_ids: torch.Tensor,
        window_size: int = 1,
        context_sequence_length: int = None,
        return_smooth_rank=False,
        return_tokenwise_loss=False,
        return_predictor_target_token_ids=False,
    ):
        """
        Einx notation:
        b: batch size
        s: full sequence length of both context and target patches
        ts: sequence length of tokens that are used as prediction targets
        xs: context sequence length
        r: batch repeat of the predictor
        rb: the total batch size (b * r) of the predictor
        ps: Predictor's input sequence length
        nd: the number of channels in the token ids, should be 4
        nwx: Number of windows in the context
        nwy: Number of windows in the target tokens to be predicted
        ws: Number of tokens per window = window_size**2
        """

        config = self.config
        device, dtype = patches.device, patches.dtype

        b, _, _ = patches.shape

        with torch.no_grad():
            target_hidden_states, *_ = self.ema_encoder(patches, token_ids)

        y_token_ids = token_ids
        if not config.should_predict_from_all_target:
            # Keep only the target hidden states from
            # patches absent from context
            target_hidden_states = target_hidden_states[:, context_sequence_length:, :]
            y_token_ids = token_ids[:, context_sequence_length:, :]

        b, target_sequence_length, _ = target_hidden_states.shape

        # Potentially do some post processing on the target hidden states
        if config.target_norm_mode == "layernorm":
            target_hidden_states = F.layer_norm(
                target_hidden_states, (target_hidden_states.shape[-1],)
            )

        elif config.target_norm_mode == "batchnorm":
            mask = y_token_ids[..., 0] != MASK_SAMPLE_ID
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
            mask = y_token_ids[..., 0] != MASK_SAMPLE_ID
            target_hidden_states = self.running_batchnorm(target_hidden_states, mask)

        elif config.target_norm_mode == "disabled":
            pass

        else:
            raise ValueError(config.target_norm_mode)

        smooth_rank = None
        if return_smooth_rank:
            smooth_rank = compute_smooth_rank(
                target_hidden_states.reshape(-1, target_hidden_states.shape[-1])
            )

        # Forward student with only the context patches
        x = patches[:, :context_sequence_length]
        x_token_ids = token_ids[:, :context_sequence_length]
        x, *_ = self.encoder(x, x_token_ids)

        # Window patches and get idx indicating context and target regions for the predictor
        # This expects that the input data is already arranged in windows
        # of window_size
        num_tokens_per_window = window_size**2
        x = einx.rearrange("b (nwx ws) d -> b nwx ws d", x, ws=num_tokens_per_window)
        x_token_ids = einx.rearrange(
            "b (nwx ws) nd -> b nwx ws nd", x_token_ids, ws=num_tokens_per_window
        )
        target_hidden_states = einx.rearrange(
            "b (nwy ws) d -> b nwy ws d", target_hidden_states, ws=num_tokens_per_window
        )
        y_token_ids = einx.rearrange(
            "b (nwy ws) nd -> b nwy ws nd", y_token_ids, ws=num_tokens_per_window
        )

        if config.should_attempt_mask_dropping:
            # Try to prevent the predictor from being given windows
            # that are filled entirely with padding tokens
            is_ctx_padding = x_token_ids[..., 0] == MASK_SAMPLE_ID
            is_ctx_padding = einx.all("b nwx [ws]", is_ctx_padding)
        else:
            is_ctx_padding = None

        ctx_num_windows = x.shape[1]
        if config.sample_predictor_context_with_replacement:
            # (rb k)
            context_idx = get_random_idx_with_replacement(
                r=config.predictor_batch_repeat,
                b=b,
                s=ctx_num_windows,
                capacity=config.predictor_context_capacity,
                mask=is_ctx_padding,
                device=device,
                dtype=dtype,
            )
        else:
            # (rb k)
            context_idx = get_random_idx_without_replacement(
                r=config.predictor_batch_repeat,
                b=b,
                s=ctx_num_windows,
                capacity=config.predictor_context_capacity,
                mask=is_ctx_padding,
                device=device,
                dtype=dtype,
            )

        if config.should_attempt_mask_dropping:
            # Try to prevent the predictor from being given windows
            # that are filled entirely with padding tokens
            is_target_padding = y_token_ids[..., 0] == MASK_SAMPLE_ID
            is_target_padding = einx.all("b nwy [ws]", is_target_padding)
        else:
            is_target_padding = None

        target_num_windows = target_hidden_states.shape[1]
        if config.sample_predictor_targets_with_replacement:
            # (rb m)
            target_idx = get_random_idx_with_replacement(
                r=config.predictor_batch_repeat,
                b=b,
                s=target_num_windows,
                capacity=config.predictor_target_capacity,
                mask=is_target_padding,
                device=device,
                dtype=dtype,
            )
        else:
            # (rb m)
            target_idx = get_random_idx_without_replacement(
                r=config.predictor_batch_repeat,
                b=b,
                s=target_num_windows,
                capacity=config.predictor_target_capacity,
                mask=is_target_padding,
                device=device,
                dtype=dtype,
            )

        # Repeat tensors for predictor
        x = einx.rearrange("b ... -> (r b) ...", x, r=config.predictor_batch_repeat)
        x_token_ids = einx.rearrange(
            "b ... -> (r b) ...", x_token_ids, r=config.predictor_batch_repeat
        )
        target_hidden_states = einx.rearrange(
            "b ... -> (r b) ...",
            target_hidden_states,
            r=config.predictor_batch_repeat,
        )
        y_token_ids = einx.rearrange(
            "b ... -> (r b) ...", y_token_ids, r=config.predictor_batch_repeat
        )

        # Gather subsets for the predictor

        # rb [nwx] ws d, rb k -> rb k ws d
        ctx = x.gather(1, expand_trailing(context_idx, x))
        # rb [nwy] ws d, rb m -> rb m ws d
        targets = target_hidden_states.gather(
            1, expand_trailing(target_idx, target_hidden_states)
        )

        # rb [nwx] ws nd, rb k -> rb k ws nd
        ctx_token_ids = x_token_ids.gather(1, expand_trailing(context_idx, x_token_ids))
        # rb [nwy] ws nd, rb m -> rb m ws nd
        target_token_ids = y_token_ids.gather(
            1, expand_trailing(target_idx, y_token_ids)
        )

        # Arrange windows back into flattened patches
        ctx = einx.rearrange("rb k ws d -> rb (k ws) d", ctx)
        ctx_token_ids = einx.rearrange("rb k ws nd -> rb (k ws) nd", ctx_token_ids)
        targets = einx.rearrange("rb m ws d -> rb (m ws) d", targets)
        target_token_ids = einx.rearrange(
            "rb k ws nd -> rb (k ws) nd", target_token_ids
        )

        _, num_ctx_tokens, _ = ctx_token_ids.shape
        _, num_target_tokens, _ = target_token_ids.shape

        # Prepare the inputs for the predictor
        x = torch.cat((ctx, targets), 1)
        combined_token_ids = torch.cat((ctx_token_ids, target_token_ids), 1)

        rb, ps, d = x.shape

        prediction_mask = torch.zeros(
            rb,
            ps,
            dtype=torch.bool,
            device=device,
        )
        prediction_mask[:, num_ctx_tokens:] = 1

        if not self.did_forward_once:
            print("predictor inputs:", x.shape)

        x = self.predictor(x, combined_token_ids, prediction_mask=prediction_mask)

        predictions = x[:, num_ctx_tokens:]

        loss = F.smooth_l1_loss(predictions, targets, reduction="none")

        # Exclude masked tokens from the loss
        target_sample_ids = target_token_ids[:, :, 0]
        is_target_mask = target_sample_ids != MASK_SAMPLE_ID

        if not config.should_predict_register_tokens:
            # Exclude register tokens from the loss
            target_register_ids = target_token_ids[:, :, 1]
            is_target_mask = is_target_mask & (target_register_ids == MASK_SAMPLE_ID)

        result_dict = dict(smooth_rank=smooth_rank)

        if return_tokenwise_loss:
            result_dict["tokenwise_loss"] = loss

        if return_predictor_target_token_ids:
            result_dict["predictor_target_token_ids"] = target_token_ids

        loss = loss[is_target_mask].mean()
        result_dict["loss"] = loss

        self.did_forward_once = True

        return result_dict
