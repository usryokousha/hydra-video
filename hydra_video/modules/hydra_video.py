# This implementation of Hydra Video is an adaptation of the official Hydra implementation:
# https://github.com/goombalab/hydra
#
# Elements needed for autoregressive generation were adapted from the Mamba v2 implementation:
# https://github.com/state-spaces/mamba/tree/main

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from mamba.mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from mamba.mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

from typing import Optional, Tuple


class HydraVideo(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv_spatial: int = 7,
        d_conv_temporal: int = 4,
        conv_init: Optional[float] = None,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        learnable_init_states: bool = False,
        activation: str = "swish",
        bias: bool = False,
        conv_bias: bool = True,
        chunk_size: int = 256,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize the HydraVideo module.

        Args:
            d_model (int): Dimension of the model.
            d_state (int): Dimension of the state. Default: 64.
            d_conv_spatial (int): Size of the spatial convolutional kernel. Default: 7.
            d_conv_temporal (int): Size of the temporal convolutional kernel. Default: 7.
            conv_init (Optional[float]): Initialization range for convolutional weights. Default: None.
            expand (int): Expansion factor for inner dimension. Default: 2.
            headdim (int): Dimension of each head. Default: 64.
            ngroups (int): Number of groups for group linear layers. Default: 1.
            dt_min (float): Minimum value for delta t. Default: 0.001.
            dt_max (float): Maximum value for delta t. Default: 0.1.
            dt_init_floor (float): Minimum value for initial delta t. Default: 1e-4.
            dt_limit (Tuple[float, float]): Limits for delta t during inference. Default: (0.0, float("inf")).
            learnable_init_states (bool): Whether to learn initial states. Default: False.
            activation (str): Activation function to use. Default: "swish".
            bias (bool): Whether to use bias in linear layers. Default: False.
            conv_bias (bool): Whether to use bias in convolutional layers. Default: True.
            chunk_size (int): Size of chunks for processing. Default: 256.
            layer_idx (Optional[int]): Index of the layer. Default: None.
            device (Optional[torch.device]): Device to use for computations. Default: None.
            dtype (Optional[torch.dtype]): Data type to use for computations. Default: None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv_spatial = d_conv_spatial
        self.d_conv_temporal = d_conv_temporal
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = (
            2 * self.d_inner + 2 * (2 * self.ngroups * self.d_state) + 2 * self.nheads
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * (2 * self.ngroups * self.d_state)
        self.conv1d_spatial = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv_spatial,
            groups=conv_dim,
            padding=d_conv_spatial // 2,
            **factory_kwargs,
        )
        self.conv1d_temporal = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv_temporal,
            groups=conv_dim,
            padding=d_conv_temporal - 1,
            **factory_kwargs,
        )

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.init_states = nn.Parameter(
            torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs),
            requires_grad=self.learnable_init_states,
        )
        self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        A = torch.ones(self.nheads, dtype=torch.float32, device=device)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(
            self.d_inner, eps=1e-5, norm_before_gate=True, **factory_kwargs
        )

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(
        self,
        u: torch.Tensor,
        num_frames: int = 1,
        inference_params: Optional[dict] = None,
    ) -> torch.Tensor:

        batch, seqlen, dim = u.shape
        spatial_size = seqlen // num_frames

        initial_states = repeat(self.init_states, "... -> b ...", b=2 * batch)

        # Initialize states from previous inference
        ssm_state = None, None
        if inference_params is not None:
            ssm_state = self._get_states_from_cache(inference_params, batch, spatial_size)
            ssm_state[batch:].copy_(initial_states[batch:])

        # Output project for parameters Z, B, C, D, and dt
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)

        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * (2 * self.ngroups * self.d_state),
                2 * self.nheads,
            ],
            dim=-1,
        )

        # We flip only the frame sequence on the reverse sequence
        seq_idx_fwd = get_seq_idx(batch, seqlen, device=u.device)
        seq_idx_bwd = get_seq_idx(batch, seqlen, num_frames, device=u.device) + seq_idx_fwd.max() + 1
        seq_idx = torch.cat([seq_idx_fwd, seq_idx_bwd], dim=0)

        # Resulting shape: (2 * B, L, nheads)
        dt = torch.cat(
            [dt[:, :, : self.nheads], flip_frames(dt[:, :, self.nheads :], num_frames)],
            dim=0,
        )
        dt = F.softplus(dt + self.dt_bias)
        assert self.activation in ["swish", "silu"], "Only swish and silu are supported"

        # 1D Convolution over spatial dimension
        xBC = rearrange(xBC, "b (l f) d -> (b f) l d", f=num_frames)
        xBC = self.act(self.conv1d_spatial(xBC.transpose(1, 2)).transpose(1, 2))

        # 1D Convolution over temporal dimension
        xBC = rearrange(xBC, "(b f) l d -> (b l) f d", f=num_frames)
        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2),
            rearrange(self.conv1d_temporal.weight, "d 1 w -> d w"),
            bias=self.conv1d_temporal.bias,
            activation=self.activation,
        ).transpose(1, 2)
        xBC = rearrange(xBC, "(b l) f d -> b (l f) d", b=batch, f=num_frames)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, BC = torch.split(
            xBC, [self.d_inner, 2 * (2 * self.ngroups * self.d_state)], dim=-1
        )
        x_original = x

        # Flip the frames on the reverse sequence
        x = torch.cat([x, flip_frames(x, num_frames)], dim=0)
        BC = torch.cat(
            [
                BC[:, :, : 2 * self.ngroups * self.d_state],
                flip_frames(BC[:, :, 2 * self.ngroups * self.d_state :], num_frames),
            ],
            dim=0,
        )

        # Split B and C
        B, C = torch.split(
            BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        # seq_idx ensures processing of correct sequences
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=None,
            z=None,
            seq_idx=seq_idx,
            initial_states=ssm_state if ssm_state is not None else initial_states,
            return_final_states=ssm_state is not None,
            **dt_limit_kwargs,
        )
        # Deal with final output state
        if ssm_state is not None:
            y, last_state = y
            ssm_state[:batch].copy_(last_state[:batch])

        # We have to deal with flipped frames in the reverse sequence
        y_fwd, y_rev = torch.chunk(y, 2, dim=0)
        y_fwd = rearrange(y_fwd, "b l h p -> b l (h p)")
        y_rev = rearrange(y_rev, "b (l f) h p -> b l f (h p)", f=num_frames)

        y_fwd = torch.roll(y_fwd, shifts=1, dims=1)
        y_rev = torch.roll(y_rev, shifts=1, dims=1)

        y_fwd[:, 0, :] = 0.0
        y_rev[:, 0, ...] = 0.0

        y_rev = torch.flip(y_rev, (2,))
        y_rev = rearrange(y_rev, "b l f d -> b (l f) d", f=num_frames)

        x_res = x_original * repeat(
            F.linear(x_original, self.fc_D.weight, bias=self.D),
            "b l h -> b l (h p)",
            p=self.headdim,
        )

        y = y_fwd + y_rev + x_res
        y = self.norm(y, z)
        y = self.out_proj(y)
        return y

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:          
            ssm_state = torch.zeros(
                2 * batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                ssm_state,
            )
        else:
            ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                ssm_state.zero_()
        return ssm_state


def flip_frames(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    # reshape flip along frame sequence length
    x = rearrange(x, "b (l f) d -> b l f d", f=num_frames)
    x = torch.flip(x, dims=(2,))
    x = rearrange(x, "b l f d -> b (l f) d", f=num_frames)
    return x


def get_seq_idx(
    batch: int, seq_len: int, num_frames: Optional[int] = None, device=None
) -> torch.Tensor:
    if num_frames is not None:
        seq_idx = torch.arange(batch * num_frames, dtype=torch.int, device=device)
        seq_idx = seq_idx.repeat_interleave(seq_len // num_frames)
        seq_idx = seq_idx.reshape(batch, -1)
    else:
        seq_idx = torch.arange(batch, dtype=torch.int, device=device)
        seq_idx = seq_idx.repeat_interleave(seq_len)
        seq_idx = seq_idx.reshape(batch, -1)
    return seq_idx
