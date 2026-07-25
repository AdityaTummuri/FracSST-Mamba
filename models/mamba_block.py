"""Mamba state-space sequence model block wrappers.

Integrates Selective State Space (Mamba-SSM) mechanisms for efficient, long-context
sequential feature extraction from radar spectrogram time frames.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as NativeMamba

    HAS_NATIVE_MAMBA = True
except ImportError:
    HAS_NATIVE_MAMBA = False


class PyTorchSelectiveSSM(nn.Module):
    """Pure PyTorch implementation of Selective State-Space Model sequence block.

    Provides CPU and GPU cross-platform compatibility without external CUDA dependencies.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=0,
            groups=self.d_inner,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # S4 parameter initialization
        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32), self.d_inner
        ).view(self.d_inner, self.d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for selective SSM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Input projection -> splits into main path and gate path
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_path, z_path = xz.chunk(2, dim=-1)

        # 2. 1D Causal Convolution over sequence dimension
        x_trans = x_path.transpose(1, 2)  # (B, d_inner, T)
        x_padded = F.pad(x_trans, (self.d_conv - 1, 0))  # Causal padding
        x_conv = self.conv1d(x_padded).transpose(1, 2)  # (B, T, d_inner)
        x_act = self.act(x_conv)


        # 3. Selective parameter projections
        ssm_proj = self.x_proj(x_act)  # (B, T, d_state*2 + d_inner)
        delta, B, C = torch.split(
            ssm_proj, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, T, d_inner)

        # State transition matrix A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretize & recurse over sequence length T
        y_list = []
        h = torch.zeros(
            batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
        )

        for t in range(seq_len):
            dt_t = delta[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            B_t = B[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            C_t = C[:, t, :].unsqueeze(-1)  # (B, d_state, 1)
            x_t = x_act[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)

            # Discretized A_bar and B_bar
            dA = torch.exp(dt_t * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt_t * B_t  # (B, d_inner, d_state)

            h = dA * h + dB * x_t
            y_t = torch.matmul(h, C_t).squeeze(-1) + self.D * x_t.squeeze(-1)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # (B, T, d_inner)

        # 4. Gate multiplication & output projection
        out = y * self.act(z_path)
        return self.out_proj(out)


class MambaBlock(nn.Module):
    """Mamba State-Space Model block wrapper module.

    Args:
        d_model: Model feature dimension size.
        d_state: State dimension size for S4/Mamba recurrence.
        d_conv: 1D Convolution kernel width inside Mamba block.
        expand: Expansion factor for inner hidden channel dimension.
        dropout: Dropout rate applied after SSM block.
        use_native: Whether to use native C++ CUDA Mamba-SSM if available.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_native: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_rate = dropout

        self.norm = nn.LayerNorm(d_model)

        if use_native and HAS_NATIVE_MAMBA:
            self.ssm = NativeMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.ssm = PyTorchSelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection and LayerNorm.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        residual = x
        x_norm = self.norm(x)
        x_ssm = self.ssm(x_norm)
        x_drop = self.dropout(x_ssm)
        return residual + x_drop
