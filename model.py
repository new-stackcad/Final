"""
model.py  -  PhysicsLSTM surrogate with TCN road encoder.

Architecture
------------
Road [B,T,6]
  └─ TCN encoder (dilated causal convolutions, receptive field ~1s at 250Hz)
       → [B, T, road_dim=64]   rich temporal road features
                   |
                   ├─ concat ← ParamMLP [B, param_dim=128] broadcast
                   ↓
              [B, T, road_dim + param_dim]
                   ↓
              LSTM (chunked, 2 layers, hidden=256)
                   ↓
          StateHead → [B, T, 6]   states
          AccelHead → [B, T, 3]   accels (qdd_z_c, qdd_th_c, qdd_ph_c)

Why TCN fixes flat z_c
-----------------------
z_c bounce requires knowing road events 0.1-2.0s in the past.
A per-timestep MLP sees only the current road value so the LSTM
defaults to predicting the mean. A TCN with dilations [1,2,4,8,16,32]
gives a receptive field of ~253 timesteps (~1s at 250Hz stored rate),
enough to capture front-to-rear axle delays and suspension transient lag.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

from config import TRAIN_CFG


# ---------------------------------------------------------------
# TCN building block: dilated causal conv + residual
# ---------------------------------------------------------------

class CausalDilatedBlock(nn.Module):
    """
    One residual TCN block with causal left-padding.
    Input/output: [B, C, T]  (channels-first for Conv1d).
    """
    def __init__(self, channels: int, dilation: int,
                 kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel - 1)
        self.conv1  = nn.Conv1d(channels, channels, kernel,
                                padding=pad, dilation=dilation)
        self.crop   = pad
        self.norm1  = nn.GroupNorm(1, channels)
        self.conv2  = nn.Conv1d(channels, channels, kernel,
                                padding=pad, dilation=dilation)
        self.norm2  = nn.GroupNorm(1, channels)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal: left-pad then crop right to remove future leakage
        y = self.conv1(x)
        if self.crop > 0:
            y = y[..., :-self.crop]
        y = self.drop(self.act(self.norm1(y)))
        y = self.conv2(y)
        if self.crop > 0:
            y = y[..., :-self.crop]
        y = self.drop(self.act(self.norm2(y)))
        return y + x   # residual


class TCNRoadEncoder(nn.Module):
    """
    Temporal Convolutional Network road encoder.

    Dilations [1,2,4,8,16,32], kernel=3:
      Receptive field = sum(2*d*(k-1) for d in dilations) + 1
                      = 2*(1+2+4+8+16+32)*2 + 1 = 253 timesteps
    At 250 Hz stored rate → ~1.0 second of road context per output step.

    Input:  [B, T, 6]
    Output: [B, T, out_dim]   always contiguous
    """
    def __init__(self, in_ch: int = 6, out_dim: int = 64,
                 hidden: int = 64, dilations: list = None,
                 dropout: float = 0.10):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]
        self.input_proj  = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.blocks      = nn.ModuleList([
            CausalDilatedBlock(hidden, d, kernel=3, dropout=dropout)
            for d in dilations
        ])
        self.output_proj = nn.Conv1d(hidden, out_dim, kernel_size=1)
        self.act         = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 1).contiguous()      # [B, 6, T]
        y = self.act(self.input_proj(y))          # [B, hidden, T]
        for block in self.blocks:
            y = block(y)                          # [B, hidden, T]
        y = self.output_proj(y)                   # [B, out_dim, T]
        return y.permute(0, 2, 1).contiguous()    # [B, T, out_dim]


# ---------------------------------------------------------------
# Param MLP
# ---------------------------------------------------------------

class ParamMLP(nn.Module):
    def __init__(self, n_params: int = 8, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),   nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


# ---------------------------------------------------------------
# Output heads with residual skip
# ---------------------------------------------------------------

class ResidualHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))) + self.skip(x)


# ---------------------------------------------------------------
# Main surrogate model
# ---------------------------------------------------------------

class PhysicsLSTM(nn.Module):
    """
    TCN road encoder + parameter-conditioned LSTM surrogate.
    Chunked LSTM forward pass avoids cuDNN sequence-length limit.
    """

    CHUNK = 8192   # safe chunk size for all cuDNN versions

    def __init__(self,
                 road_dim:    int   = TRAIN_CFG["road_dim"],
                 param_dim:   int   = TRAIN_CFG["param_dim"],
                 lstm_hidden: int   = TRAIN_CFG["lstm_hidden"],
                 lstm_layers: int   = TRAIN_CFG["lstm_layers"],
                 dropout:     float = TRAIN_CFG["dropout"],
                 n_states:    int   = 6,
                 n_accels:    int   = 3):
        super().__init__()

        self.road_enc  = TCNRoadEncoder(
            in_ch=6, out_dim=road_dim, hidden=64,
            dilations=[1, 2, 4, 8, 16, 32], dropout=dropout)

        self.param_mlp = ParamMLP(n_params=8, hidden=64, out_dim=param_dim)

        lstm_in = road_dim + param_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.h0_proj = nn.Linear(param_dim, lstm_hidden * lstm_layers)
        self.c0_proj = nn.Linear(param_dim, lstm_hidden * lstm_layers)

        self.state_head = ResidualHead(lstm_hidden, n_states)
        self.accel_head = ResidualHead(lstm_hidden, n_accels)

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self,
                road:  torch.Tensor,   # [B, T, 6]
                param: torch.Tensor,   # [B, 8]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = road.shape

        # TCN: each timestep gets ~1s of road context
        road_feat = self.road_enc(road)                        # [B, T, road_dim]

        # Param embedding broadcast over T
        p_emb = self.param_mlp(param)                          # [B, param_dim]
        p_rep = p_emb.unsqueeze(1).repeat(1, T, 1)             # [B, T, param_dim]

        lstm_in = torch.cat([road_feat, p_rep], dim=2).contiguous()

        # Init hidden state from param embedding
        h = (self.h0_proj(p_emb)
              .view(B, self.lstm_layers, self.lstm_hidden)
              .permute(1, 0, 2).contiguous())
        c = (self.c0_proj(p_emb)
              .view(B, self.lstm_layers, self.lstm_hidden)
              .permute(1, 0, 2).contiguous())

        # Chunked LSTM: avoids cuDNN sequence-length limit at 116k steps
        out_chunks = []
        for start in range(0, T, self.CHUNK):
            end   = min(start + self.CHUNK, T)
            chunk = lstm_in[:, start:end, :].contiguous()
            out_c, (h, c) = self.lstm(chunk, (h, c))
            out_chunks.append(out_c)
            h = h.contiguous()
            c = c.contiguous()

        lstm_out = torch.cat(out_chunks, dim=1)                # [B, T, hidden]

        state_pred = self.state_head(lstm_out)                 # [B, T, 6]
        accel_pred = self.accel_head(lstm_out)                 # [B, T, 3]

        return state_pred, accel_pred

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------
# Physics-informed loss
# ---------------------------------------------------------------

class PhysicsLoss(nn.Module):
    """
    Combined loss:
      L = lambda_s * weighted_MSE(state)      cabin DOFs weighted higher
        + lambda_a * weighted_MSE(accel)      cabin accels weighted higher
        + lambda_r * MSE(ISO2631_RMS)         physical RMS in m/s²
        + lambda_p * FD_consistency           z_c accel vs finite-diff
        + 2.0      * AC_amplitude_loss        KEY FIX for flat z_c

    AC amplitude loss:
      Compares std(pred_z_c, pred_th_c, pred_ph_c) vs std(true) over
      the full sequence. If the surrogate predicts flat z_c, its std=0
      while true std>0 → large gradient that forces dynamic prediction.
    """

    def __init__(self,
                 lambda_state: float = TRAIN_CFG["lambda_state"],
                 lambda_accel: float = TRAIN_CFG["lambda_accel"],
                 lambda_rms:   float = TRAIN_CFG["lambda_rms"],
                 lambda_phys:  float = TRAIN_CFG["lambda_phys"]):
        super().__init__()
        self.ls = lambda_state
        self.la = lambda_accel
        self.lr = lambda_rms
        self.lp = lambda_phys

    def forward(self,
                state_pred:  torch.Tensor,
                accel_pred:  torch.Tensor,
                state_true:  torch.Tensor,
                accel_true:  torch.Tensor,
                rms_true:    torch.Tensor,
                h_seat:      float = 0.1,
                dt:          float = 0.004,
                accel_raw:   torch.Tensor = None,
                a_std:       torch.Tensor = None,
                ) -> Tuple[torch.Tensor, dict]:

        mse = nn.functional.mse_loss

        # Weighted state MSE: z_c=3x, th_c=2x, ph_c=2x, sprung=1x
        w_s    = state_pred.new_tensor([3., 2., 2., 1., 1., 1.]).view(1, 1, 6)
        l_state = (w_s * (state_pred - state_true)**2).mean()

        # Weighted accel MSE: qdd_z_c=3x, qdd_th_c=2x, qdd_ph_c=2x
        w_a    = accel_pred.new_tensor([3., 2., 2.]).view(1, 1, 3)
        l_accel = (w_a * (accel_pred - accel_true)**2).mean()

        # AC amplitude loss: std of cabin state predictions vs truth
        # This directly penalises flat predictions for z_c
        pred_std = state_pred[:, :, :3].std(dim=1)   # [B, 3]
        true_std = state_true[:, :, :3].std(dim=1)   # [B, 3]
        l_ac    = mse(pred_std, true_std)

        # ISO 2631 physical RMS (denormalised)
        if a_std is not None:
            a_std_b  = a_std.unsqueeze(1)             # [B, 1, 3]
            ap_phys  = accel_pred * a_std_b            # [B, T, 3]
            az = ap_phys[..., 0]
            ax = -h_seat * ap_phys[..., 1]
            ay =  h_seat * ap_phys[..., 2]
            rms_pred = torch.sqrt(
                (az**2).mean(dim=1, keepdim=True) +
                (ax**2).mean(dim=1, keepdim=True) +
                (ay**2).mean(dim=1, keepdim=True) + 1e-8)
        else:
            rms_pred = torch.sqrt(
                (accel_pred[..., 0]**2).mean(dim=1, keepdim=True) + 1e-8)
            rms_true = rms_true / (rms_true.mean() + 1e-8)

        l_rms = mse(rms_pred, rms_true)

        # Finite-difference consistency on z_c
        l_phys = torch.zeros(1, device=state_pred.device)[0]
        if state_pred.shape[1] > 2:
            q_zc   = state_pred[:, :, 0]
            fd     = (q_zc[:, 2:] - 2*q_zc[:, 1:-1] + q_zc[:, :-2]) / (dt**2)
            l_phys = mse(fd, accel_pred[:, 1:-1, 0])

        total = (self.ls * l_state
               + self.la * l_accel
               + self.lr * l_rms
               + self.lp * l_phys
               + 2.0     * l_ac)

        return total, {
            "loss_total": total.item(),
            "loss_state": l_state.item(),
            "loss_accel": l_accel.item(),
            "loss_rms":   l_rms.item(),
            "loss_phys":  l_phys.item(),
            "loss_ac":    l_ac.item(),
        }
