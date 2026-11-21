"""M1-M4. Every model: forward(x, mask) -> logits (B, 8).

x:    (B, T, F) features for M1-M3, (B, N) waveform for M4
mask: (B, T) / (B, N) bool, True = real frame/sample. Padded positions never affect the output
      (checked by src.checks.check_mask_invariance).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils import N_CLASSES


# ============================================================================
# Shared pieces
# ============================================================================
def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


def lengths_to_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


class AttentionPooling(nn.Module):
    """alpha_t = softmax_t(u^T h_t),  z = sum_t alpha_t h_t   (Mirsamadi et al., 2017)."""

    def __init__(self, dim: int):
        super().__init__()
        self.u = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):
        scores = self.u(h).squeeze(-1).float().masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1).to(h.dtype)
        z = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return z, alpha


def run_lstm(lstm: nn.LSTM, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Packed LSTM: padded frames are never processed."""
    T = x.size(1)
    packed = pack_padded_sequence(x, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
    out, _ = lstm(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
    return out


# ============================================================================
# M1 — BiLSTM + attention pooling
# ============================================================================
class BiLSTMAttention(nn.Module):
    def __init__(self, in_dim, n_classes=N_CLASSES, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.pool = AttentionPooling(2 * hidden)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)
        self.last_attention = None

    def forward(self, x, mask):
        h = run_lstm(self.lstm, x, mask.sum(1))
        z, alpha = self.pool(h, mask)
        self.last_attention = alpha.detach()
        return self.fc(self.drop(z))


# ============================================================================
# M2 — CNN front-end + BiLSTM + attention pooling
# ============================================================================
class MaskedBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d on (B, C, F, T) whose TRAINING batch statistics ignore padded time steps.
    In eval mode it uses running statistics (element-wise, hence padding-invariant)."""

    def forward(self, x, mask):
        if not self.training:
            return super().forward(x)
        m = mask[:, None, None, :].to(x.dtype)
        n = m.sum() * x.size(2)
        mean = (x * m).sum(dim=(0, 2, 3)) / n
        var = (((x - mean[None, :, None, None]) ** 2) * m).sum(dim=(0, 2, 3)) / n
        with torch.no_grad():
            mom = self.momentum if self.momentum is not None else 0.1
            self.running_mean.mul_(1 - mom).add_(mom * mean.detach())
            unbiased = var.detach() * (n / (n - 1).clamp_min(1.0))
            self.running_var.mul_(1 - mom).add_(mom * unbiased)
            self.num_batches_tracked += 1
        xh = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return xh * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class ConvBlock(nn.Module):
    """Conv -> (masked) BatchNorm -> ReLU -> MaxPool, re-masking padded frames to zero."""

    def __init__(self, cin, cout, kernel=3, pool=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False)
        self.bn = MaskedBatchNorm2d(cout)
        self.pool = nn.MaxPool2d(pool)
        self.pool_t = pool[1]

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.bn(x, mask)
        x = F.relu(x)
        x = x * mask[:, None, None, :].to(x.dtype)
        x = self.pool(x)
        lengths = (mask.sum(1) // self.pool_t).clamp_min(1)
        mask = lengths_to_mask(lengths, x.size(-1))
        x = x * mask[:, None, None, :].to(x.dtype)
        return x, mask


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, in_dim, n_classes=N_CLASSES, conv_channels=(32, 64, 128), kernel=3,
                 hidden=128, dropout=0.3):
        super().__init__()
        blocks, c, f = [], 1, in_dim
        for co in conv_channels:
            blocks.append(ConvBlock(c, co, kernel, (2, 2)))
            c, f = co, f // 2
        self.blocks = nn.ModuleList(blocks)
        self.rnn_in = c * f
        self.drop_in = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.rnn_in, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling(2 * hidden)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)
        self.last_attention = None

    def forward(self, x, mask):
        h = (x * mask.unsqueeze(-1).to(x.dtype)).transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
        m = mask
        for blk in self.blocks:
            h, m = blk(h, m)
        B, C, Fd, T = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B, T, C * Fd)
        h = run_lstm(self.lstm, self.drop_in(h), m.sum(1))
        z, alpha = self.pool(h, m)
        self.last_attention = alpha.detach()
        return self.fc(self.drop(z))


# ============================================================================
# M3 — Transformer encoder from scratch
# ============================================================================
class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=4000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)


class TransformerSER(nn.Module):
    """2 strided convs (4x time reduction) -> Linear to d_model -> sinusoidal PE ->
    N pre-LN encoder layers (key-padding mask) -> masked mean-pool -> linear."""

    def __init__(self, in_dim, n_classes=N_CLASSES, d_model=256, n_heads=4, n_layers=4, ffn=512,
                 dropout=0.3, conv_channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, conv_channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, 3, stride=2, padding=1)
        self.proj = nn.Linear(conv_channels, d_model)
        self.pe = SinusoidalPE(d_model)
        self.drop_in = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    @staticmethod
    def _down(lengths):  # output length of Conv1d(k=3, s=2, p=1)
        return (lengths - 1) // 2 + 1

    def forward(self, x, mask):
        lengths = mask.sum(1)
        h = (x * mask.unsqueeze(-1).to(x.dtype)).transpose(1, 2)  # (B, F, T)
        for conv in (self.conv1, self.conv2):
            h = F.gelu(conv(h))
            lengths = self._down(lengths)
            m = lengths_to_mask(lengths, h.size(-1))
            h = h * m[:, None, :].to(h.dtype)
        h = self.drop_in(self.pe(self.proj(h.transpose(1, 2))))
        h = self.encoder(h, src_key_padding_mask=~m)
        h = self.norm(h)
        return self.fc(self.drop(masked_mean(h, m)))


# ============================================================================
# M4 — wav2vec 2.0 Base, fine-tuned
# ============================================================================
class Wav2Vec2SER(nn.Module):
    def __init__(self, n_classes=N_CLASSES, pretrained="facebook/wav2vec2-base", freeze_feature_encoder=True,
                 dropout=0.3, pass_attention_mask=False, config_overrides=None):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.w2v = Wav2Vec2Model.from_pretrained(pretrained, **(config_overrides or {}))
        if freeze_feature_encoder:
            self.w2v.freeze_feature_encoder()
        H = self.w2v.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(H, n_classes)
        self.pass_attention_mask = pass_attention_mask

    def head_parameters(self):
        return list(self.fc.parameters())

    def forward(self, x, mask):
        # per-utterance zero-mean / unit-variance over real samples (as Wav2Vec2FeatureExtractor does)
        m = mask.to(x.dtype)
        n = m.sum(1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(1, keepdim=True) / n
        var = (((x - mean) ** 2) * m).sum(1, keepdim=True) / n
        x = (x - mean) / torch.sqrt(var + 1e-7) * m
        am = mask.long() if self.pass_attention_mask else None
        h = self.w2v(x, attention_mask=am).last_hidden_state
        out_len = self.w2v._get_feat_extract_output_lengths(mask.sum(1)).long().clamp(1, h.size(1))
        fm = lengths_to_mask(out_len, h.size(1))
        return self.fc(self.drop(masked_mean(h, fm)))


# ============================================================================
def build_model(cfg: dict, in_dim: int | None = None) -> nn.Module:
    mc = cfg["model"]
    name = mc["name"]
    if name == "m1":
        return BiLSTMAttention(in_dim, N_CLASSES, hidden=mc["hidden"], layers=mc["layers"], dropout=mc["dropout"])
    if name == "m2":
        return CNNBiLSTMAttention(in_dim, N_CLASSES, conv_channels=tuple(mc["conv_channels"]),
                                  kernel=mc["kernel"], hidden=mc["hidden"], dropout=mc["dropout"])
    if name == "m3":
        return TransformerSER(in_dim, N_CLASSES, d_model=mc["d_model"], n_heads=mc["n_heads"],
                              n_layers=mc["n_layers"], ffn=mc["ffn"], dropout=mc["dropout"],
                              conv_channels=mc["conv_channels"])
    if name == "m4":
        return Wav2Vec2SER(N_CLASSES, pretrained=mc["pretrained"], freeze_feature_encoder=mc["freeze_feature_encoder"],
                           dropout=mc["dropout"], pass_attention_mask=mc["pass_attention_mask"],
                           config_overrides=mc.get("config_overrides") or {})
    raise ValueError(name)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
