import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrendAwarePatchEmbedding(nn.Module):
    """
    Encode time-series into patch embeddings while preserving local level,
    first-order differences, and curvature.
    """

    def __init__(self, patch_size, stride, d_model, max_patches, dropout=0.1):
        super().__init__()
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.max_patches = int(max_patches)

        self.raw_projection = nn.Sequential(
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.stat_projection = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.position_embedding = nn.Embedding(self.max_patches, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def num_patches_for_length(length, patch_size, stride):
        length = int(length)
        patch_size = int(patch_size)
        stride = int(stride)
        if length <= patch_size:
            return 1
        return 1 + math.ceil((length - patch_size) / stride)

    def _pad_length(self, length):
        num_patches = self.num_patches_for_length(length, self.patch_size, self.stride)
        padded_length = self.patch_size + (num_patches - 1) * self.stride
        return num_patches, padded_length

    def patchify(self, series):
        if series.dim() == 2:
            series = series.unsqueeze(-1)
        if series.dim() != 3 or series.size(-1) != 1:
            raise ValueError(
                "TrendAwarePatchEmbedding expects [batch, seq_len, 1], "
                f"but received {tuple(series.shape)}."
            )

        values = series.squeeze(-1)
        batch_size, original_length = values.shape
        num_patches, padded_length = self._pad_length(original_length)
        pad_right = padded_length - original_length
        if pad_right > 0:
            tail = values[:, -1:].expand(batch_size, pad_right)
            values = torch.cat([values, tail], dim=1)

        patches = values.unfold(dimension=1, size=self.patch_size, step=self.stride)
        if patches.size(1) != num_patches:
            raise RuntimeError(
                f"Expected {num_patches} patches but received {patches.size(1)}. "
                "Patchify bookkeeping is inconsistent."
            )
        return patches.contiguous(), {
            "orig_length": original_length,
            "padded_length": padded_length,
            "num_patches": num_patches,
        }

    def compute_patch_stats(self, patches):
        level = patches.mean(dim=-1, keepdim=True)
        last = patches[..., -1:].contiguous()
        diff = patches[..., 1:] - patches[..., :-1]
        slope = diff.mean(dim=-1, keepdim=True) if diff.size(-1) > 0 else torch.zeros_like(level)
        if diff.size(-1) >= 2:
            curve = (diff[..., 1:] - diff[..., :-1]).mean(dim=-1, keepdim=True)
        else:
            curve = torch.zeros_like(level)
        stats = torch.cat([level, last, slope, curve], dim=-1)
        centered = patches - level
        return centered, stats

    def encode(self, series, position_offset=0):
        patches, patch_meta = self.patchify(series)
        latent = self.encode_patch_values(patches, position_offset=position_offset)
        return latent, {
            "patches": patches,
            "patch_stats": self.compute_patch_stats(patches)[1],
            "patch_meta": patch_meta,
        }

    def encode_patch_values(self, patches, position_offset=0):
        if patches.dim() != 3 or patches.size(-1) != self.patch_size:
            raise ValueError(
                f"Patch values must have shape [batch, num_patches, {self.patch_size}], "
                f"but received {tuple(patches.shape)}."
            )
        centered, stats = self.compute_patch_stats(patches)

        num_patches = patches.size(1)
        position_ids = torch.arange(
            position_offset,
            position_offset + num_patches,
            device=patches.device,
        ).unsqueeze(0)
        if position_ids.max().item() >= self.max_patches:
            raise ValueError(
                f"Patch position {position_ids.max().item()} exceeds max_patches={self.max_patches}."
            )

        raw_embed = self.raw_projection(centered)
        stat_embed = self.stat_projection(stats)
        pos_embed = self.position_embedding(position_ids)
        return self.output_norm(raw_embed + stat_embed + pos_embed)

    def overlap_add(self, patch_values, target_length):
        if patch_values.dim() != 3:
            raise ValueError(
                "Patch values must have shape [batch, num_patches, patch_size], "
                f"but received {tuple(patch_values.shape)}."
            )

        batch_size, num_patches, patch_size = patch_values.shape
        total_length = patch_size + (num_patches - 1) * self.stride
        recon = patch_values.new_zeros(batch_size, total_length)
        weight = patch_values.new_zeros(batch_size, total_length)

        for patch_idx in range(num_patches):
            start = patch_idx * self.stride
            end = start + patch_size
            recon[:, start:end] += patch_values[:, patch_idx, :]
            weight[:, start:end] += 1.0

        recon = recon / weight.clamp_min(1.0)
        return recon[:, :target_length].unsqueeze(-1)


class TrendAwarePatchDecoder(nn.Module):
    """
    Decode decompensated latent patch states back into patch values.
    """

    def __init__(self, d_model, patch_size, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.patch_size = int(patch_size)

        self.base_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.patch_size),
        )

        time_index = torch.linspace(-1.0, 1.0, steps=self.patch_size)
        self.register_buffer("time_index", time_index.view(1, 1, self.patch_size), persistent=False)

    def forward(self, latent_patches):
        base_params = self.base_head(latent_patches)
        level = base_params[..., 0:1]
        slope = base_params[..., 1:2]
        curve = base_params[..., 2:3]

        base = level + slope * self.time_index + curve * self.time_index.pow(2)
        residual = self.residual_head(latent_patches)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        return base + residual
