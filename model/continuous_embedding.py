import torch
import torch.nn as nn


class ContinuousEmbedding(nn.Module):
    """
    Map normalized scalar time-series values to GPT-2 compatible embeddings.

    This module implements:

        e_i = W_x * x_i + P_i + b

    where `W_x` is a learned projection, `P_i` is a learned position embedding,
    and `b` is a learned bias.
    """

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.value_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, position_offset=0):
        if x.dim() != 3 or x.size(-1) != 1:
            raise ValueError(
                "ContinuousEmbedding expects input shape [batch, seq_len, 1], "
                f"but received {tuple(x.shape)}."
            )

        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}.")

        if isinstance(position_offset, torch.Tensor):
            position_offset = int(position_offset.item())

        position_ids = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=x.device,
        ).unsqueeze(0)
        if position_ids.max().item() >= self.max_seq_len:
            raise ValueError(
                f"Position id {position_ids.max().item()} exceeds max_seq_len={self.max_seq_len}."
            )

        value_embeds = self.value_projection(x)
        position_embeds = self.position_embedding(position_ids)
        return value_embeds + position_embeds + self.bias.view(1, 1, -1)
