import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentModule(nn.Module):
    """
    Optional CT-GPT2 alignment module with InfoNCE and trend losses.
    """

    def __init__(self, d_model, hidden_dim=None, temperature=0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.temperature = temperature

        self.refine = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.trend_head = nn.Linear(d_model, 1)

    def _contrastive_loss(self, aligned_embeddings, values):
        batch_size, seq_len, d_model = aligned_embeddings.shape
        if batch_size * seq_len <= 1:
            return aligned_embeddings.new_tensor(0.0)

        query = F.normalize(aligned_embeddings.reshape(-1, d_model), dim=-1)
        key = F.normalize(self.value_encoder(values).reshape(-1, d_model), dim=-1)
        logits = torch.matmul(query, key.transpose(0, 1)) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _trend_loss(self, aligned_embeddings, values):
        if values.size(1) < 2:
            return aligned_embeddings.new_tensor(0.0)

        aligned_scalar = self.trend_head(aligned_embeddings)
        pred_diff1 = aligned_scalar[:, 1:, :] - aligned_scalar[:, :-1, :]
        true_diff1 = values[:, 1:, :] - values[:, :-1, :]
        loss_first = F.mse_loss(pred_diff1, true_diff1)

        if values.size(1) < 3:
            return loss_first

        pred_diff2 = pred_diff1[:, 1:, :] - pred_diff1[:, :-1, :]
        true_diff2 = true_diff1[:, 1:, :] - true_diff1[:, :-1, :]
        loss_second = F.mse_loss(pred_diff2, true_diff2)
        return 0.5 * (loss_first + loss_second)

    def forward(self, embeddings, values=None, compute_losses=True):
        aligned_embeddings = embeddings + self.refine(embeddings)
        if values is None or not compute_losses:
            zero = aligned_embeddings.new_tensor(0.0)
            return aligned_embeddings, {"con_loss": zero, "trend_loss": zero}

        return aligned_embeddings, {
            "con_loss": self._contrastive_loss(aligned_embeddings, values),
            "trend_loss": self._trend_loss(aligned_embeddings, values),
        }
