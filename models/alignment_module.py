import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentModule(nn.Module):
    """
    Refine continuous embeddings and optionally supervise them with:

    1. InfoNCE contrastive loss between augmented embedding/value views.
    2. Trend loss tying embedding dynamics to first/second-order value changes.
    3. Continuity loss encouraging embedding increments to track signal magnitude.
    """

    def __init__(
        self,
        d_model,
        hidden_dim=None,
        temperature=0.1,
        dropout=0.1,
        augmentation_std=0.02,
        second_order_weight=0.5,
        continuity_weight=0.25,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.temperature = temperature
        self.augmentation_std = augmentation_std
        self.second_order_weight = second_order_weight
        self.continuity_weight = continuity_weight

        self.refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.trend_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.view_dropout = nn.Dropout(dropout)

    def _augment_embeddings(self, embeddings):
        noise = torch.randn_like(embeddings) * self.augmentation_std
        return self.view_dropout(embeddings + noise)

    def _augment_values(self, values):
        noise = torch.randn_like(values) * self.augmentation_std
        return values + noise

    def _info_nce_loss(self, view_a, view_b):
        query = F.normalize(view_a, dim=-1)
        key = F.normalize(view_b, dim=-1)
        logits = torch.matmul(query, key.transpose(0, 1)) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_ab + loss_ba)

    def _contrastive_loss(self, aligned_embeddings, values):
        batch_size, seq_len, _ = aligned_embeddings.shape
        if batch_size * seq_len <= 1:
            return aligned_embeddings.new_tensor(0.0)

        emb_view = self.projection(self._augment_embeddings(aligned_embeddings))
        value_view = self.projection(self.value_encoder(self._augment_values(values)))
        view_dim = emb_view.size(-1)
        emb_view = emb_view.reshape(-1, view_dim)
        value_view = value_view.reshape(-1, view_dim)
        return self._info_nce_loss(emb_view, value_view)

    def _trend_loss(self, aligned_embeddings, values):
        if values.size(1) < 2:
            return aligned_embeddings.new_tensor(0.0)

        trend_signal = self.trend_head(aligned_embeddings)
        pred_diff1 = trend_signal[:, 1:, :] - trend_signal[:, :-1, :]
        true_diff1 = values[:, 1:, :] - values[:, :-1, :]

        first_order_loss = F.smooth_l1_loss(pred_diff1, true_diff1)
        magnitude_loss = F.smooth_l1_loss(pred_diff1.abs(), true_diff1.abs())

        embed_delta_norm = (aligned_embeddings[:, 1:, :] - aligned_embeddings[:, :-1, :]).norm(dim=-1, keepdim=True)
        continuity_loss = F.smooth_l1_loss(embed_delta_norm, true_diff1.abs())

        second_order_loss = aligned_embeddings.new_tensor(0.0)
        if values.size(1) >= 3:
            pred_diff2 = pred_diff1[:, 1:, :] - pred_diff1[:, :-1, :]
            true_diff2 = true_diff1[:, 1:, :] - true_diff1[:, :-1, :]
            second_order_loss = F.smooth_l1_loss(pred_diff2, true_diff2)

        return (
            first_order_loss
            + 0.5 * magnitude_loss
            + self.second_order_weight * second_order_loss
            + self.continuity_weight * continuity_loss
        )

    def forward(
        self,
        embeddings,
        values=None,
        compute_losses=True,
        use_contrastive=True,
        use_trend=True,
    ):
        aligned_embeddings = embeddings + self.refine(embeddings)
        zero = aligned_embeddings.new_tensor(0.0)

        if values is None or not compute_losses:
            return aligned_embeddings, {"con_loss": zero, "trend_loss": zero}

        con_loss = self._contrastive_loss(aligned_embeddings, values) if use_contrastive else zero
        trend_loss = self._trend_loss(aligned_embeddings, values) if use_trend else zero
        return aligned_embeddings, {"con_loss": con_loss, "trend_loss": trend_loss}
