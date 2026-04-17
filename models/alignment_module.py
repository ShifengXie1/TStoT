import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentModule(nn.Module):
    """
    Project continuous embeddings into the GPT-2 token embedding space.

    The core alignment mapping follows:

        e_aligned = W_align * e_cont + b_align

    where the output dimension is constrained to the GPT-2 hidden size so the
    aligned embeddings can be passed to GPT-2 through `inputs_embeds`.

    During training the module can also provide:

    1. An InfoNCE contrastive loss between two stochastic views of the aligned
       embeddings.
    2. A trend continuity loss that matches first/second-order embedding
       differences to the underlying value differences.
    """

    def __init__(
        self,
        input_dim,
        hidden_size,
        projection_dim=None,
        temperature=0.1,
        dropout=0.1,
        augmentation_std=0.02,
        second_order_weight=0.5,
    ):
        super().__init__()
        projection_dim = projection_dim or hidden_size
        self.temperature = temperature
        self.augmentation_std = augmentation_std
        self.second_order_weight = second_order_weight

        self.align = nn.Linear(input_dim, hidden_size, bias=True)
        self.view_dropout = nn.Dropout(dropout)
        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        # Read aligned embeddings back to a scalar signal so temporal
        # differences can be compared directly with the original values.
        self.trend_readout = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.align.in_features == self.align.out_features:
            nn.init.eye_(self.align.weight)
        else:
            nn.init.xavier_uniform_(self.align.weight)
        nn.init.zeros_(self.align.bias)

    def _augment_embeddings(self, embeddings):
        if self.augmentation_std > 0:
            embeddings = embeddings + torch.randn_like(embeddings) * self.augmentation_std
        return self.view_dropout(embeddings)

    def _info_nce_loss(self, view_a, view_b):
        if view_a.size(0) <= 1:
            return view_a.new_tensor(0.0)

        query = F.normalize(view_a, dim=-1)
        key = F.normalize(view_b, dim=-1)
        logits = torch.matmul(query, key.transpose(0, 1)) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_ab + loss_ba)

    def _contrastive_loss(self, aligned_embeddings):
        batch_size, seq_len, _ = aligned_embeddings.shape
        if batch_size * seq_len <= 1:
            return aligned_embeddings.new_tensor(0.0)

        view_a = self.contrastive_head(self._augment_embeddings(aligned_embeddings))
        view_b = self.contrastive_head(self._augment_embeddings(aligned_embeddings))
        return self._info_nce_loss(
            view_a.reshape(-1, view_a.size(-1)),
            view_b.reshape(-1, view_b.size(-1)),
        )

    def _trend_loss(self, aligned_embeddings, values):
        if values is None or values.size(1) < 2:
            return aligned_embeddings.new_tensor(0.0)

        trend_signal = self.trend_readout(aligned_embeddings)
        pred_diff1 = trend_signal[:, 1:, :] - trend_signal[:, :-1, :]
        true_diff1 = values[:, 1:, :] - values[:, :-1, :]
        first_order_loss = F.smooth_l1_loss(pred_diff1, true_diff1)

        second_order_loss = aligned_embeddings.new_tensor(0.0)
        if values.size(1) >= 3:
            pred_diff2 = pred_diff1[:, 1:, :] - pred_diff1[:, :-1, :]
            true_diff2 = true_diff1[:, 1:, :] - true_diff1[:, :-1, :]
            second_order_loss = F.smooth_l1_loss(pred_diff2, true_diff2)

        return first_order_loss + self.second_order_weight * second_order_loss

    def forward(
        self,
        embeddings,
        values=None,
        compute_losses=True,
        use_contrastive=True,
        use_trend=True,
    ):
        aligned_embeddings = self.align(embeddings)
        zero = aligned_embeddings.new_tensor(0.0)

        if not compute_losses:
            return aligned_embeddings, {"con_loss": zero, "trend_loss": zero}

        con_loss = self._contrastive_loss(aligned_embeddings) if use_contrastive else zero
        trend_loss = self._trend_loss(aligned_embeddings, values) if use_trend else zero
        return aligned_embeddings, {"con_loss": con_loss, "trend_loss": trend_loss}
