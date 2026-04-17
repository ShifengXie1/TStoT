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
    3. An explicit distribution-matching loss that pulls aligned time-series
       embeddings toward the pretrained GPT token embedding distribution.
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
        token_distribution_samples=256,
        token_distribution_bandwidth=1.0,
        token_moment_weight=0.1,
    ):
        super().__init__()
        projection_dim = projection_dim or hidden_size
        self.temperature = temperature
        self.augmentation_std = augmentation_std
        self.second_order_weight = second_order_weight
        self.token_distribution_samples = max(8, int(token_distribution_samples))
        self.token_distribution_bandwidth = float(token_distribution_bandwidth)
        self.token_moment_weight = float(token_moment_weight)

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

    @staticmethod
    def _sample_rows(matrix, num_samples):
        if matrix.size(0) <= num_samples:
            return matrix
        indices = torch.randperm(matrix.size(0), device=matrix.device)[:num_samples]
        return matrix.index_select(0, indices)

    def _rbf_kernel(self, x, y):
        dist2 = torch.cdist(x, y, p=2).pow(2)
        scale = max(1e-6, 2.0 * self.token_distribution_bandwidth * self.token_distribution_bandwidth * x.size(-1))
        return torch.exp(-dist2 / scale)

    def _token_distribution_loss(self, aligned_embeddings, token_embedding_matrix):
        if token_embedding_matrix is None:
            return aligned_embeddings.new_tensor(0.0)

        aligned_flat = aligned_embeddings.reshape(-1, aligned_embeddings.size(-1))
        if aligned_flat.size(0) <= 1:
            return aligned_embeddings.new_tensor(0.0)

        aligned_flat = self._sample_rows(aligned_flat, self.token_distribution_samples)
        token_embeddings = token_embedding_matrix.to(
            device=aligned_embeddings.device,
            dtype=aligned_embeddings.dtype,
        )
        token_embeddings = self._sample_rows(
            token_embeddings,
            max(self.token_distribution_samples, aligned_flat.size(0)),
        )

        aligned_flat = F.normalize(aligned_flat, dim=-1)
        token_embeddings = F.normalize(token_embeddings, dim=-1)

        k_xx = self._rbf_kernel(aligned_flat, aligned_flat).mean()
        k_yy = self._rbf_kernel(token_embeddings, token_embeddings).mean()
        k_xy = self._rbf_kernel(aligned_flat, token_embeddings).mean()
        mmd_loss = k_xx + k_yy - 2.0 * k_xy

        moment_loss = F.mse_loss(aligned_flat.mean(dim=0), token_embeddings.mean(dim=0))
        moment_loss = moment_loss + F.mse_loss(
            aligned_flat.std(dim=0, unbiased=False),
            token_embeddings.std(dim=0, unbiased=False),
        )
        return mmd_loss + self.token_moment_weight * moment_loss

    def forward(
        self,
        embeddings,
        values=None,
        token_embedding_matrix=None,
        compute_losses=True,
        use_contrastive=True,
        use_trend=True,
        use_token_distribution=True,
    ):
        aligned_embeddings = self.align(embeddings)
        zero = aligned_embeddings.new_tensor(0.0)

        if not compute_losses:
            return aligned_embeddings, {"con_loss": zero, "trend_loss": zero, "token_dist_loss": zero}

        con_loss = self._contrastive_loss(aligned_embeddings) if use_contrastive else zero
        trend_loss = self._trend_loss(aligned_embeddings, values) if use_trend else zero
        token_dist_loss = (
            self._token_distribution_loss(aligned_embeddings, token_embedding_matrix)
            if use_token_distribution
            else zero
        )
        return aligned_embeddings, {
            "con_loss": con_loss,
            "trend_loss": trend_loss,
            "token_dist_loss": token_dist_loss,
        }
