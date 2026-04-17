import torch
import torch.nn as nn
import torch.nn.functional as F


class CompensationAlignmentModule(nn.Module):
    """
    Align time-series patch latents to the pretrained token embedding space via
    an explicit affine compensation:

        u = s(z) * z + b(z)

    GPT operates on `u`. Predicted GPT states are decompensated back to the
    time-series latent space before patch decoding.
    """

    def __init__(
        self,
        hidden_size,
        projection_dim=None,
        temperature=0.1,
        dropout=0.1,
        token_distribution_samples=256,
        token_topk=64,
        compensation_limit=0.25,
        token_moment_weight=0.1,
    ):
        super().__init__()
        projection_dim = projection_dim or hidden_size
        self.temperature = float(temperature)
        self.token_distribution_samples = max(8, int(token_distribution_samples))
        self.token_topk = max(8, int(token_topk))
        self.compensation_limit = float(compensation_limit)
        self.token_moment_weight = float(token_moment_weight)

        self.scale_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.bias_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.inverse_scale_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.inverse_bias_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.view_dropout = nn.Dropout(dropout)

    def _bounded_scale(self, tensor):
        return 1.0 + self.compensation_limit * torch.tanh(tensor)

    def _bounded_bias(self, tensor):
        return self.compensation_limit * torch.tanh(tensor)

    def compensate(self, latent):
        scale = self._bounded_scale(self.scale_head(latent))
        bias = self._bounded_bias(self.bias_head(latent))
        aligned = scale * latent + bias
        return aligned, {"scale": scale, "bias": bias}

    def decompensate(self, aligned_latent):
        scale = self._bounded_scale(self.inverse_scale_head(aligned_latent)).clamp_min(1e-3)
        bias = self._bounded_bias(self.inverse_bias_head(aligned_latent))
        latent = (aligned_latent - bias) / scale
        return latent, {"inv_scale": scale, "inv_bias": bias}

    @staticmethod
    def _sample_rows(matrix, num_samples):
        if matrix.size(0) <= num_samples:
            return matrix
        indices = torch.randperm(matrix.size(0), device=matrix.device)[:num_samples]
        return matrix.index_select(0, indices)

    def _augment(self, embeddings):
        noise = torch.randn_like(embeddings) * 0.01
        return self.view_dropout(embeddings + noise)

    def _contrastive_loss(self, aligned_embeddings):
        flat = aligned_embeddings.reshape(-1, aligned_embeddings.size(-1))
        if flat.size(0) <= 1:
            return aligned_embeddings.new_tensor(0.0)

        view_a = F.normalize(self.contrastive_head(self._augment(flat)), dim=-1)
        view_b = F.normalize(self.contrastive_head(self._augment(flat)), dim=-1)
        logits = view_a @ view_b.transpose(0, 1) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_ab + loss_ba)

    def _token_distribution_loss(self, aligned_embeddings, token_embedding_matrix):
        if token_embedding_matrix is None:
            return aligned_embeddings.new_tensor(0.0)

        aligned_flat = aligned_embeddings.reshape(-1, aligned_embeddings.size(-1))
        if aligned_flat.size(0) == 0:
            return aligned_embeddings.new_tensor(0.0)

        aligned_flat = self._sample_rows(aligned_flat, self.token_distribution_samples)
        token_embeddings = token_embedding_matrix.to(
            device=aligned_embeddings.device,
            dtype=aligned_embeddings.dtype,
        )
        token_embeddings = self._sample_rows(
            token_embeddings,
            max(self.token_distribution_samples * 4, self.token_topk),
        )

        aligned_norm = F.normalize(aligned_flat, dim=-1)
        token_norm = F.normalize(token_embeddings, dim=-1)
        logits = aligned_norm @ token_norm.transpose(0, 1) / self.temperature
        topk = min(self.token_topk, logits.size(-1))
        top_values, top_indices = torch.topk(logits, k=topk, dim=-1)
        top_token_embeddings = token_embeddings.index_select(0, top_indices.reshape(-1)).view(
            aligned_flat.size(0),
            topk,
            -1,
        )
        weights = torch.softmax(top_values, dim=-1).unsqueeze(-1)
        token_proto = (weights * top_token_embeddings).sum(dim=1)

        proto_loss = F.mse_loss(aligned_flat, token_proto)
        moment_loss = F.mse_loss(aligned_flat.mean(dim=0), token_embeddings.mean(dim=0))
        moment_loss = moment_loss + F.mse_loss(
            aligned_flat.std(dim=0, unbiased=False),
            token_embeddings.std(dim=0, unbiased=False),
        )
        return proto_loss + self.token_moment_weight * moment_loss

    def forward(
        self,
        latent,
        token_embedding_matrix=None,
        compute_losses=True,
        use_contrastive=True,
        use_token_distribution=True,
    ):
        aligned, comp = self.compensate(latent)
        zero = aligned.new_tensor(0.0)
        if not compute_losses:
            return aligned, {
                "con_loss": zero,
                "token_dist_loss": zero,
                "comp_reg_loss": zero,
                "scale": comp["scale"],
                "bias": comp["bias"],
            }

        con_loss = self._contrastive_loss(aligned) if use_contrastive else zero
        token_dist_loss = (
            self._token_distribution_loss(aligned, token_embedding_matrix)
            if use_token_distribution
            else zero
        )
        comp_reg_loss = (comp["scale"] - 1.0).pow(2).mean() + comp["bias"].pow(2).mean()
        return aligned, {
            "con_loss": con_loss,
            "token_dist_loss": token_dist_loss,
            "comp_reg_loss": comp_reg_loss,
            "scale": comp["scale"],
            "bias": comp["bias"],
        }
