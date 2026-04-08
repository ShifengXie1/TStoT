import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer(nn.Module):
    """
    Patchify time series and quantize each patch into a discrete token id.
    """
    def __init__(self, patch_size, stride, in_channels, d_model, vocab_size, use_learnable_codebook=True):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_learnable_codebook = use_learnable_codebook

        self.patch_proj = nn.Linear(patch_size * in_channels, d_model)

        if use_learnable_codebook:
            self.codebook = nn.Embedding(vocab_size, d_model)
        else:
            self.register_buffer("codebook", torch.randn(vocab_size, d_model))

    def patchify(self, x):
        """
        x: [B, L, C]
        Returns: patches [B, N, patch_size, C]
        """
        b, l, c = x.shape
        if l < self.patch_size:
            raise ValueError(f"seq_len {l} < patch_size {self.patch_size}")

        remaining = max(0, l - self.patch_size)
        num_patches = 1 + (remaining + self.stride - 1) // self.stride
        total_len = self.patch_size + (num_patches - 1) * self.stride
        pad_len = total_len - l
        if pad_len > 0:
            # Replicate the tail so the last patch can still cover the sequence end.
            x = x.transpose(1, 2)
            x = F.pad(x, (0, pad_len), mode="replicate")
            x = x.transpose(1, 2).contiguous()

        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        # unfold returns [B, N, C, patch_size]; move to [B, N, patch_size, C]
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches

    def quantize(self, patch_emb):
        """
        patch_emb: [B, N, D]
        Returns:
          token_ids: [B, N]
          quantized: [B, N, D]
          vq_loss: scalar
        """
        b, n, d = patch_emb.shape
        flat = patch_emb.reshape(b * n, d)

        if isinstance(self.codebook, nn.Embedding):
            codebook = self.codebook.weight
        else:
            codebook = self.codebook

        # L2 distance to codebook entries
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        token_ids = torch.argmin(dist, dim=1)
        quantized = codebook[token_ids].view(b, n, d)

        # VQ-VAE style loss (codebook + commitment)
        codebook_loss = F.mse_loss(quantized, patch_emb.detach())
        commitment_loss = F.mse_loss(patch_emb, quantized.detach())
        vq_loss = codebook_loss + commitment_loss

        # Straight-through estimator
        quantized_st = patch_emb + (quantized - patch_emb).detach()
        return token_ids.view(b, n), quantized_st, vq_loss

    def forward(self, x):
        """
        x: [B, L, C]
        Returns:
          token_ids: [B, N]
          patch_embeddings: [B, N, D]
          codebook_embeddings: [B, N, D]
          vq_loss: scalar
        """
        patches = self.patchify(x)
        b, n, p, c = patches.shape
        patch_flat = patches.reshape(b, n, p * c)
        patch_emb = self.patch_proj(patch_flat)

        token_ids, quantized, vq_loss = self.quantize(patch_emb)
        return token_ids, patch_emb, quantized, vq_loss
