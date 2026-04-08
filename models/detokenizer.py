import torch
import torch.nn as nn
import torch.nn.functional as F


class Detokenizer(nn.Module):
    """
    Map token ids back to patch values and stitch into time series.
    """
    def __init__(self, codebook, patch_size, stride, out_channels, d_model):
        super().__init__()
        self.codebook = codebook
        self.patch_size = patch_size
        self.stride = stride
        self.out_channels = out_channels
        self.d_model = d_model
        self.patch_recon = nn.Linear(d_model, patch_size * out_channels)

    def _get_codebook_weight(self):
        if isinstance(self.codebook, nn.Embedding):
            return self.codebook.weight
        return self.codebook

    def tokens_to_patches(self, token_ids):
        codebook = self._get_codebook_weight()
        emb = codebook[token_ids]
        return self.embeddings_to_patches(emb)

    def embeddings_to_patches(self, emb):
        b, n, _ = emb.shape
        patch_flat = self.patch_recon(emb)
        patches = patch_flat.view(b, n, self.patch_size, self.out_channels)
        return patches

    def patches_to_sequence(self, patches, target_len=None):
        """
        patches: [B, N, patch_size, C]
        Returns: [B, L, C]
        """
        b, n, p, c = patches.shape
        total_len = self.patch_size + (n - 1) * self.stride
        seq = torch.zeros(b, total_len, c, device=patches.device)
        counts = torch.zeros(b, total_len, c, device=patches.device)
        for i in range(n):
            start = i * self.stride
            end = start + self.patch_size
            seq[:, start:end, :] += patches[:, i, :, :]
            counts[:, start:end, :] += 1.0
        seq = seq / torch.clamp(counts, min=1.0)
        if target_len is not None:
            seq = seq[:, :target_len, :]
        return seq

    def embeddings_to_sequence(self, emb, target_len=None):
        patches = self.embeddings_to_patches(emb)
        seq = self.patches_to_sequence(patches, target_len=target_len)
        return seq, patches

    def logits_to_sequence(self, token_logits, target_len=None, temperature=1.0):
        probs = F.softmax(token_logits / temperature, dim=-1)
        emb = probs @ self._get_codebook_weight()
        return self.embeddings_to_sequence(emb, target_len=target_len)

    def forward(self, token_ids, target_len=None):
        patches = self.tokens_to_patches(token_ids)
        seq = self.patches_to_sequence(patches, target_len=target_len)
        return seq, patches
