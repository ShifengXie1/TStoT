import torch
import torch.nn as nn

from models.tokenizer import PatchTokenizer
from models.token_forecaster import TokenForecaster
from models.detokenizer import Detokenizer


class TokenLLMForecasting(nn.Module):
    """
    End-to-end: patch -> tokenize -> token forecasting -> detokenize -> forecast.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.vocab_size = configs.vocab_size
        self.d_model = configs.d_model
        self.n_layers = configs.n_layers
        self.n_heads = configs.n_heads
        self.dropout = getattr(configs, "dropout", 0.1)

        self.in_channels = configs.c_in
        self.out_channels = configs.c_out

        self.tokenizer = PatchTokenizer(
            patch_size=self.patch_size,
            stride=self.stride,
            in_channels=self.in_channels,
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            use_learnable_codebook=True,
        )

        self.token_forecaster = TokenForecaster(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            max_len=4096,
        )

        self.detokenizer = Detokenizer(
            codebook=self.tokenizer.codebook,
            patch_size=self.patch_size,
            stride=self.stride,
            out_channels=self.out_channels,
            d_model=self.d_model,
        )

    def _num_patches(self, length):
        if length < self.patch_size:
            return 0
        # TODO: consider padding to cover the tail when length is not aligned with stride.
        return 1 + (length - self.patch_size) // self.stride

    def forward(self, x, y=None, teacher_forcing=True):
        """
        x: [B, seq_len, C]
        y: [B, pred_len, C] (optional)
        Returns:
          forecast: [B, pred_len, C]
          token_logits: [B, N_pred, vocab]
          pred_token_ids: [B, N_pred]
          recon: [B, L_recon, C]
          aux: dict
        """
        past_token_ids, past_patch_emb, past_codebook_emb, vq_loss_past = self.tokenizer(x)

        future_token_ids = None
        vq_loss_future = torch.tensor(0.0, device=x.device)
        if y is not None:
            future_token_ids, _, _, vq_loss_future = self.tokenizer(y)

        pred_steps = self._num_patches(self.pred_len)
        token_logits, pred_token_ids = self.token_forecaster(
            past_token_ids,
            future_tokens=future_token_ids,
            pred_steps=pred_steps,
            teacher_forcing=teacher_forcing,
        )

        forecast, _ = self.detokenizer(pred_token_ids, target_len=self.pred_len)

        # Reconstruction for token reconstruction loss
        recon_past, _ = self.detokenizer(past_token_ids, target_len=self.seq_len)
        recon_future = None
        if future_token_ids is not None:
            recon_future, _ = self.detokenizer(future_token_ids, target_len=self.pred_len)

        aux = {
            "past_token_ids": past_token_ids,
            "future_token_ids": future_token_ids,
            "recon_past": recon_past,
            "recon_future": recon_future,
            "vq_loss": vq_loss_past + vq_loss_future,
            "past_patch_emb": past_patch_emb,
            "past_codebook_emb": past_codebook_emb,
        }
        return forecast, token_logits, pred_token_ids, recon_future, aux
