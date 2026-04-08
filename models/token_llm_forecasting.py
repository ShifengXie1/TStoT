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
        self.max_token_len = self._num_patches(self.seq_len) + self._num_patches(self.pred_len)

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
            max_len=max(2, self.max_token_len),
            model_name=getattr(configs, "gpt_model_name", "openai-community/gpt2"),
            local_model_path=getattr(configs, "gpt_local_path", "./gpt"),
            use_pretrained=getattr(configs, "use_pretrained_gpt2", True),
            prefer_local=getattr(configs, "prefer_local_gpt2", True),
            local_files_only=getattr(configs, "gpt_local_files_only", True),
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
        remaining = max(0, length - self.patch_size)
        return 1 + (remaining + self.stride - 1) // self.stride

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
        past_token_ids, _, past_quantized_emb, vq_loss_past = self.tokenizer(x)

        future_token_ids = None
        future_quantized_emb = None
        vq_loss_future = torch.tensor(0.0, device=x.device)
        if y is not None:
            future_token_ids, _, future_quantized_emb, vq_loss_future = self.tokenizer(y)

        pred_steps = self._num_patches(self.pred_len)
        token_logits, pred_token_ids, token_loss = self.token_forecaster(
            past_token_ids,
            future_tokens=future_token_ids,
            pred_steps=pred_steps,
            teacher_forcing=teacher_forcing,
        )

        if teacher_forcing and token_logits is not None and future_token_ids is not None:
            forecast, _ = self.detokenizer.logits_to_sequence(
                token_logits,
                target_len=self.pred_len,
            )
        else:
            forecast, _ = self.detokenizer(pred_token_ids, target_len=self.pred_len)

        recon_past, _ = self.detokenizer.embeddings_to_sequence(
            past_quantized_emb,
            target_len=self.seq_len,
        )
        recon_future = None
        if future_quantized_emb is not None:
            recon_future, _ = self.detokenizer.embeddings_to_sequence(
                future_quantized_emb,
                target_len=self.pred_len,
            )

        aux = {
            "past_token_ids": past_token_ids,
            "future_token_ids": future_token_ids,
            "recon_past": recon_past,
            "vq_loss": vq_loss_past + vq_loss_future,
            "token_loss": token_loss,
        }
        return forecast, token_logits, pred_token_ids, recon_future, aux
