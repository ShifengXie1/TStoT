import torch
import torch.nn as nn


class TokenForecaster(nn.Module):
    """
    GPT-style decoder for autoregressive token prediction.
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, dropout, max_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def _forward_tokens(self, token_ids):
        b, t = token_ids.shape
        if t > self.max_len:
            raise ValueError(f"token length {t} > max_len {self.max_len}")
        pos = torch.arange(t, device=token_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        mask = self._causal_mask(t, token_ids.device)
        h = self.decoder(x, mask=mask)
        logits = self.lm_head(h)
        return logits

    def forward(self, history_tokens, future_tokens=None, pred_steps=None, teacher_forcing=True):
        """
        history_tokens: [B, H]
        future_tokens: [B, F] (optional, for teacher forcing)
        Returns:
          logits_future: [B, F, vocab]
          pred_ids: [B, F]
        """
        if future_tokens is not None and teacher_forcing:
            # Shift future tokens right; first future input is last history token
            shifted_future = torch.cat(
                [history_tokens[:, -1:].detach(), future_tokens[:, :-1]], dim=1
            )
            input_tokens = torch.cat([history_tokens, shifted_future], dim=1)
            logits = self._forward_tokens(input_tokens)
            future_logits = logits[:, -future_tokens.size(1):, :]
            pred_ids = torch.argmax(future_logits, dim=-1)
            return future_logits, pred_ids

        if pred_steps is None:
            raise ValueError("pred_steps must be provided for autoregressive generation")

        # Autoregressive generation (greedy)
        generated = history_tokens
        logits_all = []
        for _ in range(pred_steps):
            logits = self._forward_tokens(generated)
            next_logits = logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            logits_all.append(next_logits.unsqueeze(1))
            generated = torch.cat([generated, next_id], dim=1)

        logits_future = torch.cat(logits_all, dim=1)
        pred_ids = generated[:, -pred_steps:]
        return logits_future, pred_ids
