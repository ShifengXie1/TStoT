import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError:  # pragma: no cover - handled at runtime with a clearer error.
    GPT2Config = None
    GPT2LMHeadModel = None


class TokenForecaster(nn.Module):
    """
    GPT-2 based autoregressive token predictor for time-series token ids.
    """
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        dropout,
        max_len=2048,
        model_name="openai-community/gpt2",
        local_model_path=None,
        use_pretrained=True,
        prefer_local=True,
        local_files_only=True,
    ):
        super().__init__()
        if GPT2LMHeadModel is None or GPT2Config is None:
            raise ImportError(
                "transformers is required for GPT-2 token forecasting. "
                "Install it with `pip install transformers`."
            )

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.use_pretrained = use_pretrained
        self.prefer_local = prefer_local
        self.local_files_only = local_files_only

        resolved_model_name, resolved_local_files_only = self._resolve_pretrained_source()
        self.load_source = resolved_model_name
        self.loaded_from_local = bool(
            isinstance(resolved_model_name, str) and os.path.isdir(resolved_model_name)
        )

        if use_pretrained:
            if self.loaded_from_local:
                print(f"Loading GPT-2 weights from local directory: {self.load_source}")
            else:
                print(
                    f"Loading GPT-2 weights by model name: {self.load_source} "
                    f"(local_files_only={resolved_local_files_only})"
                )
            try:
                self.decoder = GPT2LMHeadModel.from_pretrained(
                    resolved_model_name,
                    local_files_only=resolved_local_files_only,
                )
            except OSError as exc:
                raise OSError(
                    "Failed to load GPT-2 weights. If you want offline loading, make sure "
                    f"the local checkpoint directory exists and is complete: {self.local_model_path}. "
                    "Otherwise disable local-only loading or use random initialization with "
                    "`--use_pretrained_gpt2 false`."
                ) from exc
            self.decoder.resize_token_embeddings(vocab_size)

            # Keep position embeddings consistent with the runtime context length.
            if self.decoder.config.n_positions < max_len:
                raise ValueError(
                    f"GPT-2 context window {self.decoder.config.n_positions} is smaller than "
                    f"requested max_len={max_len}. Reduce the sequence length or switch to a "
                    "larger GPT-2 checkpoint."
                )
        else:
            print("Initializing GPT-2 token predictor from scratch.")
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=max_len,
                n_ctx=max_len,
                n_embd=d_model,
                n_layer=n_layers,
                n_head=n_heads,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
            )
            self.decoder = GPT2LMHeadModel(config)

        self.max_len = self.decoder.config.n_positions

    def _resolve_pretrained_source(self):
        if not self.use_pretrained:
            return self.model_name, self.local_files_only

        if self.prefer_local and self.local_model_path:
            local_dir = os.path.abspath(self.local_model_path)
            required_files = ("config.json", "model.safetensors")
            if os.path.isdir(local_dir) and all(
                os.path.exists(os.path.join(local_dir, file_name)) for file_name in required_files
            ):
                return local_dir, True

        return self.model_name, self.local_files_only

    def _forward_tokens(self, token_ids):
        seq_len = token_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"token length {seq_len} > max_len {self.max_len}")

        attention_mask = torch.ones_like(token_ids, dtype=torch.long)
        outputs = self.decoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits

    def forward(self, history_tokens, future_tokens=None, pred_steps=None, teacher_forcing=True):
        """
        history_tokens: [B, H]
        future_tokens: [B, F] (optional, for teacher forcing)
        Returns:
          logits_future: [B, F, vocab]
          pred_ids: [B, F]
          token_loss: scalar or None
        """
        token_loss = None

        if future_tokens is not None and teacher_forcing:
            history_len = history_tokens.size(1)
            shifted_future = future_tokens[:, :-1]
            input_tokens = torch.cat([history_tokens, shifted_future], dim=1)
            logits = self._forward_tokens(input_tokens)
            future_logits = logits[:, history_len - 1 :, :]
            pred_ids = torch.argmax(future_logits, dim=-1)
            token_loss = F.cross_entropy(
                future_logits.reshape(-1, future_logits.size(-1)),
                future_tokens.reshape(-1),
            )
            return future_logits, pred_ids, token_loss

        if pred_steps is None:
            raise ValueError("pred_steps must be provided for autoregressive generation")

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
        return logits_future, pred_ids, token_loss
