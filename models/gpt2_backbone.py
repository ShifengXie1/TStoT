import os

import torch
import torch.nn as nn

try:
    from transformers import GPT2Config, GPT2Model
except ImportError:  # pragma: no cover
    GPT2Config = None
    GPT2Model = None


class GPT2BackboneWrapper(nn.Module):
    """
    GPT-2 wrapper for CT-GPT2 continuous embeddings.

    CT-GPT2 provides `inputs_embeds` directly instead of `input_ids`. GPT-2 then
    consumes tensors with shape [batch, seq_len, hidden_size].

    The wrapper also owns the backbone fine-tuning policy so training code can
    switch between:
    1. Fully frozen GPT-2.
    2. Partial fine-tuning of only the last N transformer blocks plus `ln_f`.
    """

    def __init__(
        self,
        model_name="openai-community/gpt2",
        local_model_path=None,
        use_pretrained=True,
        prefer_local=True,
        local_files_only=True,
        max_seq_len=2048,
        d_model=768,
        n_layers=12,
        n_heads=12,
        dropout=0.1,
        disable_internal_position_embeddings=False,
        freeze_gpt2=True,
        gpt2_trainable_layers=1,
    ):
        super().__init__()
        if GPT2Config is None or GPT2Model is None:
            raise ImportError(
                "transformers is required for GPT-2 backbone support. "
                "Install it with `pip install transformers`."
            )

        self.model_name = model_name
        self.local_model_path = local_model_path
        self.use_pretrained = use_pretrained
        self.prefer_local = prefer_local
        self.local_files_only = local_files_only
        self.max_seq_len = max_seq_len
        self.freeze_gpt2 = freeze_gpt2
        self.gpt2_trainable_layers = max(0, int(gpt2_trainable_layers))
        self.is_pretrained_backbone = bool(use_pretrained)

        resolved_model_name, resolved_local_files_only = self._resolve_pretrained_source()
        self.pretrained_source = resolved_model_name if use_pretrained else "random-init"
        if use_pretrained:
            self.gpt2 = GPT2Model.from_pretrained(
                resolved_model_name,
                local_files_only=resolved_local_files_only,
            )
        else:
            self.gpt2 = GPT2Model(
                GPT2Config(
                    n_positions=max_seq_len,
                    n_ctx=max_seq_len,
                    n_embd=d_model,
                    n_layer=n_layers,
                    n_head=n_heads,
                    resid_pdrop=dropout,
                    embd_pdrop=dropout,
                    attn_pdrop=dropout,
                )
            )

        self.hidden_size = getattr(self.gpt2.config, "hidden_size", self.gpt2.config.n_embd)
        self.max_seq_len = self.gpt2.config.n_positions
        if self.max_seq_len < max_seq_len:
            raise ValueError(
                f"GPT-2 context window {self.max_seq_len} is smaller than requested "
                f"max_seq_len={max_seq_len}."
            )

        if disable_internal_position_embeddings:
            self._disable_internal_positions()
        self._apply_trainability_policy()

    def _resolve_pretrained_source(self):
        if not self.use_pretrained:
            return self.model_name, self.local_files_only

        if self.prefer_local and self.local_model_path:
            local_dir = os.path.abspath(self.local_model_path)
            if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
                return local_dir, True
        return self.model_name, self.local_files_only

    def _disable_internal_positions(self):
        if hasattr(self.gpt2, "wpe") and self.gpt2.wpe is not None:
            with torch.no_grad():
                self.gpt2.wpe.weight.zero_()
            self.gpt2.wpe.weight.requires_grad = False

    @staticmethod
    def _set_module_trainable(module, trainable):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    def _apply_trainability_policy(self):
        """
        Configure GPT-2 trainability in one place.

        Policy:
        1. `freeze_gpt2=True`: freeze the full GPT-2 backbone.
        2. `freeze_gpt2=False`: keep only the last
           `gpt2_trainable_layers` transformer blocks plus the final
           layer norm trainable. The selected transformer blocks include their
           internal attention/MLP submodules automatically.
        """
        total_layers = len(self.gpt2.h) if hasattr(self.gpt2, "h") else 0
        for param in self.gpt2.parameters():
            param.requires_grad = False

        if self.freeze_gpt2:
            self._trainability_report = {
                "mode": "frozen",
                "total_layers": total_layers,
                "trainable_layers": [],
                "frozen_layers": list(range(total_layers)),
                "trainable_modules": [],
                "frozen_modules": ["wte", "wpe", "h[*]", "ln_f"],
            }
            return

        trainable_layers = []
        num_layers = min(self.gpt2_trainable_layers, total_layers)
        if num_layers > 0:
            start_idx = total_layers - num_layers
            trainable_layers = list(range(start_idx, total_layers))
            for layer_idx in trainable_layers:
                self._set_module_trainable(self.gpt2.h[layer_idx], True)

        # In partial fine-tuning mode we also unfreeze the final normalization
        # stage so the selected top blocks can adapt their output statistics.
        if hasattr(self.gpt2, "ln_f") and self.gpt2.ln_f is not None:
            self._set_module_trainable(self.gpt2.ln_f, True)

        frozen_layers = [idx for idx in range(total_layers) if idx not in trainable_layers]
        trainable_modules = [f"h.{idx}" for idx in trainable_layers]
        if hasattr(self.gpt2, "ln_f") and self.gpt2.ln_f is not None:
            trainable_modules.append("ln_f")

        self._trainability_report = {
            "mode": "partial",
            "total_layers": total_layers,
            "trainable_layers": trainable_layers,
            "frozen_layers": frozen_layers,
            "trainable_modules": trainable_modules,
            "frozen_modules": ["wte", "wpe"] + [f"h.{idx}" for idx in frozen_layers],
        }

    def get_trainability_report(self):
        report = dict(self._trainability_report)
        report["use_pretrained"] = self.is_pretrained_backbone
        report["pretrained_source"] = self.pretrained_source
        return report

    def get_token_embedding_matrix(self):
        if not hasattr(self.gpt2, "wte") or self.gpt2.wte is None:
            return None
        return self.gpt2.wte.weight.detach()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_gpt2:
            # Keep a fully frozen backbone deterministic during training.
            self.gpt2.eval()
        return self

    def forward(
        self,
        continuous_embeddings,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if continuous_embeddings.dim() != 3:
            raise ValueError(
                "continuous_embeddings must have shape [batch, seq_len, hidden_size], "
                f"but got {tuple(continuous_embeddings.shape)}."
            )

        batch_size, seq_len, hidden_size = continuous_embeddings.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Embedding hidden size {hidden_size} does not match GPT-2 hidden size "
                f"{self.hidden_size}."
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.long,
                device=continuous_embeddings.device,
            )

        outputs = self.gpt2(
            inputs_embeds=continuous_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
        }
