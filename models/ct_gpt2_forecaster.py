import torch
import torch.nn as nn

from models.alignment_module import AlignmentModule
from models.continuous_embedding import ContinuousEmbedding
from models.gpt2_backbone import GPT2BackboneWrapper
from models.output_decoder import OutputDecodingModule


class ContinuousGPT2Forecaster(nn.Module):
    """
    CT-GPT2 forecaster:
    normalized scalars -> ContinuousEmbedding -> AlignmentModule -> GPT-2
    -> OutputDecodingModule.

    `ContinuousEmbedding` first produces `e_cont`. When alignment is enabled,
    `AlignmentModule` applies a learnable projection to obtain `e_aligned`,
    which is the tensor that GPT-2 receives as `inputs_embeds`.
    """

    def __init__(
        self,
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
        decoder_hidden_dim=None,
        num_output_mixtures=1,
        num_sampling_paths=0,
        min_log_variance=-10.0,
        max_log_variance=5.0,
        use_alignment=False,
        use_contrastive_loss=True,
        use_trend_loss=True,
        alignment_hidden_dim=None,
        contrastive_temperature=0.1,
        alignment_dropout=0.1,
        alignment_augmentation_std=0.02,
        use_token_distribution_loss=True,
        token_distribution_samples=256,
        token_distribution_bandwidth=1.0,
        token_moment_weight=0.1,
        decoder_dropout=0.1,
        use_trend_regression=True,
        freeze_gpt2=True,
        gpt2_trainable_layers=1,
    ):
        super().__init__()
        self.num_sampling_paths = num_sampling_paths
        self.use_alignment = use_alignment
        self.use_contrastive_loss = use_contrastive_loss
        self.use_trend_loss = use_trend_loss
        self.use_token_distribution_loss = use_token_distribution_loss

        self.backbone = GPT2BackboneWrapper(
            model_name=model_name,
            local_model_path=local_model_path,
            use_pretrained=use_pretrained,
            prefer_local=prefer_local,
            local_files_only=local_files_only,
            max_seq_len=max_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            disable_internal_position_embeddings=True,
            freeze_gpt2=freeze_gpt2,
            gpt2_trainable_layers=gpt2_trainable_layers,
        )
        self.hidden_size = self.backbone.hidden_size
        self.max_len = self.backbone.max_seq_len
        self.embedding = ContinuousEmbedding(self.hidden_size, self.max_len)
        self.alignment_module = AlignmentModule(
            input_dim=self.hidden_size,
            hidden_size=self.backbone.hidden_size,
            projection_dim=alignment_hidden_dim,
            temperature=contrastive_temperature,
            dropout=alignment_dropout,
            augmentation_std=alignment_augmentation_std,
            token_distribution_samples=token_distribution_samples,
            token_distribution_bandwidth=token_distribution_bandwidth,
            token_moment_weight=token_moment_weight,
        ) if use_alignment else None
        self.output_decoder = OutputDecodingModule(
            hidden_size=self.hidden_size,
            output_dim=1,
            decoder_hidden_dim=decoder_hidden_dim,
            num_mixtures=num_output_mixtures,
            min_log_variance=min_log_variance,
            max_log_variance=max_log_variance,
            dropout=decoder_dropout,
            use_trend_regression=use_trend_regression,
        )

    def get_gpt2_trainability_report(self):
        return self.backbone.get_trainability_report()

    def get_token_embedding_matrix(self):
        return self.backbone.get_token_embedding_matrix()

    def _get_past_length(self, past_key_values):
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            try:
                return int(past_key_values.get_seq_length())
            except TypeError:
                return int(past_key_values.get_seq_length(0))

        if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
            first_key = past_key_values.key_cache[0]
            if first_key is not None:
                return first_key.size(-2)

        if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
            return past_key_values[0][0].size(-2)

        raise TypeError(
            "Unsupported past_key_values type for CT-GPT2 cache handling: "
            f"{type(past_key_values).__name__}"
        )

    def embed_values(self, values, past_key_values=None):
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        return self.embedding(values, position_offset=self._get_past_length(past_key_values))

    def align_embeddings(self, embeddings, values=None, compute_losses=True):
        if self.alignment_module is None:
            zero = embeddings.new_tensor(0.0)
            return embeddings, {"con_loss": zero, "trend_loss": zero, "token_dist_loss": zero}
        return self.alignment_module(
            embeddings,
            values=values,
            token_embedding_matrix=self.get_token_embedding_matrix(),
            compute_losses=compute_losses,
            use_contrastive=self.use_contrastive_loss,
            use_trend=self.use_trend_loss,
            use_token_distribution=self.use_token_distribution_loss and self.backbone.is_pretrained_backbone,
        )

    def _build_attention_mask(self, values, past_key_values=None):
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        return torch.ones(
            values.size(0),
            values.size(1) + self._get_past_length(past_key_values),
            dtype=torch.long,
            device=values.device,
        )

    def backbone_forward(self, embeddings, attention_mask, past_key_values=None, use_cache=False):
        return self.backbone(
            continuous_embeddings=embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def decode_hidden_states(self, hidden_states, base_values=None):
        params = self.output_decoder(hidden_states, base_values=base_values)
        return self.output_decoder.point_forecast(params), params

    def forward_components(self, values, past_key_values=None, use_cache=False, compute_alignment_losses=False):
        e_cont = self.embed_values(values, past_key_values=past_key_values)
        e_aligned, alignment_aux = self.align_embeddings(
            e_cont,
            values=values,
            compute_losses=compute_alignment_losses,
        )
        backbone_outputs = self.backbone_forward(
            embeddings=e_aligned,
            attention_mask=self._build_attention_mask(values, past_key_values=past_key_values),
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        point_forecast, params = self.decode_hidden_states(
            backbone_outputs["last_hidden_state"],
            base_values=values,
        )
        return {
            "embeddings": e_cont,
            "aligned_embeddings": e_aligned,
            "hidden_states": backbone_outputs["last_hidden_state"],
            "past_key_values": backbone_outputs["past_key_values"],
            "point_forecast": point_forecast,
            "params": params,
            "con_loss": alignment_aux["con_loss"],
            "trend_loss": alignment_aux["trend_loss"],
            "token_dist_loss": alignment_aux["token_dist_loss"],
        }

    def generate_sampling_paths(self, prefix_values, horizon, num_paths):
        if prefix_values.dim() == 2:
            prefix_values = prefix_values.unsqueeze(-1)
        if num_paths <= 0:
            raise ValueError("num_paths must be positive.")

        batch_size = prefix_values.size(0)
        repeated_prefix = prefix_values.repeat_interleave(num_paths, dim=0)
        generated = repeated_prefix
        sampled_steps = []
        mean_steps = []
        past_key_values = None

        for _ in range(horizon):
            decoder_input = generated if past_key_values is None else generated[:, -1:, :]
            step_outputs = self.forward_components(
                decoder_input,
                past_key_values=past_key_values,
                use_cache=True,
                compute_alignment_losses=False,
            )
            past_key_values = step_outputs["past_key_values"]
            sampled_next = self.output_decoder.sample(
                {
                    "mu": step_outputs["params"]["mu"][:, -1:, ...],
                    "log_sigma2": step_outputs["params"]["log_sigma2"][:, -1:, ...],
                    "mixture_logits": None if step_outputs["params"]["mixture_logits"] is None else step_outputs["params"]["mixture_logits"][:, -1:, ...],
                    "mixture_probs": None if step_outputs["params"]["mixture_probs"] is None else step_outputs["params"]["mixture_probs"][:, -1:, ...],
                },
                num_samples=1,
            )[:, 0, :, :]
            mean_next = step_outputs["point_forecast"][:, -1:, :]
            sampled_steps.append(sampled_next)
            mean_steps.append(mean_next)
            generated = torch.cat([generated, sampled_next], dim=1)

        sampled_paths = torch.cat(sampled_steps, dim=1).view(batch_size, num_paths, horizon, -1)
        mean_paths = torch.cat(mean_steps, dim=1).view(batch_size, num_paths, horizon, -1)
        return sampled_paths, mean_paths

    def forward(self, history_values, future_values=None, pred_steps=None, teacher_forcing=True):
        if history_values.dim() == 2:
            history_values = history_values.unsqueeze(-1)
        if future_values is not None and future_values.dim() == 2:
            future_values = future_values.unsqueeze(-1)

        if teacher_forcing and future_values is not None:
            history_len = history_values.size(1)
            input_values = torch.cat([history_values, future_values[:, :-1, :]], dim=1)
            outputs = self.forward_components(
                input_values,
                use_cache=False,
                compute_alignment_losses=self.use_alignment,
            )
            prev_values = input_values[:, history_len - 1 :, :]
            params = {
                key: None if value is None else value[:, history_len - 1 :, ...]
                for key, value in outputs["params"].items()
            }
            forecast = outputs["point_forecast"][:, history_len - 1 :, :]
            return forecast, {
                "embeddings": outputs["embeddings"],
                "aligned_embeddings": outputs["aligned_embeddings"],
                "hidden_states": outputs["hidden_states"],
                "mu": params["mu"],
                "log_sigma2": params["log_sigma2"],
                "mixture_logits": params["mixture_logits"],
                "mixture_probs": params["mixture_probs"],
                "delta": params.get("delta"),
                "distribution_loss": self.output_decoder.negative_log_likelihood(future_values, params),
                "point_loss": self.output_decoder.point_loss(future_values, params),
                "delta_loss": self.output_decoder.trend_regression_loss(future_values, prev_values, params),
                "con_loss": outputs["con_loss"],
                "trend_loss": outputs["trend_loss"],
                "token_dist_loss": outputs["token_dist_loss"],
                "sample_paths": None,
                "mean_paths": None,
            }

        if pred_steps is None:
            raise ValueError("pred_steps must be provided for autoregressive prediction.")

        generated = history_values
        forecast_steps = []
        mu_steps = []
        log_sigma2_steps = []
        mixture_logits_steps = []
        mixture_probs_steps = []
        delta_steps = []
        prev_value_steps = []
        past_key_values = None

        for _ in range(pred_steps):
            decoder_input = generated if past_key_values is None else generated[:, -1:, :]
            step_outputs = self.forward_components(
                decoder_input,
                past_key_values=past_key_values,
                use_cache=True,
                compute_alignment_losses=False,
            )
            past_key_values = step_outputs["past_key_values"]
            next_value = step_outputs["point_forecast"][:, -1:, :]
            prev_value_steps.append(decoder_input[:, -1:, :])
            forecast_steps.append(next_value)
            mu_steps.append(step_outputs["params"]["mu"][:, -1:, ...])
            log_sigma2_steps.append(step_outputs["params"]["log_sigma2"][:, -1:, ...])
            if step_outputs["params"].get("delta") is not None:
                delta_steps.append(step_outputs["params"]["delta"][:, -1:, ...])
            if step_outputs["params"]["mixture_logits"] is not None:
                mixture_logits_steps.append(step_outputs["params"]["mixture_logits"][:, -1:, ...])
                mixture_probs_steps.append(step_outputs["params"]["mixture_probs"][:, -1:, ...])
            generated = torch.cat([generated, next_value], dim=1)

        mu = torch.cat(mu_steps, dim=1)
        log_sigma2 = torch.cat(log_sigma2_steps, dim=1)
        mixture_logits = torch.cat(mixture_logits_steps, dim=1) if mixture_logits_steps else None
        mixture_probs = torch.cat(mixture_probs_steps, dim=1) if mixture_probs_steps else None
        delta = torch.cat(delta_steps, dim=1) if delta_steps else None
        prev_values = torch.cat(prev_value_steps, dim=1) if prev_value_steps else None
        forecast = torch.cat(forecast_steps, dim=1)
        distribution_loss = None
        point_loss = None
        delta_loss = None
        if future_values is not None:
            params = {
                "mu": mu,
                "log_sigma2": log_sigma2,
                "mixture_logits": mixture_logits,
                "mixture_probs": mixture_probs,
                "delta": delta,
            }
            distribution_loss = self.output_decoder.negative_log_likelihood(future_values, params)
            point_loss = self.output_decoder.point_loss(future_values, params)
            delta_loss = self.output_decoder.trend_regression_loss(future_values, prev_values, params)

        sample_paths = None
        mean_paths = None
        if self.num_sampling_paths > 0:
            sample_paths, mean_paths = self.generate_sampling_paths(
                prefix_values=history_values,
                horizon=pred_steps,
                num_paths=self.num_sampling_paths,
            )

        return forecast, {
            "embeddings": None,
            "aligned_embeddings": None,
            "hidden_states": None,
            "mu": mu,
            "log_sigma2": log_sigma2,
            "mixture_logits": mixture_logits,
            "mixture_probs": mixture_probs,
            "delta": delta,
            "distribution_loss": distribution_loss,
            "point_loss": point_loss,
            "delta_loss": delta_loss,
            "con_loss": forecast.new_tensor(0.0),
            "trend_loss": forecast.new_tensor(0.0),
            "token_dist_loss": forecast.new_tensor(0.0),
            "sample_paths": sample_paths,
            "mean_paths": mean_paths,
        }
