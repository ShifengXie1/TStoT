import torch
import torch.nn as nn

from models.alignment_module import AlignmentModule
from models.continuous_embedding import ContinuousEmbedding
from models.gpt2_backbone import GPT2BackboneWrapper
from models.output_decoder import OutputDecodingModule


class ContinuousGPT2Forecaster(nn.Module):
    """
    CT-GPT2 forecaster:
    normalized scalars -> ContinuousEmbedding -> Alignment(optional) -> GPT-2
    -> OutputDecodingModule.
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
        alignment_hidden_dim=None,
        contrastive_temperature=0.1,
    ):
        super().__init__()
        self.num_sampling_paths = num_sampling_paths
        self.use_alignment = use_alignment

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
        )
        self.hidden_size = self.backbone.hidden_size
        self.max_len = self.backbone.max_seq_len
        self.embedding = ContinuousEmbedding(self.hidden_size, self.max_len)
        self.alignment_module = AlignmentModule(
            d_model=self.hidden_size,
            hidden_dim=alignment_hidden_dim,
            temperature=contrastive_temperature,
        ) if use_alignment else None
        self.output_decoder = OutputDecodingModule(
            hidden_size=self.hidden_size,
            output_dim=1,
            decoder_hidden_dim=decoder_hidden_dim,
            num_mixtures=num_output_mixtures,
            min_log_variance=min_log_variance,
            max_log_variance=max_log_variance,
        )

    def _get_past_length(self, past_key_values):
        if past_key_values is None:
            return 0
        return past_key_values[0][0].size(-2)

    def embed_values(self, values, past_key_values=None):
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        return self.embedding(values, position_offset=self._get_past_length(past_key_values))

    def align_embeddings(self, embeddings, values=None, compute_losses=True):
        if self.alignment_module is None:
            zero = embeddings.new_tensor(0.0)
            return embeddings, {"con_loss": zero, "trend_loss": zero}
        return self.alignment_module(embeddings, values=values, compute_losses=compute_losses)

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

    def decode_hidden_states(self, hidden_states):
        params = self.output_decoder(hidden_states)
        return self.output_decoder.point_forecast(params), params

    def forward_components(self, values, past_key_values=None, use_cache=False, compute_alignment_losses=False):
        embeddings = self.embed_values(values, past_key_values=past_key_values)
        aligned_embeddings, alignment_aux = self.align_embeddings(
            embeddings,
            values=values,
            compute_losses=compute_alignment_losses,
        )
        backbone_outputs = self.backbone_forward(
            embeddings=aligned_embeddings,
            attention_mask=self._build_attention_mask(values, past_key_values=past_key_values),
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        point_forecast, params = self.decode_hidden_states(backbone_outputs["last_hidden_state"])
        return {
            "embeddings": embeddings,
            "aligned_embeddings": aligned_embeddings,
            "hidden_states": backbone_outputs["last_hidden_state"],
            "past_key_values": backbone_outputs["past_key_values"],
            "point_forecast": point_forecast,
            "params": params,
            "con_loss": alignment_aux["con_loss"],
            "trend_loss": alignment_aux["trend_loss"],
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
                "distribution_loss": self.output_decoder.negative_log_likelihood(future_values, params),
                "con_loss": outputs["con_loss"],
                "trend_loss": outputs["trend_loss"],
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
            forecast_steps.append(next_value)
            mu_steps.append(step_outputs["params"]["mu"][:, -1:, ...])
            log_sigma2_steps.append(step_outputs["params"]["log_sigma2"][:, -1:, ...])
            if step_outputs["params"]["mixture_logits"] is not None:
                mixture_logits_steps.append(step_outputs["params"]["mixture_logits"][:, -1:, ...])
                mixture_probs_steps.append(step_outputs["params"]["mixture_probs"][:, -1:, ...])
            generated = torch.cat([generated, next_value], dim=1)

        mu = torch.cat(mu_steps, dim=1)
        log_sigma2 = torch.cat(log_sigma2_steps, dim=1)
        mixture_logits = torch.cat(mixture_logits_steps, dim=1) if mixture_logits_steps else None
        mixture_probs = torch.cat(mixture_probs_steps, dim=1) if mixture_probs_steps else None
        forecast = torch.cat(forecast_steps, dim=1)
        distribution_loss = None
        if future_values is not None:
            distribution_loss = self.output_decoder.negative_log_likelihood(
                future_values,
                {
                    "mu": mu,
                    "log_sigma2": log_sigma2,
                    "mixture_logits": mixture_logits,
                    "mixture_probs": mixture_probs,
                },
            )

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
            "distribution_loss": distribution_loss,
            "con_loss": forecast.new_tensor(0.0),
            "trend_loss": forecast.new_tensor(0.0),
            "sample_paths": sample_paths,
            "mean_paths": mean_paths,
        }
