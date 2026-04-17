import torch
import torch.nn as nn
import torch.nn.functional as F

from models.compensation_alignment import CompensationAlignmentModule
from models.gpt2_backbone import GPT2BackboneWrapper
from models.patch_embedding import TrendAwarePatchDecoder, TrendAwarePatchEmbedding


class ContinuousGPT2Forecaster(nn.Module):
    """
    Patch-based CT-GPT2 forecaster with compensation alignment.

    Time-series patches are encoded into a trend-aware latent space `z`, then
    aligned to the pretrained token manifold through an affine compensation:

        u = s(z) * z + b(z)

    GPT-2 operates on `u`. The predicted GPT states are decompensated back to
    `z` before patch decoding and overlap-add reconstruction.
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
        patch_size=8,
        patch_stride=8,
    ):
        super().__init__()
        del num_output_mixtures, min_log_variance, max_log_variance, alignment_augmentation_std, token_distribution_bandwidth
        self.num_sampling_paths = num_sampling_paths
        self.use_alignment = use_alignment
        self.use_contrastive_loss = use_contrastive_loss
        self.use_trend_loss = use_trend_loss
        self.use_token_distribution_loss = use_token_distribution_loss
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

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
        self.embedding = TrendAwarePatchEmbedding(
            patch_size=self.patch_size,
            stride=self.patch_stride,
            d_model=self.hidden_size,
            max_patches=self.max_len,
            dropout=dropout,
        )
        self.alignment_module = CompensationAlignmentModule(
            hidden_size=self.backbone.hidden_size,
            projection_dim=alignment_hidden_dim,
            temperature=contrastive_temperature,
            dropout=alignment_dropout,
            token_distribution_samples=token_distribution_samples,
            token_moment_weight=token_moment_weight,
        ) if use_alignment else None
        self.latent_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, decoder_hidden_dim or self.hidden_size),
            nn.GELU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(decoder_hidden_dim or self.hidden_size, self.hidden_size),
        )
        self.patch_decoder = TrendAwarePatchDecoder(
            d_model=self.hidden_size,
            patch_size=self.patch_size,
            hidden_dim=decoder_hidden_dim or self.hidden_size,
            dropout=decoder_dropout,
        )
        self.log_variance = nn.Parameter(torch.tensor(0.0))

    def get_gpt2_trainability_report(self):
        return self.backbone.get_trainability_report()

    @staticmethod
    def num_patches_for_length(length, patch_size, stride):
        return TrendAwarePatchEmbedding.num_patches_for_length(length, patch_size, stride)

    def get_token_embedding_matrix(self):
        return self.backbone.get_token_embedding_matrix()

    def _zero_alignment_losses(self, reference):
        zero = reference.new_tensor(0.0)
        return {"con_loss": zero, "token_dist_loss": zero, "comp_reg_loss": zero}

    def align_embeddings(self, embeddings, compute_losses=True):
        if self.alignment_module is None:
            return embeddings, self._zero_alignment_losses(embeddings)
        return self.alignment_module(
            embeddings,
            token_embedding_matrix=self.get_token_embedding_matrix(),
            compute_losses=compute_losses,
            use_contrastive=self.use_contrastive_loss,
            use_token_distribution=self.use_token_distribution_loss and self.backbone.is_pretrained_backbone,
        )

    def decompensate_embeddings(self, aligned_embeddings):
        if self.alignment_module is None:
            zero = aligned_embeddings.new_tensor(0.0)
            return aligned_embeddings, {"inv_scale": zero, "inv_bias": zero}
        return self.alignment_module.decompensate(aligned_embeddings)

    @staticmethod
    def _build_attention_mask(embeddings):
        return torch.ones(
            embeddings.size(0),
            embeddings.size(1),
            dtype=torch.long,
            device=embeddings.device,
        )

    def backbone_forward(self, aligned_embeddings):
        return self.backbone(
            continuous_embeddings=aligned_embeddings,
            attention_mask=self._build_attention_mask(aligned_embeddings),
            past_key_values=None,
            use_cache=False,
        )

    def _compute_prediction_losses(self, forecast, target):
        mse_loss = F.mse_loss(forecast, target)
        point_loss = F.smooth_l1_loss(forecast, target)

        if target.size(1) >= 2:
            pred_diff = forecast[:, 1:, :] - forecast[:, :-1, :]
            true_diff = target[:, 1:, :] - target[:, :-1, :]
            delta_loss = F.smooth_l1_loss(pred_diff, true_diff)
        else:
            delta_loss = forecast.new_tensor(0.0)

        if target.size(1) >= 3:
            pred_diff2 = pred_diff[:, 1:, :] - pred_diff[:, :-1, :]
            true_diff2 = true_diff[:, 1:, :] - true_diff[:, :-1, :]
            curve_loss = F.smooth_l1_loss(pred_diff2, true_diff2)
        else:
            curve_loss = forecast.new_tensor(0.0)

        return mse_loss, point_loss, delta_loss, curve_loss

    def _decode_latent_patches(self, hidden_states):
        predicted_aligned = self.latent_head(hidden_states)
        predicted_latent, _ = self.decompensate_embeddings(predicted_aligned)
        patch_values = self.patch_decoder(predicted_latent)
        return predicted_aligned, predicted_latent, patch_values

    def _prepare_patch_sequence(self, history_values, future_values=None):
        history_latent, history_aux = self.embedding.encode(history_values, position_offset=0)
        future_latent = None
        future_aux = None
        if future_values is not None:
            future_latent, future_aux = self.embedding.encode(
                future_values,
                position_offset=history_latent.size(1),
            )
        return history_latent, history_aux, future_latent, future_aux

    def _forecast_from_patch_values(self, patch_values, target_length):
        return self.embedding.overlap_add(patch_values, target_length=target_length)

    def generate_sampling_paths(self, prefix_values, horizon, num_paths):
        point_forecast, aux = self.forward(
            history_values=prefix_values,
            future_values=None,
            pred_steps=horizon,
            teacher_forcing=False,
        )
        paths = point_forecast.unsqueeze(1).repeat(1, num_paths, 1, 1)
        return paths, paths

    def forward(self, history_values, future_values=None, pred_steps=None, teacher_forcing=True):
        if history_values.dim() == 2:
            history_values = history_values.unsqueeze(-1)
        if future_values is not None and future_values.dim() == 2:
            future_values = future_values.unsqueeze(-1)

        history_latent, history_aux, future_latent, future_aux = self._prepare_patch_sequence(
            history_values,
            future_values=future_values,
        )
        history_patch_count = history_latent.size(1)

        if teacher_forcing and future_values is not None:
            input_latent = torch.cat([history_latent, future_latent[:, :-1, :]], dim=1)
            aligned_input, alignment_aux = self.align_embeddings(
                input_latent,
                compute_losses=self.use_alignment,
            )
            backbone_outputs = self.backbone_forward(aligned_input)
            future_hidden = backbone_outputs["last_hidden_state"][:, history_patch_count - 1 :, :]
            predicted_aligned, predicted_latent, future_patch_values = self._decode_latent_patches(future_hidden)
            forecast = self._forecast_from_patch_values(
                future_patch_values,
                target_length=future_values.size(1),
            )

            distribution_loss, point_loss, delta_loss, curve_loss = self._compute_prediction_losses(
                forecast,
                future_values,
            )
            latent_recon_loss = F.smooth_l1_loss(predicted_latent, future_latent)
            point_loss = point_loss + 0.2 * latent_recon_loss
            comp_reg_loss = alignment_aux.get("comp_reg_loss", forecast.new_tensor(0.0))

            return forecast, {
                "embeddings": input_latent,
                "aligned_embeddings": aligned_input,
                "hidden_states": backbone_outputs["last_hidden_state"],
                "mu": forecast,
                "log_sigma2": torch.ones_like(forecast) * self.log_variance.clamp(-6.0, 3.0),
                "mixture_logits": None,
                "mixture_probs": None,
                "delta": None,
                "distribution_loss": distribution_loss,
                "point_loss": point_loss,
                "delta_loss": delta_loss,
                "con_loss": alignment_aux.get("con_loss", forecast.new_tensor(0.0))
                + 0.01 * comp_reg_loss,
                "trend_loss": curve_loss,
                "token_dist_loss": alignment_aux.get("token_dist_loss", forecast.new_tensor(0.0)),
                "sample_paths": None,
                "mean_paths": None,
            }

        if pred_steps is None:
            raise ValueError("pred_steps must be provided for autoregressive prediction.")

        target_patch_count = self.embedding.num_patches_for_length(
            pred_steps,
            self.patch_size,
            self.patch_stride,
        )
        generated_latent = history_latent
        predicted_patch_values = []
        predicted_aligned_states = []
        predicted_latent_states = []

        for _ in range(target_patch_count):
            aligned_input, _ = self.align_embeddings(generated_latent, compute_losses=False)
            backbone_outputs = self.backbone_forward(aligned_input)
            next_hidden = backbone_outputs["last_hidden_state"][:, -1:, :]
            next_aligned, next_latent, next_patch = self._decode_latent_patches(next_hidden)
            next_input_latent = self.embedding.encode_patch_values(
                next_patch,
                position_offset=generated_latent.size(1),
            )
            generated_latent = torch.cat([generated_latent, next_input_latent], dim=1)
            predicted_aligned_states.append(next_aligned)
            predicted_latent_states.append(next_latent)
            predicted_patch_values.append(next_patch)

        predicted_patch_values = torch.cat(predicted_patch_values, dim=1)
        forecast = self._forecast_from_patch_values(predicted_patch_values, target_length=pred_steps)

        mu = forecast
        log_sigma2 = torch.ones_like(forecast) * self.log_variance.clamp(-6.0, 3.0)
        distribution_loss = None
        point_loss = None
        delta_loss = None
        trend_loss = None
        if future_values is not None:
            distribution_loss, point_loss, delta_loss, trend_loss = self._compute_prediction_losses(
                forecast,
                future_values,
            )

        return forecast, {
            "embeddings": history_latent,
            "aligned_embeddings": None if not predicted_aligned_states else torch.cat(predicted_aligned_states, dim=1),
            "hidden_states": None,
            "mu": mu,
            "log_sigma2": log_sigma2,
            "mixture_logits": None,
            "mixture_probs": None,
            "delta": None,
            "distribution_loss": distribution_loss,
            "point_loss": point_loss,
            "delta_loss": delta_loss,
            "con_loss": forecast.new_tensor(0.0),
            "trend_loss": trend_loss if trend_loss is not None else forecast.new_tensor(0.0),
            "token_dist_loss": forecast.new_tensor(0.0),
            "sample_paths": None,
            "mean_paths": None,
        }
