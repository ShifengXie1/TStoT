import torch
import torch.nn as nn

from models.chronos_scaler import ChronosMeanScaler
from models.ct_gpt2_forecaster import ContinuousGPT2Forecaster


class CTGPT2Forecasting(nn.Module):
    """
    Complete CT-GPT2 model with optional Chronos-style sample scaling.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_linear_shortcut = getattr(configs, "use_linear_shortcut", True)
        self.use_chronos_scaling = getattr(configs, "use_chronos_scaling", False)

        if configs.c_in != 1 or configs.c_out != 1:
            raise ValueError(
                "CT-GPT2 currently expects univariate inputs with c_in=c_out=1, "
                f"but received c_in={configs.c_in}, c_out={configs.c_out}."
            )

        self.sample_scaler = ChronosMeanScaler(eps=getattr(configs, "scaling_eps", 1e-8))
        self.forecaster = ContinuousGPT2Forecaster(
            d_model=configs.d_model,
            n_layers=configs.n_layers,
            n_heads=configs.n_heads,
            dropout=getattr(configs, "dropout", 0.1),
            max_len=max(2, self.seq_len + self.pred_len),
            model_name=getattr(configs, "gpt_model_name", "openai-community/gpt2"),
            local_model_path=getattr(configs, "gpt_local_path", "./gpt"),
            use_pretrained=getattr(configs, "use_pretrained_gpt2", True),
            prefer_local=getattr(configs, "prefer_local_gpt2", True),
            local_files_only=getattr(configs, "gpt_local_files_only", True),
            decoder_hidden_dim=getattr(configs, "decoder_hidden_dim", None),
            num_output_mixtures=getattr(configs, "num_output_mixtures", 1),
            num_sampling_paths=getattr(configs, "num_sampling_paths", 0),
            min_log_variance=getattr(configs, "min_log_variance", -10.0),
            max_log_variance=getattr(configs, "max_log_variance", 5.0),
            use_alignment=getattr(configs, "use_alignment", False),
            use_contrastive_loss=getattr(configs, "use_con_loss", True),
            use_trend_loss=getattr(configs, "use_trend_loss", True),
            alignment_hidden_dim=getattr(configs, "alignment_hidden_dim", None),
            contrastive_temperature=getattr(configs, "contrastive_temperature", 0.1),
            alignment_dropout=getattr(configs, "alignment_dropout", getattr(configs, "dropout", 0.1)),
            alignment_augmentation_std=getattr(configs, "alignment_augmentation_std", 0.02),
            decoder_dropout=getattr(configs, "decoder_dropout", getattr(configs, "dropout", 0.1)),
            use_trend_regression=getattr(configs, "use_trend_regression", True),
        )
        if self.use_linear_shortcut:
            self.linear_shortcut = nn.Linear(self.seq_len, self.pred_len)

    def _compute_shortcut(self, model_x, horizon=None):
        """
        Compute the deterministic linear shortcut on the model-space inputs.

        The shortcut head is trained for exactly `pred_len` future steps. To
        keep stochastic path sampling consistent with the main forward pass, we
        only apply it when the requested horizon matches `pred_len`.
        """
        if not self.use_linear_shortcut:
            return None

        horizon = self.pred_len if horizon is None else horizon
        if horizon != self.pred_len:
            raise ValueError(
                "CT-GPT2 linear shortcut is defined for pred_len steps only. "
                f"Received horizon={horizon}, pred_len={self.pred_len}."
            )

        return self.linear_shortcut(model_x.transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def _expand_scale(scale, tensor):
        if tensor is None:
            return None
        while scale.dim() < tensor.dim():
            scale = scale.unsqueeze(-2)
        return scale

    def _apply_inverse_scaling(self, forecast, aux, scale, target=None):
        forecast = self.sample_scaler.unscale(forecast, scale)
        if aux.get("mu") is not None:
            aux["mu"] = self.sample_scaler.unscale(aux["mu"], self._expand_scale(scale, aux["mu"]))
        if aux.get("log_sigma2") is not None:
            aux["log_sigma2"] = self.sample_scaler.unscale_log_variance(
                aux["log_sigma2"],
                self._expand_scale(scale, aux["log_sigma2"]),
            )
        if aux.get("delta") is not None:
            aux["delta"] = self.sample_scaler.unscale(aux["delta"], self._expand_scale(scale, aux["delta"]))
        if aux.get("sample_paths") is not None:
            aux["sample_paths"] = self.sample_scaler.unscale(aux["sample_paths"], scale.unsqueeze(1))
        if aux.get("mean_paths") is not None:
            aux["mean_paths"] = self.sample_scaler.unscale(aux["mean_paths"], scale.unsqueeze(1))
        if target is not None and aux.get("mu") is not None and aux.get("log_sigma2") is not None:
            aux["distribution_loss"] = self.forecaster.output_decoder.negative_log_likelihood(
                target,
                {
                    "mu": aux["mu"],
                    "log_sigma2": aux["log_sigma2"],
                    "mixture_logits": aux.get("mixture_logits"),
                    "mixture_probs": aux.get("mixture_probs"),
                },
            )
            aux["point_loss"] = self.forecaster.output_decoder.point_loss(
                target,
                {
                    "mu": aux["mu"],
                    "mixture_probs": aux.get("mixture_probs"),
                },
            )
        return forecast, aux

    def forward(self, x, y=None, teacher_forcing=True):
        model_x, model_y = x, y
        scale = None
        if self.use_chronos_scaling:
            model_x, model_y, scale = self.sample_scaler.scale(x, y)

        forecast, aux = self.forecaster(
            model_x,
            future_values=model_y,
            pred_steps=self.pred_len,
            teacher_forcing=teacher_forcing,
        )

        shortcut = self._compute_shortcut(model_x) if self.use_linear_shortcut else None
        if shortcut is not None:
            forecast = forecast + shortcut
            if aux.get("mu") is not None:
                if aux.get("mixture_logits") is None:
                    aux["mu"] = aux["mu"] + shortcut
                else:
                    aux["mu"] = aux["mu"] + shortcut.unsqueeze(-2)
            if model_y is not None and aux.get("mu") is not None:
                aux["point_loss"] = self.forecaster.output_decoder.point_loss(
                    model_y,
                    {
                        "mu": aux["mu"],
                        "mixture_probs": aux.get("mixture_probs"),
                    },
                )

        if self.use_chronos_scaling:
            forecast, aux = self._apply_inverse_scaling(forecast, aux, scale, target=y)
        elif y is not None and aux.get("mu") is not None and aux.get("log_sigma2") is not None:
            aux["distribution_loss"] = self.forecaster.output_decoder.negative_log_likelihood(
                y,
                {
                    "mu": aux["mu"],
                    "log_sigma2": aux["log_sigma2"],
                    "mixture_logits": aux.get("mixture_logits"),
                    "mixture_probs": aux.get("mixture_probs"),
                },
            )
            aux["point_loss"] = self.forecaster.output_decoder.point_loss(
                y,
                {
                    "mu": aux["mu"],
                    "mixture_probs": aux.get("mixture_probs"),
                },
            )

        aux["scale"] = scale
        return forecast, None, None, None, aux

    def forward_batch(self, x, y, teacher_forcing=True):
        forecast, _, _, _, aux = self.forward(x, y=y, teacher_forcing=teacher_forcing)
        return {
            "forecast": forecast,
            "mu": aux.get("mu"),
            "log_sigma2": aux.get("log_sigma2"),
            "mixture_logits": aux.get("mixture_logits"),
            "mixture_probs": aux.get("mixture_probs"),
            "delta": aux.get("delta"),
            "distribution_loss": aux.get("distribution_loss"),
            "point_loss": aux.get("point_loss"),
            "delta_loss": aux.get("delta_loss"),
            "con_loss": aux.get("con_loss", forecast.new_tensor(0.0)),
            "trend_loss": aux.get("trend_loss", forecast.new_tensor(0.0)),
            "hidden_states": aux.get("hidden_states"),
            "embeddings": aux.get("embeddings"),
            "aligned_embeddings": aux.get("aligned_embeddings"),
            "sample_paths": aux.get("sample_paths"),
            "mean_paths": aux.get("mean_paths"),
            "scale": aux.get("scale"),
        }

    def sample_paths(self, x, horizon=None, num_paths=None):
        horizon = self.pred_len if horizon is None else horizon
        num_paths = self.forecaster.num_sampling_paths if num_paths is None else num_paths
        model_x = x
        scale = None
        if self.use_chronos_scaling:
            model_x, _, scale = self.sample_scaler.scale(x, None)
        sampled_paths, mean_paths = self.forecaster.generate_sampling_paths(model_x, horizon, num_paths)
        shortcut = self._compute_shortcut(model_x, horizon=horizon) if self.use_linear_shortcut else None
        if shortcut is not None:
            shortcut = shortcut.unsqueeze(1)
            sampled_paths = sampled_paths + shortcut
            mean_paths = mean_paths + shortcut
        if self.use_chronos_scaling:
            sampled_paths = self.sample_scaler.unscale(sampled_paths, scale.unsqueeze(1))
            mean_paths = self.sample_scaler.unscale(mean_paths, scale.unsqueeze(1))
        return sampled_paths, mean_paths
