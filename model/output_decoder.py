import math

import torch
import torch.nn as nn


class OutputDecodingModule(nn.Module):
    """
    Map GPT-2 hidden states to predictive distributions.

    Supports:
    - Gaussian output: `mu`, `log_sigma2`
    - Optional GMM output for richer stochastic sampling
    """

    def __init__(
        self,
        hidden_size,
        output_dim=1,
        decoder_hidden_dim=None,
        num_mixtures=1,
        min_log_variance=-10.0,
        max_log_variance=5.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance

        decoder_hidden_dim = decoder_hidden_dim or hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, decoder_hidden_dim),
            nn.GELU(),
        )
        if num_mixtures == 1:
            self.param_head = nn.Linear(decoder_hidden_dim, output_dim * 2)
        else:
            self.mixture_logits_head = nn.Linear(decoder_hidden_dim, num_mixtures)
            self.mu_head = nn.Linear(decoder_hidden_dim, num_mixtures * output_dim)
            self.log_var_head = nn.Linear(decoder_hidden_dim, num_mixtures * output_dim)

    def forward(self, hidden_states):
        features = self.proj(hidden_states)
        if self.num_mixtures == 1:
            mu, log_sigma2 = torch.chunk(self.param_head(features), chunks=2, dim=-1)
            log_sigma2 = log_sigma2.clamp(self.min_log_variance, self.max_log_variance)
            return {
                "mu": mu,
                "log_sigma2": log_sigma2,
                "mixture_logits": None,
                "mixture_probs": None,
            }

        mixture_logits = self.mixture_logits_head(features)
        mixture_probs = torch.softmax(mixture_logits, dim=-1)
        batch_size, seq_len = hidden_states.shape[:2]
        mu = self.mu_head(features).view(batch_size, seq_len, self.num_mixtures, self.output_dim)
        log_sigma2 = self.log_var_head(features).view(
            batch_size,
            seq_len,
            self.num_mixtures,
            self.output_dim,
        )
        log_sigma2 = log_sigma2.clamp(self.min_log_variance, self.max_log_variance)
        return {
            "mu": mu,
            "log_sigma2": log_sigma2,
            "mixture_logits": mixture_logits,
            "mixture_probs": mixture_probs,
        }

    def point_forecast(self, params):
        if self.num_mixtures == 1:
            return params["mu"]
        return (params["mixture_probs"].unsqueeze(-1) * params["mu"]).sum(dim=-2)

    def gaussian_nll(self, target, mu, log_sigma2):
        return (((target - mu) ** 2) / (2.0 * torch.exp(log_sigma2)) + 0.5 * log_sigma2).mean()

    def negative_log_likelihood(self, target, params):
        if self.num_mixtures == 1:
            return self.gaussian_nll(target, params["mu"], params["log_sigma2"])

        target = target.unsqueeze(-2)
        component_log_prob = -0.5 * (
            math.log(2.0 * math.pi)
            + params["log_sigma2"]
            + (target - params["mu"]) ** 2 * torch.exp(-params["log_sigma2"])
        ).sum(dim=-1)
        mixture_log_prob = torch.log_softmax(params["mixture_logits"], dim=-1) + component_log_prob
        return (-torch.logsumexp(mixture_log_prob, dim=-1)).mean()

    def sample(self, params, num_samples=1):
        """
        Draw samples with shape [B, num_samples, L, output_dim].
        """
        if self.num_mixtures == 1:
            mu = params["mu"].unsqueeze(1)
            std = torch.exp(0.5 * params["log_sigma2"]).unsqueeze(1)
            eps = torch.randn(
                mu.size(0),
                num_samples,
                mu.size(2),
                mu.size(3),
                device=mu.device,
                dtype=mu.dtype,
            )
            return mu + eps * std

        mixture_probs = params["mixture_probs"]
        mu = params["mu"]
        std = torch.exp(0.5 * params["log_sigma2"])
        batch_size, seq_len, num_mixtures, output_dim = mu.shape
        component_ids = torch.distributions.Categorical(probs=mixture_probs).sample((num_samples,))
        component_ids = component_ids.permute(1, 0, 2)
        gather_index = component_ids.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            num_samples,
            seq_len,
            1,
            output_dim,
        )
        mu = mu.unsqueeze(1).expand(batch_size, num_samples, seq_len, num_mixtures, output_dim)
        std = std.unsqueeze(1).expand(batch_size, num_samples, seq_len, num_mixtures, output_dim)
        chosen_mu = torch.gather(mu, dim=3, index=gather_index).squeeze(3)
        chosen_std = torch.gather(std, dim=3, index=gather_index).squeeze(3)
        return chosen_mu + torch.randn_like(chosen_mu) * chosen_std

    @staticmethod
    def rescale_to_original_scale(values, scaler_mean=None, scaler_scale=None):
        if scaler_mean is None or scaler_scale is None:
            return values
        mean = torch.as_tensor(scaler_mean, device=values.device, dtype=values.dtype)
        scale = torch.as_tensor(scaler_scale, device=values.device, dtype=values.dtype)
        return values * scale + mean
