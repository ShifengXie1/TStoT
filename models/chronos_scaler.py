import torch


class ChronosMeanScaler:
    """
    Optional Chronos-style per-sample mean scaler.

    This scaler uses only the historical context window to compute the scale:

        s = mean(abs(history))

    The same scale is then applied to both the history and the forecast target.
    If the scale is too small, it is replaced by 1.0 for numerical stability.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def scale(self, history, target=None):
        scale = history.abs().mean(dim=1, keepdim=True)
        scale = torch.where(scale < self.eps, torch.ones_like(scale), scale)
        history_scaled = history / scale
        target_scaled = None if target is None else target / scale
        return history_scaled, target_scaled, scale

    @staticmethod
    def unscale(values, scale):
        if values is None:
            return None
        return values * scale

    @staticmethod
    def unscale_log_variance(log_sigma2, scale):
        if log_sigma2 is None:
            return None
        return log_sigma2 + 2.0 * torch.log(scale)
