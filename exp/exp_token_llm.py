import os
import time
from datetime import datetime
from glob import glob

import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.ct_gpt2 import CTGPT2Forecasting
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual


def build_setting(args):
    return (
        f"{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nl{args.n_layers}_nh{args.n_heads}_ctgpt2"
    )


def get_target_index(args):
    if isinstance(args.target_col, int):
        return args.target_col

    csv_path = os.path.join(args.root_path, args.data_path)
    with open(csv_path, "r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")

    columns = header[1:]
    if args.target_col in columns:
        return columns.index(args.target_col)
    return 0


def select_channel(x, target_idx):
    if x.ndim == 2 and x.shape[-1] == 1:
        return x[:, 0]
    if x.ndim == 2:
        return x
    if target_idx is None:
        return x[..., 0]
    return x[..., target_idx]


class TokenLLM_Main(Exp_Basic):
    def __init__(self, args):
        self._ensure_runtime_args(args)
        super(TokenLLM_Main, self).__init__(args)
        self.min_test_loss = np.inf
        self.min_test_mae = np.inf
        self.epoch_for_min_test_loss = 0
        self.run_dt = None
        self.run_dir = None

    def _ensure_runtime_args(self, args):
        if not hasattr(args, "use_gpu"):
            args.use_gpu = False
        if not hasattr(args, "use_multi_gpu"):
            args.use_multi_gpu = False
        if not hasattr(args, "gpu"):
            args.gpu = 0
        if not hasattr(args, "devices"):
            args.devices = str(args.gpu)
        if not hasattr(args, "device_ids"):
            args.device_ids = [args.gpu]
        if not hasattr(args, "use_amp"):
            args.use_amp = False

    def _build_model(self):
        model = CTGPT2Forecasting(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_runtime_hidden_size(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return getattr(getattr(model, "forecaster", None), "hidden_size", self.args.d_model)

    def _build_results_dir(self, setting):
        if self.run_dir is None:
            run_dt = datetime.now()
            date_str = run_dt.strftime("%Y-%m-%d-%H%M%S")
            base_dir = "results"
            results_dir = os.path.join(base_dir, f"{date_str}_{setting}")
            run_index = 2
            while os.path.exists(results_dir):
                results_dir = os.path.join(base_dir, f"{date_str}_{setting}_run{run_index}")
                run_index += 1
            self.run_dir = results_dir
            self.run_dt = run_dt

        os.makedirs(self.run_dir, exist_ok=True)
        return self.run_dir

    def _resolve_checkpoint_path(self, setting, checkpoint_path=None):
        if checkpoint_path:
            return checkpoint_path

        if self.run_dir is not None:
            run_checkpoint = os.path.join(self.run_dir, "checkpoint.pth")
            if os.path.exists(run_checkpoint):
                return run_checkpoint

        result_patterns = [
            os.path.join("results", f"*_{setting}", "checkpoint.pth"),
            os.path.join("results", f"*_{setting}_run*", "checkpoint.pth"),
        ]
        candidate_paths = []
        for pattern in result_patterns:
            candidate_paths.extend(glob(pattern))

        if candidate_paths:
            candidate_paths = sorted(candidate_paths, key=os.path.getmtime, reverse=True)
            return candidate_paths[0]

        legacy_checkpoint = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
        if os.path.exists(legacy_checkpoint):
            return legacy_checkpoint

        return None

    def _append_results_summary(self, run_dt, metrics_dict):
        summary_path = os.path.join("results", "results.txt")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        summary_line = (
            f"[{run_dt.strftime('%Y-%m-%d-%H%M%S')}] "
            f"model={self.args.model} "
            f"\n"
            f"data={self.args.data} "
            f"\n"
            f"seq_len={self.args.seq_len} "
            f"pred_len={self.args.pred_len}"
            f"\n"
            f"d_model={self.args.d_model} "
            f"\n"
            f"runtime_hidden_size={self._get_runtime_hidden_size()} "
            f"\n"
            f"n_layers={self.args.n_layers} "
            f"\n"
            f"n_heads={self.args.n_heads} "
            f"\n"
            f"mse={metrics_dict['mse']:.5f} "
            f"mae={metrics_dict['mae']:.5f} "
            f"\n"
            f"rmse={metrics_dict['rmse']:.5f} "
            f"mape={metrics_dict['mape']:.5f} "
            f"mspe={metrics_dict['mspe']:.5f} "
            f"\n"
            f"loss={metrics_dict['loss']:.5f} "
            f"\n"
        )
        with open(summary_path, "a", encoding="utf-8") as file:
            file.write(summary_line + "\n")

    def _select_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    @staticmethod
    def _inverse_transform_array(data_set, array):
        scaler = getattr(data_set, "scaler", None)
        if scaler is None:
            return array

        original_shape = array.shape
        flattened = array.reshape(-1, original_shape[-1])
        restored = scaler.inverse_transform(flattened)
        return restored.reshape(original_shape)

    def _forward_ct_gpt2_batch(self, batch_x, batch_y, teacher_forcing):
        """
        Modular CT-GPT2 forward for one batch.

        The underlying model performs:
        1. ContinuousEmbedding on standardized scalar inputs.
        2. Optional AlignmentModule refinement with contrastive / trend losses.
        3. GPT-2 backbone forward via `inputs_embeds`.
        4. Output decoder mapping hidden states to `mu` and `log_sigma2`.
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        if self.args.use_amp:
            with torch.amp.autocast(device_type="cuda"):
                forecast, _, _, _, aux = self.model(batch_x, batch_y, teacher_forcing=teacher_forcing)
        else:
            forecast, _, _, _, aux = self.model(batch_x, batch_y, teacher_forcing=teacher_forcing)

        return {
            "target": batch_y,
            "forecast": forecast,
            "mu": aux.get("mu"),
            "log_sigma2": aux.get("log_sigma2"),
            "mixture_logits": aux.get("mixture_logits"),
            "mixture_probs": aux.get("mixture_probs"),
            "distribution_loss": aux.get("distribution_loss"),
            "con_loss": aux.get("con_loss", forecast.new_tensor(0.0)),
            "trend_loss": aux.get("trend_loss", forecast.new_tensor(0.0)),
            "hidden_states": aux.get("hidden_states"),
            "embeddings": aux.get("embeddings"),
            "aligned_embeddings": aux.get("aligned_embeddings"),
            "sample_paths": aux.get("sample_paths"),
            "mean_paths": aux.get("mean_paths"),
        }

    def _compute_ct_gpt2_losses(self, outputs):
        """
        Compute weighted CT-GPT2 training loss.

        Total loss:
            lambda_pred * pred_loss
          + lambda_con * con_loss
          + lambda_trend * trend_loss
        """
        target = outputs["target"]
        pred_loss = outputs.get("distribution_loss")
        if pred_loss is None:
            mu = outputs["mu"]
            log_sigma2 = outputs["log_sigma2"]
            pred_loss = 0.5 * (
                log_sigma2 + (target - mu) ** 2 * torch.exp(-log_sigma2)
            ).mean()

        con_loss = outputs.get("con_loss")
        if con_loss is None:
            con_loss = pred_loss.new_tensor(0.0)

        trend_loss = outputs.get("trend_loss")
        if trend_loss is None:
            trend_loss = pred_loss.new_tensor(0.0)

        total_loss = (
            self.args.lambda_pred * pred_loss
            + self.args.lambda_con * con_loss
            + self.args.lambda_trend * trend_loss
        )
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "con_loss": con_loss,
            "trend_loss": trend_loss,
        }

    def _train_step(self, batch_x, batch_y, optimizer, scaler=None):
        """
        One standard PyTorch training step:
        model.train() -> zero_grad() -> forward -> backward -> optimizer.step().
        """
        optimizer.zero_grad(set_to_none=True)
        outputs = self._forward_ct_gpt2_batch(batch_x, batch_y, teacher_forcing=True)
        loss_dict = self._compute_ct_gpt2_losses(outputs)
        total_loss = loss_dict["loss"]

        if self.args.use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            if self.args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            optimizer.step()

        metrics = {
            "loss": float(total_loss.detach().item()),
            "pred_loss": float(loss_dict["pred_loss"].detach().item()),
            "con_loss": float(loss_dict["con_loss"].detach().item()),
            "trend_loss": float(loss_dict["trend_loss"].detach().item()),
        }
        return outputs, metrics

    def _train_epoch(self, train_loader, optimizer, scaler=None):
        self.model.train()
        running = {"loss": [], "pred_loss": [], "con_loss": [], "trend_loss": []}

        for batch_x, batch_y in train_loader:
            _, step_metrics = self._train_step(batch_x, batch_y, optimizer, scaler=scaler)
            for key in running:
                running[key].append(step_metrics[key])

        return {key: float(np.mean(values)) if values else 0.0 for key, values in running.items()}

    def _run_loader(self, data_set, data_loader, train_mode):
        losses = []
        preds_scaled, trues_scaled = [], []

        self.model.train() if train_mode else self.model.eval()

        with torch.set_grad_enabled(train_mode):
            for batch_x, batch_y in data_loader:
                outputs = self._forward_ct_gpt2_batch(
                    batch_x,
                    batch_y,
                    teacher_forcing=train_mode,
                )
                loss_dict = self._compute_ct_gpt2_losses(outputs)
                forecast = outputs["forecast"]
                target = outputs["target"]
                total_loss = loss_dict["loss"]
                losses.append(total_loss.item())
                preds_scaled.append(forecast.detach().cpu())
                trues_scaled.append(target.detach().cpu())

        preds_scaled = torch.cat(preds_scaled).numpy()
        trues_scaled = torch.cat(trues_scaled).numpy()

        # Metrics are reported on inverse-transformed real values, not on normalized tensors.
        preds = self._inverse_transform_array(data_set, preds_scaled)
        trues = self._inverse_transform_array(data_set, trues_scaled)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        return float(np.mean(losses)), mae, mse, preds, trues, rmse, mape, mspe

    def vali(self, vali_data, vali_loader):
        loss, mae, mse, preds, trues, rmse, mape, mspe = self._run_loader(
            vali_data, vali_loader, train_mode=False
        )
        self.model.train()
        return loss, mae, mse, preds, trues, rmse, mape, mspe

    def train(self, setting, optunaTrialReport=None):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        del train_data, test_data

        checkpoint_dir = self._build_results_dir(setting)
        print(f"Checkpoint directory: {checkpoint_dir}")

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        scaler = torch.amp.GradScaler(device="cuda", init_scale=1024) if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_metrics = self._train_epoch(train_loader, model_optim, scaler=scaler)
            train_loss = train_metrics["loss"]
            vali_loss, vali_mae, _, _, _, _, _, _ = self.vali(vali_data, vali_loader)
            test_loss, test_mae, test_mse, _, _, _, _, _ = self.vali(None, test_loader)

            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss
                self.min_test_mae = test_mae
                self.epoch_for_min_test_loss = epoch

            if optunaTrialReport is not None:
                import optuna

                optunaTrialReport.report(test_loss, epoch)
                if optunaTrialReport.should_prune():
                    raise optuna.exceptions.TrialPruned()

            print(
                "Epoch {0}: Steps-{1} | Train Loss: {2:.5f} Pred: {3:.5f} "
                "Con: {4:.5f} Trend: {5:.5f} Vali.Loss: {6:.5f} "
                "Vali.MAE: {7:.5f} Test.MSE: {8:.5f} Test.MAE: {9:.5f} | {10:.2f}s".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    train_metrics["pred_loss"],
                    train_metrics["con_loss"],
                    train_metrics["trend_loss"],
                    vali_loss,
                    vali_mae,
                    test_mse,
                    test_mae,
                    time.time() - epoch_time,
                )
            )

            early_stopping(vali_loss, self.model, checkpoint_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if np.isnan(train_loss):
                print("Stopping: train loss is NaN")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args, printout=False)

        best_model_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    @staticmethod
    def usage_example():
        """
        Small usage example for one CT-GPT2 training step.

        Example:
            batch_x = torch.randn(8, 96, 1)
            batch_y = torch.randn(8, 24, 1)
            outputs = model.forward_batch(batch_x, batch_y, teacher_forcing=True)
            mu = outputs["mu"]                # [8, 24, 1]
            log_sigma2 = outputs["log_sigma2"]  # [8, 24, 1]
            pred_loss = outputs["distribution_loss"]
            total_loss = (
                lambda_pred * pred_loss
                + lambda_con * outputs["con_loss"]
                + lambda_trend * outputs["trend_loss"]
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        """
        return None

    def _save_visualization(self, results_dir, preds, trues):
        target_idx = get_target_index(self.args) if self.args.use_multivariate else None
        y_true = select_channel(trues[0], target_idx)
        y_pred = select_channel(preds[0], target_idx)
        visual(y_true, y_pred, name=os.path.join(results_dir, "forecast.png"))

    def _save_tokens(self, results_dir):
        print("Skipping token export because CT-GPT2 uses continuous embeddings.")

    def test(self, setting=None, checkpoint_path=None, save_tokens=True, load_checkpoint=True):
        if setting is None:
            setting = build_setting(self.args)

        if load_checkpoint:
            checkpoint_path = self._resolve_checkpoint_path(setting, checkpoint_path=checkpoint_path)
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            self.run_dir = os.path.dirname(checkpoint_path)
            if self.run_dt is None:
                self.run_dt = datetime.now()
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print("Skipping checkpoint load and evaluating current model weights.")
            self._build_results_dir(setting)

        test_data, test_loader = self._get_data(flag="test")
        loss, mae, mse, preds, trues, rmse, mape, mspe = self.vali(test_data, test_loader)

        print(f"Test Loss {loss:.5f} MSE {mse:.5f} MAE {mae:.5f}")

        results_dir = self._build_results_dir(setting)
        run_dt = self.run_dt or datetime.now()
        np.save(os.path.join(results_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(results_dir, "pred.npy"), preds)
        np.save(os.path.join(results_dir, "true.npy"), trues)
        self._save_visualization(results_dir, preds, trues)

        if save_tokens:
            self._save_tokens(results_dir)

        self._append_results_summary(
            run_dt,
            {
                "loss": loss,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "mspe": mspe,
            },
        )
        print(f"Results saved to {results_dir}")

        return mse, mae
