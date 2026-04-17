import math
import os
import time
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.ct_gpt2 import CTGPT2Forecasting
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

plt.switch_backend("agg")


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
        self._log_gpt2_trainability(model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_runtime_hidden_size(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return getattr(getattr(model, "forecaster", None), "hidden_size", self.args.d_model)

    def _base_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_eval_num_paths(self):
        return max(0, int(getattr(self.args, "eval_num_sampling_paths", 0)))

    @staticmethod
    def _format_layer_list(layer_ids):
        if not layer_ids:
            return "none"
        return ", ".join(str(layer_id) for layer_id in layer_ids)

    def _log_gpt2_trainability(self, model):
        if not hasattr(model, "get_gpt2_trainability_report"):
            return

        report = model.get_gpt2_trainability_report()
        print(
            "GPT-2 trainability | mode={0} | total_layers={1} | trainable_layers={2} | frozen_layers={3}".format(
                report.get("mode", "unknown"),
                report.get("total_layers", 0),
                self._format_layer_list(report.get("trainable_layers", [])),
                self._format_layer_list(report.get("frozen_layers", [])),
            )
        )
        print(
            "GPT-2 modules | trainable={0} | frozen={1}".format(
                ", ".join(report.get("trainable_modules", [])) or "none",
                ", ".join(report.get("frozen_modules", [])) or "none",
            )
        )

    def _trainable_parameters(self):
        # Optimizer/scheduler only ever see parameters that are still marked as
        # trainable after the GPT-2 freeze policy has been applied.
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters were found for optimization.")
        return trainable_params

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
        trainable_params = self._trainable_parameters()
        return torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _select_scheduler(self, optimizer):
        scheduler_type = getattr(self.args, "scheduler_type", "legacy").lower()
        if scheduler_type in {"legacy", "none"}:
            return None
        if scheduler_type != "warmup_cosine":
            raise ValueError(f"Unsupported scheduler_type: {self.args.scheduler_type}")

        total_epochs = max(1, self.args.train_epochs)
        warmup_epochs = min(max(0, self.args.warmup_epochs), max(0, total_epochs - 1))
        min_lr_ratio = float(self.args.min_lr_ratio)

        def lr_lambda(epoch):
            current_epoch = epoch + 1
            if warmup_epochs > 0 and current_epoch <= warmup_epochs:
                return current_epoch / warmup_epochs
            progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
        Forward one batch through CT-GPT2 and return all supervised outputs.
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
        }

    def _compute_ct_gpt2_losses(self, outputs):
        """
        CT-GPT2 objective with embedding alignment supervision.

        Total loss:
            lambda_pred  * pred_loss
          + lambda_con   * con_loss
          + lambda_trend * trend_loss

        `point_loss` and `delta_loss` are still computed for monitoring, but the
        training objective follows the alignment-aware weighting requested by
        the task definition.
        """
        target = outputs["target"]
        forecast = outputs["forecast"]
        pred_loss = outputs.get("distribution_loss")
        if pred_loss is None:
            mu = outputs["mu"]
            log_sigma2 = outputs["log_sigma2"]
            pred_loss = 0.5 * (
                math.log(2.0 * math.pi) + log_sigma2 + (target - mu) ** 2 * torch.exp(-log_sigma2)
            ).mean()

        point_loss = outputs.get("point_loss")
        if point_loss is None:
            point_loss = torch.nn.functional.smooth_l1_loss(forecast, target)

        delta_loss = outputs.get("delta_loss")
        if delta_loss is None:
            delta_loss = pred_loss.new_tensor(0.0)

        con_loss = outputs.get("con_loss")
        if con_loss is None or not (self.args.use_alignment and self.args.use_con_loss):
            con_loss = pred_loss.new_tensor(0.0)

        trend_loss = outputs.get("trend_loss")
        if trend_loss is None or not (self.args.use_alignment and self.args.use_trend_loss):
            trend_loss = pred_loss.new_tensor(0.0)

        total_loss = (
            self.args.lambda_pred * pred_loss
            + self.args.lambda_con * con_loss
            + self.args.lambda_trend * trend_loss
        )
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "point_loss": point_loss,
            "delta_loss": delta_loss,
            "con_loss": con_loss,
            "trend_loss": trend_loss,
        }

    def _train_step(self, batch_x, batch_y, optimizer, scaler=None):
        optimizer.zero_grad(set_to_none=True)
        outputs = self._forward_ct_gpt2_batch(batch_x, batch_y, teacher_forcing=True)
        loss_dict = self._compute_ct_gpt2_losses(outputs)
        total_loss = loss_dict["loss"]

        if self.args.use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            if self.args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.args.max_grad_norm)
            optimizer.step()

        metrics = {
            "loss": float(total_loss.detach().item()),
            "pred_loss": float(loss_dict["pred_loss"].detach().item()),
            "point_loss": float(loss_dict["point_loss"].detach().item()),
            "delta_loss": float(loss_dict["delta_loss"].detach().item()),
            "con_loss": float(loss_dict["con_loss"].detach().item()),
            "trend_loss": float(loss_dict["trend_loss"].detach().item()),
        }
        return outputs, metrics

    def _train_epoch(self, train_loader, optimizer, scaler=None):
        self.model.train()
        running = {
            "loss": [],
            "pred_loss": [],
            "point_loss": [],
            "delta_loss": [],
            "con_loss": [],
            "trend_loss": [],
        }

        for batch_x, batch_y in train_loader:
            _, step_metrics = self._train_step(batch_x, batch_y, optimizer, scaler=scaler)
            for key in running:
                running[key].append(step_metrics[key])

        return {key: float(np.mean(values)) if values else 0.0 for key, values in running.items()}

    def _run_loader(self, data_set, data_loader, train_mode):
        losses = []
        preds_scaled, trues_scaled = [], []
        eval_num_paths = self._get_eval_num_paths() if not train_mode else 0

        self.model.train() if train_mode else self.model.eval()

        with torch.set_grad_enabled(train_mode):
            for batch_x, batch_y in data_loader:
                batch_x_device = batch_x.float().to(self.device)
                outputs = self._forward_ct_gpt2_batch(
                    batch_x,
                    batch_y,
                    teacher_forcing=train_mode,
                )
                loss_dict = self._compute_ct_gpt2_losses(outputs)
                target = outputs["target"]
                losses.append(loss_dict["loss"].item())

                if train_mode or eval_num_paths <= 0:
                    preds_batch = outputs["forecast"]
                else:
                    model = self._base_model()
                    if self.args.use_amp:
                        with torch.amp.autocast(device_type="cuda"):
                            sampled_paths, _ = model.sample_paths(
                                batch_x_device,
                                horizon=self.args.pred_len,
                                num_paths=eval_num_paths,
                            )
                    else:
                        sampled_paths, _ = model.sample_paths(
                            batch_x_device,
                            horizon=self.args.pred_len,
                            num_paths=eval_num_paths,
                        )
                    # Use only the sample mean across trajectories for evaluation.
                    preds_batch = sampled_paths.mean(dim=1)

                preds_scaled.append(preds_batch.detach().cpu())
                trues_scaled.append(target.detach().cpu())

        preds_scaled = torch.cat(preds_scaled).numpy()
        trues_scaled = torch.cat(trues_scaled).numpy()

        preds = self._inverse_transform_array(data_set, preds_scaled)
        trues = self._inverse_transform_array(data_set, trues_scaled)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        return float(np.mean(losses)), mae, mse, preds, trues, rmse, mape, mspe

    def vali(self, vali_data, vali_loader):
        loss, mae, mse, preds, trues, rmse, mape, mspe = self._run_loader(
            vali_data,
            vali_loader,
            train_mode=False,
        )
        self.model.train()
        return loss, mae, mse, preds, trues, rmse, mape, mspe

    def train(self, setting, optunaTrialReport=None):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        del train_data

        checkpoint_dir = self._build_results_dir(setting)
        print(f"Checkpoint directory: {checkpoint_dir}")

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        scaler = torch.amp.GradScaler(device="cuda", init_scale=1024) if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_metrics = self._train_epoch(train_loader, model_optim, scaler=scaler)
            train_loss = train_metrics["loss"]
            vali_loss, vali_mae, _, _, _, _, _, _ = self.vali(vali_data, vali_loader)
            test_loss, test_mae, test_mse, _, _, _, _, _ = self.vali(test_data, test_loader)

            if test_loss < self.min_test_loss:
                self.min_test_loss = test_loss
                self.min_test_mae = test_mae
                self.epoch_for_min_test_loss = epoch

            if optunaTrialReport is not None:
                import optuna

                optunaTrialReport.report(test_loss, epoch)
                if optunaTrialReport.should_prune():
                    raise optuna.exceptions.TrialPruned()

            current_lr = model_optim.param_groups[0]["lr"]
            print(
                "Epoch {0}: Steps-{1} | Train Loss: {2:.5f} Pred: {3:.5f} "
                "Point: {4:.5f} Delta: {5:.5f} Align: {6:.5f} Trend: {7:.5f} "
                "Vali.Loss: {8:.5f} Vali.MAE: {9:.5f} Test.MSE: {10:.5f} "
                "Test.MAE: {11:.5f} LR: {12:.6f} | {13:.2f}s".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    train_metrics["pred_loss"],
                    train_metrics["point_loss"],
                    train_metrics["delta_loss"],
                    train_metrics["con_loss"],
                    train_metrics["trend_loss"],
                    vali_loss,
                    vali_mae,
                    test_mse,
                    test_mae,
                    current_lr,
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

            if scheduler is not None:
                scheduler.step()
            else:
                adjust_learning_rate(model_optim, None, epoch + 1, self.args, printout=False)

        best_model_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    @staticmethod
    def usage_example():
        """
        Example weighted CT-GPT2 losses:

            outputs = model.forward_batch(batch_x, batch_y, teacher_forcing=True)
            total_loss = (
                lambda_pred * outputs["distribution_loss"]
                + lambda_con * outputs["con_loss"]
                + lambda_trend * outputs["trend_loss"]
            )
        """
        return None

    def _save_visualization(self, results_dir, preds, trues):
        target_idx = get_target_index(self.args) if self.args.use_multivariate else None
        y_true = np.asarray(select_channel(trues[0], target_idx))
        y_pred = np.asarray(select_channel(preds[0], target_idx))

        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="GroundTruth", linewidth=2, color="#1f77b4")
        plt.plot(y_pred, label="Sample Mean", linewidth=2, color="#ff7f0e")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "forecast.png"), bbox_inches="tight")
        plt.close()

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
        np.save(os.path.join(results_dir, "sample_mean.npy"), preds)
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
