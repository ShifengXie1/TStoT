import os
import time
from datetime import datetime
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.token_llm_forecasting import TokenLLMForecasting
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual


def build_setting(args):
    return (
        f"{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_"
        f"ps{args.patch_size}_st{args.stride}_dm{args.d_model}_v{args.vocab_size}_"
        f"predgpt2"
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
        model = TokenLLMForecasting(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

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
            f"patch_size={self.args.patch_size} "
            f"\n"
            f"stride={self.args.stride} "
            f"\n"
            f"d_model={self.args.d_model} "
            f"\n"
            f"vocab_size={self.args.vocab_size} "
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

    def _select_criterion(self):
        return torch.nn.MSELoss()

    def _process_one_batch(self, batch_x, batch_y, teacher_forcing):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if self.args.use_amp:
            with torch.amp.autocast(device_type="cuda"):
                forecast, token_logits, _, recon, aux = self.model(
                    batch_x, batch_y, teacher_forcing=teacher_forcing
                )
        else:
            forecast, token_logits, _, recon, aux = self.model(
                batch_x, batch_y, teacher_forcing=teacher_forcing
            )

        return batch_x, forecast, batch_y, token_logits, recon, aux

    def _compute_total_loss(self, criterion, history, forecast, token_logits, recon, aux, target):
        forecast_loss = criterion(forecast, target)

        recon_terms = []
        recon_past = aux.get("recon_past")
        if recon_past is not None:
            recon_terms.append(criterion(recon_past, history))
        if recon is not None:
            recon_terms.append(criterion(recon, target))
        recon_loss = torch.stack(recon_terms).mean() if recon_terms else torch.tensor(
            0.0, device=target.device
        )

        token_ce = aux.get("token_loss")
        future_token_ids = aux.get("future_token_ids")
        if token_ce is None:
            token_ce = torch.tensor(0.0, device=target.device)

        if future_token_ids is not None and token_logits is not None and aux.get("token_loss") is None:
            token_ce = F.cross_entropy(
                token_logits.reshape(-1, token_logits.size(-1)),
                future_token_ids.reshape(-1),
            )

        vq_loss = aux["vq_loss"]
        total_loss = (
            forecast_loss
            + self.args.alpha * recon_loss
            + self.args.beta * token_ce
            + self.args.gamma * vq_loss
        )
        return total_loss

    def _run_loader(self, data_loader, criterion, train_mode):
        losses = []
        preds, trues = [], []

        self.model.train() if train_mode else self.model.eval()

        with torch.set_grad_enabled(train_mode):
            for batch_x, batch_y in data_loader:
                history, forecast, target, token_logits, recon, aux = self._process_one_batch(
                    batch_x, batch_y, teacher_forcing=train_mode
                )
                total_loss = self._compute_total_loss(
                    criterion, history, forecast, token_logits, recon, aux, target
                )

                losses.append(total_loss.item())
                preds.append(forecast.detach().cpu())
                trues.append(target.detach().cpu())

        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        return float(np.mean(losses)), mae, mse, preds, trues, rmse, mape, mspe

    def vali(self, vali_data, vali_loader, criterion):
        del vali_data
        loss, mae, mse, preds, trues, rmse, mape, mspe = self._run_loader(
            vali_loader, criterion, train_mode=False
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
        criterion = self._select_criterion()
        scaler = torch.amp.GradScaler(device="cuda", init_scale=1024) if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_losses = []
            self.model.train()

            for batch_x, batch_y in train_loader:
                model_optim.zero_grad(set_to_none=True)
                history, forecast, target, token_logits, recon, aux = self._process_one_batch(
                    batch_x, batch_y, teacher_forcing=True
                )
                total_loss = self._compute_total_loss(
                    criterion, history, forecast, token_logits, recon, aux, target
                )

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    if self.args.max_grad_norm > 0:
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    model_optim.step()

                train_losses.append(total_loss.item())

            train_loss = float(np.mean(train_losses))
            vali_loss, vali_mae, _, _, _, _, _, _ = self.vali(
                vali_data, vali_loader, criterion
            )
            test_loss, test_mae, test_mse, _, _, _, _, _ = self.vali(
                None, test_loader, criterion
            )

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
                "Epoch {0}: Steps-{1} | Train Loss: {2:.5f} Vali.Loss: {3:.5f} "
                "Vali.MAE: {4:.5f} Test.MSE: {5:.5f} Test.MAE: {6:.5f} | {7:.2f}s".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
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

    def _save_visualization(self, results_dir, preds, trues):
        target_idx = get_target_index(self.args) if self.args.use_multivariate else None
        y_true = select_channel(trues[0], target_idx)
        y_pred = select_channel(preds[0], target_idx)
        visual(y_true, y_pred, name=os.path.join(results_dir, "forecast.png"))

    def _save_tokens(self, test_loader, results_dir):
        batch_x, batch_y = next(iter(test_loader))
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, _, pred_token_ids, _, aux = self.model(
                batch_x, batch_y, teacher_forcing=False
            )

        np.save(
            os.path.join(results_dir, "tokens_past.npy"),
            aux["past_token_ids"].detach().cpu().numpy(),
        )
        np.save(
            os.path.join(results_dir, "tokens_future_pred.npy"),
            pred_token_ids.detach().cpu().numpy(),
        )
        if aux["future_token_ids"] is not None:
            np.save(
                os.path.join(results_dir, "tokens_future_true.npy"),
                aux["future_token_ids"].detach().cpu().numpy(),
            )

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
        criterion = self._select_criterion()
        loss, mae, mse, preds, trues, rmse, mape, mspe = self.vali(
            test_data, test_loader, criterion
        )

        print(f"Test Loss {loss:.5f} MSE {mse:.5f} MAE {mae:.5f}")

        results_dir = self._build_results_dir(setting)
        run_dt = self.run_dt or datetime.now()
        np.save(os.path.join(results_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(results_dir, "pred.npy"), preds)
        np.save(os.path.join(results_dir, "true.npy"), trues)
        self._save_visualization(results_dir, preds, trues)

        if save_tokens:
            self._save_tokens(test_loader, results_dir)

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
