import argparse
import os

import torch

from exp.exp_token_llm import TokenLLM_Main, build_setting
from utils.tools import set_random_seed


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def infer_num_channels(args):
    if not args.use_multivariate:
        return 1

    csv_path = os.path.join(args.root_path, args.data_path)
    with open(csv_path, "r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")
    return max(1, len(header) - 1)


def resolve_gpt_local_path(path):
    if path is None:
        return None
    return os.path.abspath(path)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--save_tokens", type=str2bool, default=True)
    parser.add_argument("--zero_shot", type=str2bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="ct_gpt2")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./data/ETT")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--patch_stride", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_linear_shortcut", type=str2bool, default=True)
    parser.add_argument("--use_chronos_scaling", type=str2bool, default=False)
    parser.add_argument("--scaling_eps", type=float, default=1e-8)
    parser.add_argument("--gpt_model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--gpt_local_path", type=str, default="./gpt")
    parser.add_argument("--use_pretrained_gpt2", type=str2bool, default=True)
    parser.add_argument("--prefer_local_gpt2", type=str2bool, default=True)
    parser.add_argument("--gpt_local_files_only", type=str2bool, default=True)
    parser.add_argument("--freeze_gpt2", type=str2bool, default=False)
    parser.add_argument("--gpt2_trainable_layers", type=int, default=2)
    parser.add_argument("--decoder_hidden_dim", type=int, default=768)
    parser.add_argument("--decoder_dropout", type=float, default=0.2)
    parser.add_argument("--num_output_mixtures", type=int, default=1)
    parser.add_argument("--num_sampling_paths", type=int, default=0)
    parser.add_argument("--eval_num_sampling_paths", type=int, default=20)
    parser.add_argument("--eval_use_sampling", type=str2bool, default=False)
    parser.add_argument("--min_log_variance", type=float, default=-6.0)
    parser.add_argument("--max_log_variance", type=float, default=2.0)
    parser.add_argument("--use_alignment", type=str2bool, default=True)
    parser.add_argument("--use_trend_loss", type=str2bool, default=True)
    parser.add_argument("--use_con_loss", type=str2bool, default=True)
    parser.add_argument("--use_trend_regression", type=str2bool, default=True)
    parser.add_argument("--alignment_hidden_dim", type=int, default=512)
    parser.add_argument("--alignment_dropout", type=float, default=0.2)
    parser.add_argument("--alignment_augmentation_std", type=float, default=0.02)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--use_token_distribution_loss", type=str2bool, default=True)
    parser.add_argument("--lambda_token", type=float, default=0.2)
    parser.add_argument("--token_distribution_samples", type=int, default=256)
    parser.add_argument("--token_distribution_bandwidth", type=float, default=1.0)
    parser.add_argument("--token_moment_weight", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stop_metric", type=str, default="loss")
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_point", type=float, default=0.5)
    parser.add_argument("--lambda_diff", type=float, default=0.2)
    parser.add_argument("--lambda_con", type=float, default=0.1)
    parser.add_argument("--lambda_trend", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--teacher_forcing_ratio_start", type=float, default=1.0)
    parser.add_argument("--teacher_forcing_ratio_end", type=float, default=0.3)
    parser.add_argument("--teacher_forcing_anneal_epochs", type=int, default=10)
    parser.add_argument("--lradj", type=str, default="type3")
    parser.add_argument("--scheduler_type", type=str, default="warmup_cosine")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_multivariate", type=str2bool, default=False)
    parser.add_argument("--use_multi_gpu", type=str2bool, default=False)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--target_col", type=str, default="OT")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    args.use_multivariate = bool(args.use_multivariate)
    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()
    args.use_multi_gpu = bool(args.use_multi_gpu)
    args.use_amp = bool(args.use_amp) and args.use_gpu
    args.gpt_local_path = resolve_gpt_local_path(args.gpt_local_path)
    args.freeze_gpt2 = bool(args.freeze_gpt2)
    args.gpt2_trainable_layers = max(0, int(args.gpt2_trainable_layers))
    args.decoder_hidden_dim = None if args.decoder_hidden_dim <= 0 else args.decoder_hidden_dim
    args.alignment_hidden_dim = None if args.alignment_hidden_dim <= 0 else args.alignment_hidden_dim
    args.eval_num_sampling_paths = max(0, int(args.eval_num_sampling_paths))
    args.eval_use_sampling = bool(args.eval_use_sampling)
    args.early_stop_metric = str(args.early_stop_metric).lower()
    args.use_token_distribution_loss = bool(args.use_token_distribution_loss)
    args.token_distribution_samples = max(8, int(args.token_distribution_samples))
    args.patch_size = max(2, int(args.patch_size))
    args.patch_stride = max(1, int(args.patch_stride))
    args.teacher_forcing_ratio_start = float(args.teacher_forcing_ratio_start)
    args.teacher_forcing_ratio_end = float(args.teacher_forcing_ratio_end)
    args.teacher_forcing_anneal_epochs = max(1, int(args.teacher_forcing_anneal_epochs))
    args.devices = str(args.devices if args.devices is not None else args.gpu)
    args.device_ids = [
        int(device)
        for device in args.devices.replace(" ", "").split(",")
        if device != ""
    ] or [args.gpu]
    if args.use_multi_gpu:
        args.gpu = args.device_ids[0]

    args.features = "M" if args.use_multivariate else "S"
    args.target = args.target_col
    args.c_in = infer_num_channels(args)
    args.c_out = args.c_in if args.use_multivariate else 1
    if args.use_multivariate:
        raise ValueError(
            "The current CT-GPT2 implementation only supports univariate inputs. "
            "Please run with `--use_multivariate false`."
        )
    return args


def main():
    args = build_args()
    if args.seed is not None:
        set_random_seed(args.seed)

    setting = build_setting(args)
    exp = TokenLLM_Main(args)

    should_train = bool(args.is_training) and not args.zero_shot
    load_checkpoint = not args.zero_shot or bool(args.checkpoint)

    if should_train:
        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)

    if args.zero_shot:
        mode = "zero-shot evaluating"
    else:
        mode = "testing" if args.is_training else "evaluating"

    print(f">>>>>>>{mode} : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    exp.test(
        setting,
        checkpoint_path=args.checkpoint,
        save_tokens=args.save_tokens,
        load_checkpoint=load_checkpoint,
    )


if __name__ == "__main__":
    main()
