import argparse
import os

import torch

from exp.exp_token_llm import TokenLLM_Main
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


def build_setting(args):
    return (
        f"{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_"
        f"ps{args.patch_size}_st{args.stride}_dm{args.d_model}_v{args.vocab_size}_"
        f"predgpt2"
    )


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
    parser.add_argument("--model", type=str, default="token_llm_forecasting")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./data/ETT")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpt_model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--gpt_local_path", type=str, default="./gpt")
    parser.add_argument("--use_pretrained_gpt2", type=str2bool, default=True)
    parser.add_argument("--prefer_local_gpt2", type=str2bool, default=True)
    parser.add_argument("--gpt_local_files_only", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lradj", type=str, default="type3")
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
