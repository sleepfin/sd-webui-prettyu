import os
import ast
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("--gender", default='a young man', type=str)
parser.add_argument("--name", default='zhangzhenyu', type=str)
parser.add_argument("--train_steps", default=7000, type=int)
parser.add_argument("--train_epochs", default=20, type=int)
parser.add_argument("--train_batch_size", default=2, type=int)
parser.add_argument("--res", default=512, type=int)
parser.add_argument("--xformers", default=True, type=ast.literal_eval)
parser.add_argument("--mixed_precision", default="no", type=str, choices=["no", "fp16", "bf16"])
parser.add_argument("--cache_latents", default=True, type=ast.literal_eval)
parser.add_argument("--gradient_checkpointing", default=False, type=ast.literal_eval)
parser.add_argument("--pretrained_model", default="/home/zzy2/workspace/stable-diffusion-webui/models/Stable-diffusion/majicmixRealistic_betterV2V25.safetensors", type=str)
parser.add_argument("--max_data_loader_n_workers", default=8, type=int)

args = parser.parse_args()


def main():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    train_script = os.path.join(cur_dir, 'third_party', 'sd-scripts', 'train_network.py')
    if not os.path.exists(train_script):
        raise ValueError(f'train script not found in {train_script}. you should init git submodule first. execute: `git submodule update --init`')
    
    prepare_data_script = os.path.join(cur_dir, 'prepare_lora_data.py')
    prepare_data_cmd = f'{sys.executable} {prepare_data_script} '\
                       f'--gender="{args.gender}" ' \
                       f'--name="{args.name}" ' \
                       f'--train_steps={args.train_steps} ' \
                       f'--train_epochs={args.train_epochs} ' \
                       f'--train_batch_size={args.train_batch_size} ' \
                       f'--res={args.res}'
    print(prepare_data_cmd)
    subprocess.check_call(prepare_data_cmd, shell=True)

    train_script = os.path.join(cur_dir, 'train_sd15_lora.py')
    train_script = f'{sys.executable} {train_script} '\
                   f'--gender="{args.gender}" ' \
                   f'--name="{args.name}" ' \
                   f'--train_steps={args.train_steps} ' \
                   f'--train_epochs={args.train_epochs} ' \
                   f'--train_batch_size={args.train_batch_size} ' \
                   f'--res={args.res} ' \
                   f'--pretrained_model={args.pretrained_model} ' \
                   f'--xformers={args.xformers} ' \
                   f'--mixed_precision={args.mixed_precision} ' \
                   f'--cache_latents={args.cache_latents} ' \
                   f'--max_data_loader_n_workers={args.max_data_loader_n_workers} ' \
                   f'--gradient_checkpointing={args.gradient_checkpointing}'
    print(train_script)
    subprocess.check_call(train_script, shell=True)


if __name__ == "__main__":
    main()
