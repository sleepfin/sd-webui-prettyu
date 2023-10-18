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


def train_lora(train_data_dir):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    train_script = os.path.join(cur_dir, 'third_party', 'sd-scripts', 'train_network.py')
    model_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lora", "models", f"sd15_{args.name}")
    cmd = f'{sys.executable} {train_script} '\
          f'--pretrained_model_name_or_path={args.pretrained_model} ' \
          f'--train_data_dir={train_data_dir} ' \
          f'--output_dir={model_output_dir} ' \
          f'--prior_loss_weight=1.0 ' \
          f'--resolution="{args.res},{args.res}" ' \
          f'--train_batch_size={args.train_batch_size} ' \
          f'--unet_lr=1e-4 ' \
          f'--text_encoder_lr=1e-5 ' \
          f'--max_train_epochs=20 ' \
          f'--save_every_n_epochs=20 ' \
          f'--save_model_as="safetensors" ' \
          f'--clip_skip=2 ' \
          f'--seed=42 ' \
          f'--no_half_vae ' \
          f'--network_module="networks.lora" ' \
          f'--network_dim=64 ' \
          f'--network_alpha=32 ' \
          f'--lr_scheduler=cosine_with_restarts ' \
          f'--lr_scheduler_num_cycles=1 ' \
          f'--max_data_loader_n_workers={args.max_data_loader_n_workers} ' \
          f'--mixed_precision={args.mixed_precision}'
    
    if args.xformers:
        cmd += " --xformers"
    if args.cache_latents:
        cmd += " --cache_latents"
    if args.gradient_checkpointing:
        cmd += " --gradient_checkpointing"

    print(cmd)
    subprocess.check_call(cmd, shell=True)


def main():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_output_dir = os.path.join(cur_dir, 'lora', 'data', f'train_{args.name}')

    train_lora(data_output_dir)
    

if __name__ == "__main__":
    main()
