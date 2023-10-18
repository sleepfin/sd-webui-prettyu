import hashlib
import base64
import os
import subprocess
import sys
import shutil
import uuid
import threading
import time
import re

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.join(os.path.dirname(CUR_DIR), 'prettyu')


class DataFileAlreadyExistsError(ValueError):
    pass


def hash10(input_str):
   return hashlib.md5(input_str.encode('utf-8')).hexdigest()[:10]


def create_lock(unique_name, unique_hash_id, gender):
    lock_dir = os.path.join(PROJ_DIR, 'lora', 'data')
    os.makedirs(lock_dir, exist_ok=True)
    lora_data_lock_file = os.path.join(lock_dir, f'{unique_name}({unique_hash_id}).lock')
    if os.path.exists(lora_data_lock_file):
        raise DataFileAlreadyExistsError()
    
    lock_tmp_file = os.path.join(PROJ_DIR, 'lora', 'data', uuid.uuid4().hex)
    with open(lock_tmp_file, 'w') as f:
        f.write(gender)
        f.flush()
    
    os.rename(lock_tmp_file, lora_data_lock_file)


def update_pbar(log_file_path, gradio_pbar, event: threading.Event):
    # wait until log_file is created
    while not os.path.exists(log_file_path):
        if event.is_set():
            return
        
        time.sleep(0.5)

    # tail the log_file
    with open(log_file_path, 'r') as fp:
        while True:
            if event.is_set():
                return
            
            new_log = fp.readline().strip()
            if new_log:
                # print(new_log)
                # reg to match logs like:
                # steps:  14%|█   | 957/7020 [09:48<1:02:09,  1.63it/s, loss=0.00542]
                pattern = r"steps:\s+([0-9]|[1-9][0-9]|100)%\|(.*?)\|\s+(.*)"
                match = re.match(pattern, new_log)
                if match:
                    p, _, msg = match.groups()
                    gradio_pbar(int(p) / 100., desc=msg)
            else:
                time.sleep(0.5)


def train_sd15_lora_pipeline(    
        sd_model_checkpoint: str,
        unique_name: str,
        gender: str,
        train_images: list,
        train_steps=7000,
        train_epochs=20,
        train_batch_size=2,
        res=512,
        progress=None,
        xformers=True,
        mixed_precision="no",
        cache_latents=True,
        gradient_checkpointing=False,
        max_data_loader_n_workers=8,
        *args):
    
    unique_hash_id = hash10(unique_name)
    print(f'Start Training. sd={sd_model_checkpoint}, name={unique_name}({unique_hash_id}), gender={gender}, num_imgs={len(train_images)}, args={args}')

    # move training images to lora folder
    create_lock(unique_name, unique_hash_id, gender)
    lora_data_folder = os.path.join(PROJ_DIR, 'lora', 'data', 'raw', unique_hash_id)
    os.makedirs(lora_data_folder, exist_ok=True)
    for temp_file in train_images:
        temp_file.flush()
        shutil.copyfile(temp_file.name ,os.path.join(lora_data_folder, os.path.basename(temp_file.name)))


    train_script = os.path.join(PROJ_DIR, 'third_party', 'sd-scripts', 'train_network.py')
    if not os.path.exists(train_script):
        raise ValueError(f'train script not found in {train_script}. you should init git submodule first. execute: `git submodule update --init`')
    
    # preprocess training images
    prepare_data_script = os.path.join(PROJ_DIR, 'prepare_lora_data.py')
    prepare_data_cmd = f'{sys.executable} {prepare_data_script} '\
                       f'--gender="{gender}" ' \
                       f'--name="{unique_hash_id}" ' \
                       f'--train_steps={train_steps} ' \
                       f'--train_epochs={train_epochs} ' \
                       f'--train_batch_size={train_batch_size} '
    print(prepare_data_cmd)
    subprocess.check_call(prepare_data_cmd, shell=True)

    # train lora by kohya
    progress(0.0, "Start Training...")
    train_script = os.path.join(PROJ_DIR, 'train_sd15_lora.py')
    os.makedirs(log_dir := os.path.join(PROJ_DIR, 'lora', 'logs'), exist_ok=True)
    log_file_path = os.path.join(log_dir, f'train_sd15_lora_{unique_hash_id}.log')
    train_script = f'{sys.executable} {train_script} '\
                   f'--gender="{gender}" ' \
                   f'--name="{unique_hash_id}" ' \
                   f'--train_steps={train_steps} ' \
                   f'--train_epochs={train_epochs} ' \
                   f'--train_batch_size={train_batch_size} ' \
                   f'--res={res} ' \
                   f'--pretrained_model={sd_model_checkpoint} ' \
                   f'--xformers={xformers} ' \
                   f'--mixed_precision={mixed_precision} ' \
                   f'--cache_latents={cache_latents} ' \
                   f'--gradient_checkpointing={gradient_checkpointing} ' \
                   f'--max_data_loader_n_workers={max_data_loader_n_workers} ' \
                   f'> {log_file_path} 2>&1'
    
    # update progress in sub thread
    event = threading.Event()
    t = threading.Thread(target=update_pbar, args=(log_file_path, progress, event))
    t.daemon = True
    t.start()

    print(train_script)
    subprocess.check_call(train_script, shell=True)
    event.set()

    # copy model
    additional_networks_lora_path = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 'sd-webui-additional-networks', 'models', 'lora', f'{unique_hash_id}.safetensors')
    latest_model_path = os.path.join(PROJ_DIR, 'lora', 'models', f'sd15_{unique_hash_id}', 'last.safetensors')
    os.symlink(latest_model_path, additional_networks_lora_path)
