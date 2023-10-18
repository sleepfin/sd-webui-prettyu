import os
import cv2
import glob
import tqdm
import math
import logging
import argparse

import numpy as np

from cv_utils import segmentation, retouching, resizing, common
from PIL import Image


SUPPORTED_EXT = ['.jpg', '.jpeg', '.png', '.bmp']

parser = argparse.ArgumentParser()

parser.add_argument("--gender", default='a young man', type=str)
parser.add_argument("--name", default='zhangzhenyu', type=str)
parser.add_argument("--res", default=512, type=int)
parser.add_argument("--train_steps", default=7000, type=int)
parser.add_argument("--train_epochs", default=20, type=int)
parser.add_argument("--train_batch_size", default=2, type=int)

args = parser.parse_args()


def mask_to_rgb(mask, color_map):
    """Convert a labeled mask to RGB image using color map."""
    
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for label, color in color_map.items():
        rgb_image[mask == label] = color
    
    return np.array(rgb_image)


def remove_black_border(img, mask):
    # 找到所有非零像素
    non_zero_rows = np.any(mask, axis=1)
    non_zero_cols = np.any(mask, axis=0)

    # 找到非零像素的边界
    non_zero_rows_idx = np.where(non_zero_rows)[0]
    non_zero_cols_idx = np.where(non_zero_cols)[0]

    # 裁剪图像
    cropped_img = img[non_zero_rows_idx[0]:non_zero_rows_idx[-1] + 1, non_zero_cols_idx[0]:non_zero_cols_idx[-1] + 1]
    cropped_mask = mask[non_zero_rows_idx[0]:non_zero_rows_idx[-1] + 1, non_zero_cols_idx[0]:non_zero_cols_idx[-1] + 1]
    return cropped_img, cropped_mask


def get_prompt(output_dir):
    candidates = ['looking at viewer', 'realistic', 'simple background', 'solo', 'white background']
    if args.gender:
        candidates.insert(0, args.gender)
    if args.name:
        candidates.insert(0, args.name)

    for img_path in tqdm.tqdm(glob.glob(os.path.join(output_dir, '**', '*.png'), recursive=True)):
        with open(os.path.splitext(img_path)[0] + '.txt', 'w') as f:
            f.write(','.join(candidates))


def rename_output_dir(output_dir):
    num_images = len(os.listdir(output_dir))
    iter_num = args.train_batch_size * args.train_steps / args.train_epochs / num_images
    iter_num = int(math.ceil(iter_num))
    print(f' iter_num={iter_num}, train_steps={args.train_steps}, train_epochs={args.train_epochs}, train_batch_size={args.train_batch_size}, num_images={num_images}')
    new_base_name = f'{iter_num}_{os.path.basename(output_dir)}'
    new_output_dir = os.path.join(os.path.dirname(output_dir), new_base_name)
    os.rename(output_dir, new_output_dir)
    

def get_face_images(data_dir, output_dir):
    output_dir = os.path.join(output_dir, os.path.basename(data_dir))
    os.makedirs(output_dir, exist_ok=True)
    for file_path in tqdm.tqdm(glob.glob(os.path.join(data_dir, '**', '*.*'), recursive=True)):
        if os.path.splitext(file_path)[1].lower() not in SUPPORTED_EXT:
            continue

        # img = cv2.imread(file_path)  # cv2.imread不支持中文路径
        img = common.read_img_rgb(file_path)
        # 缩放
        img_resized = resizing.keep_ratio_resize(img, target_size=args.res)
        # 美颜
        img_retouched = retouching.skin_retouch(img_resized)
        # 分割脸部和头发
        seg_img, mask_ratio = segmentation.segment_face_and_hair(img_retouched)
        # pad
        pad_img = resizing.pad_to_square(seg_img)
        # TODO：脸部关键点检测并转正
        
        # 数据增强，将头部放小
        if mask_ratio > 0.2:
            resized_pad_images = smaller_face_augmentation(pad_img, ratio=[1.0, 0.5, 0.25])
        elif mask_ratio > 0.05:
            resized_pad_images = smaller_face_augmentation(pad_img, ratio=[1.0, 0.5])
        else:
            resized_pad_images = [pad_img]

        for i, result_img in enumerate(resized_pad_images):
            file_rel_path = os.path.relpath(file_path, data_dir)
            file_rel_path = os.path.splitext(file_rel_path)[0] + f'_{i}.png'
            save_file_path = os.path.join(output_dir, file_rel_path)
            # imwrite not support chinses
            # cv2.imwrite(save_file_path, result_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            Image.fromarray(result_img).save(save_file_path, quality=100)
    
    rename_output_dir(output_dir)


def smaller_face_augmentation(img, ratio=[1.0, 0.5, 0.25]):
    # 把图片按ratio缩小，并在周围padding一圈白色背景
    h, w, _ = img.shape
    resized_images = []
    for r in ratio:
        if r == 1:
            resized_images.append(img)
            continue

        new_w, new_h = int(w * r), int(h * r)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 计算padding的位置
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2

        # 将调整大小的图像粘贴到白色背景上
        background = np.ones((h, w, 3), dtype=np.uint8) * 255
        background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        resized_images.append(background)
    
    return resized_images


def main():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cur_dir, 'lora', 'data', 'raw', args.name)
    data_output_dir = os.path.join(cur_dir, 'lora', 'data', f'train_{args.name}')

    if not os.path.exists(data_dir):
        raise ValueError(f"No data found in {data_dir}. You should put your Lora image data in `lora/data/raw/$YOUR_NAME/`, and start this script with --name=$YOUR_NAME")
    
    if os.path.exists(data_output_dir) and len(glob.glob(os.path.join(data_output_dir, '**', '*.*'), recursive=True)) > 0:
        raise ValueError(f"output data already exists in {data_output_dir}. you should remove everything in this folder first.")
    
    data_files = glob.glob(os.path.join(data_dir, '**', '*.*'), recursive=True)
    data_files = list(filter(lambda x: os.path.splitext(x)[1].lower() in SUPPORTED_EXT, data_files))
    if len(data_files) <= 0:
        raise ValueError("No data found. You should put your Lora image data in `lora/data/raw/$YOUR_NAME/`, and start this script with --name=$YOUR_NAME")
    elif len(data_files) < 20:
        logging.warning(f"image data is less than 20({len(data_files)}), more images will increase the performance of Lora model.")

    get_face_images(data_dir, data_output_dir)
    get_prompt(data_output_dir)


if __name__ == "__main__":
    main()
