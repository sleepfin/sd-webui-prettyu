import numpy as np
import cv2


def keep_ratio_resize(img, target_size=1024):
    # 获取图像尺寸
    h, w, _ = img.shape

    # 计算缩放因子和新的尺寸
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    # 调整图像大小
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized


def pad_to_square(img):
    # 获取图像尺寸
    h, w, _ = img.shape
    if h == w:
        return img
    
    long = max(h, w)
    
    # 创建白色背景
    background = np.ones((long, long, 3), dtype=np.uint8) * 255

    # 计算padding的位置
    y_offset = (long - h) // 2
    x_offset = (long - w) // 2

    # 将调整大小的图像粘贴到白色背景上
    background[y_offset:y_offset + h, x_offset:x_offset + w] = img

    return background


def resize_and_pad(img, target_size=1024):
    img_resized = keep_ratio_resize(img, target_size=target_size)
    target_img = pad_to_square(img_resized)
    return target_img
