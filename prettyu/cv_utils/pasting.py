import cv2
import numpy as np
from . import pointing


def affine_transform(src_img, src_pts, dst_img, dst_pts):
    M = cv2.estimateAffinePartial2D(np.array(src_pts), np.array(dst_pts))[0]
    transformed_img = cv2.warpAffine(src_img, M, (dst_img.shape[1], dst_img.shape[0]), dst_img, borderMode=cv2.BORDER_TRANSPARENT)
    return transformed_img


def face_paste(user_lora_img, style_lora_img):
    user_face_points, user_face_box = pointing.face_points_detect(user_lora_img, return_box=True, relative=True)
    x_min, y_min, x_max, y_max = user_face_box
    user_face_box_img = user_lora_img[y_min: y_max, x_min: x_max, :]
    style_face_points = pointing.face_points_detect(style_lora_img, return_box=False, relative=False)
    pasted_head_img = affine_transform(user_face_box_img, user_face_points, style_lora_img, style_face_points)
    return pasted_head_img
    

def multi_face_paste(user_lora_img_list, style_lora_img):
    style_face_points_list = pointing.face_points_detect(style_lora_img, return_box=False, relative=False, single=False)
    if len(style_face_points_list) != len(user_lora_img_list):
        raise ValueError(f'total faces in style image({len(style_face_points_list)}) not match total faces in user lora({len(user_lora_img_list)})')
    
    # paste from left to right
    for i in range(len(user_lora_img_list)):
        user_lora_img = user_lora_img_list[i]
        style_face_points = style_face_points_list[i]
        user_face_points, user_face_box = pointing.face_points_detect(user_lora_img, return_box=True, relative=True)
        x_min, y_min, x_max, y_max = user_face_box
        user_face_box_img = user_lora_img[y_min: y_max, x_min: x_max, :]
        pasted_head_img = affine_transform(user_face_box_img, user_face_points, style_lora_img, style_face_points)

    return pasted_head_img
