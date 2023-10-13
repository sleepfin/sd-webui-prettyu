import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class PersonRGBMaskNumberNotMatch(ValueError):
    pass

def fix_mask(mask, k=None):
    h, w = mask.shape[:2]
    if k is None:
        r = 0.05
        k = int(r * (h + w) / 2.0)
        if k % 2 == 0:
            k += 1

    print(f"using kernel={k}, h={h}, w={w}")
    kernel =  np.ones((k, k),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)
    return mask


def keep_largest_mask_area(mask):
    # 查找所有联通区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 通过面积找到最大联通区域
    max_contour = max(contours, key=cv2.contourArea)

    # 创建一个全0的图像，大小与原mask相同
    output_mask = np.zeros_like(mask)

    # 将最大联通区域填充为1
    cv2.drawContours(output_mask, [max_contour], -1, 1, thickness=cv2.FILLED)
    return output_mask


class PersonSegmentation:
    def __init__(self):
        self.pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')

    def __call__(self, img):
        return self.pipeline(img)


def segment_person(img):
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = PersonSegmentation()

    return _PIPELINE(img)


def segment_face_and_hair(img, pad_color=255):
    seg_result = segment_person(img)

    mask_array = []
    for i in range(len(seg_result['labels'])):
        if seg_result['labels'][i] in ['Hair', 'Face']:
            mask_array.append(seg_result['masks'][i])

    mask = np.max(mask_array, axis=0)
    mask = fix_mask(mask, k=11)
    mask = keep_largest_mask_area(mask)

    ret_img = pad_color * np.ones(img.shape, dtype=img.dtype)
    ret_img[mask == 1] = img[mask == 1]
    mask_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
    return ret_img, mask_ratio


def segment_head_for_mask(img, dilation=4):
    seg_result = segment_person(img)

    hairs, faces = [], []
    for i in range(len(seg_result['labels'])):
        if seg_result['labels'][i] == 'Hair':
            hairs.append(seg_result['masks'][i])
        if seg_result['labels'][i] == 'Face':
            faces.append(seg_result['masks'][i])

    if len(hairs) != len(faces):
        raise ValueError(f'num haris({len(hairs)}) not match num faces({len(faces)})')
    
    all_mask = []
    for i in range(len(faces)):
        f, h = faces[i], hairs[i]
        mask = np.max([f, h], axis=0)
        mask = fix_mask(mask, k=11)
        mask = keep_largest_mask_area(mask)
        if dilation:
            kernel = np.ones((dilation, dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask[mask>0.5] = 1
            mask[mask<=0.5] = 0
        
        mask[mask==1] = 255
        all_mask.append(mask)

    return all_mask


def segment_person_rgb(img, output_ratio=1., expected_num=None):
    seg_result = segment_person(img)

    rgb_mask_list = []

    for i in range(len(seg_result['labels'])):
        if seg_result['labels'][i] == 'Human':
            human_mask = seg_result['masks'][i]
            human_mask = fix_mask(human_mask)
            human_mask = human_mask * 255
            rgb_mask_list.append(human_mask)
            if len(rgb_mask_list) > 3:
                raise PersonRGBMaskNumberNotMatch('More than 3 persons is not supported.')

    if len(rgb_mask_list) < 1:
        raise PersonRGBMaskNumberNotMatch('At least 1 person is needed')
    
    if expected_num is not None and len(rgb_mask_list) != expected_num:
        raise PersonRGBMaskNumberNotMatch(f'rgb mask number={len(rgb_mask_list)} not match expected number={expected_num}')

    while len(rgb_mask_list) < 3:
        rgb_mask_list.append(np.zeros(shape=img.shape[:2], dtype=np.uint8))

    rgb_mask = np.stack(rgb_mask_list, axis=-1)
    if output_ratio != 1:
        new_w, new_h = int(output_ratio * rgb_mask.shape[1]), int(output_ratio * rgb_mask.shape[0])
        rgb_mask = cv2.resize(rgb_mask, (new_w, new_h))

    return rgb_mask


def rgb_mask_choose(rgb_mask, channel):
    return rgb_mask[:, :, channel] / 255

_PIPELINE = None
