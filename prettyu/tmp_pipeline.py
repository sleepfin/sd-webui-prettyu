
import cv2
import numpy as np
from sd.pipelines import StableDiffusionGenerator, ControlnetOpenpose, ControlnetCanny
from cv_utils import pasting, segmentation


def f1():
    """
    先放弃，暂时不知道怎么加Lora
    """
    prompt="a man"
    negative_prompt='low quality, bad quality, sketches'
    controlnet_type = 'openpose'

    user_lora_img = cv2.imread("/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/user_lora.png")
    style_lora_img = cv2.imread("/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/style.png")
    pasted_head_img = pasting.face_paste(user_lora_img, style_lora_img)

    if controlnet_type == 'openpose':
        controlnet = ControlnetOpenpose()
        controlnet_input = pasted_head_img
    elif controlnet_type == 'canny':
        controlnet = ControlnetCanny()
        controlnet_input = style_lora_img
    else:
        raise ValueError()

    # lora get error for now
    lora_path1 = r"/home/zzy2/workspace/stable-diffusion-webui/models/Lora/locon_aug3.safetensors"
    sd = StableDiffusionGenerator(controlnet=controlnet, lora_paths=[lora_path1])

    results = sd(prompt=prompt, negative_prompt=negative_prompt, controlnet_input=controlnet_input)

    cv2.imwrite('./test.png', np.array(results[0]))


def f2():
    style_image = cv2.imread("/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/2person_style9.png")
    rgb_mask = segmentation.segment_person_rgb(style_image)
    cv2.imwrite("/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/rgb_mask9.png", cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))
    

def f3():
    user_lora_img_1 = cv2.imread("/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/zzy.jpg")
    user_lora_img_2 = cv2.imread("/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/jn.jpg")
    style_lora_img = cv2.imread("/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/2person_style2.png")
    pasted_head_img = pasting.multi_face_paste([user_lora_img_1, user_lora_img_2], style_lora_img)
    cv2.imwrite('./pasted_head_img2.png', pasted_head_img)

def f4():
    face_img = cv2.imread("/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/face_to_mask.png")
    all_mask = segmentation.segment_head_for_mask(face_img)
    for i, mask in enumerate(all_mask):
        cv2.imwrite(f'/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/facemask{i}.png', mask)



def f5():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    from cv_utils import face_utils

    fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition, 'damo/cv_resnet34_face-attribute-recognition_fairface')
    src_img_path = '/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/couple_saved/style/good/2.png'

    face_photo = face_utils.PortraitPhoto(image_path=src_img_path, expected_face_num=1)



    raw_result1 = fair_face_attribute_func(face_photo.face_images[0].face_img)
    print('face attribute output1: {} - {}.'.format(raw_result1, face_photo.face_images[0].box))

    raw_result2 = fair_face_attribute_func(face_photo.face_images[1].face_img)
    print('face attribute output2: {} - {}.'.format(raw_result2, face_photo.face_images[1].box))

f2()