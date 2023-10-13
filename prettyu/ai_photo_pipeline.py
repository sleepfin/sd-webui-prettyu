import os
import json
import logging

import numpy as np
from sd import interfaces
from cv_utils import common as cv_common
from cv_utils import face_utils
from cv_utils import segmentation
from cv_utils import gender_const
from cv_utils.gender_const import GenderPrompt
from prompts import girl_prompts, couple_prompts

    
steps = 40
width = 512
height = 512
CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def solo_girl_photo(name, 
                    lora, 
                    style, 
                    steps=40, 
                    width=512, 
                    height=512, 
                    batch_size=1,
                    n_iter=1, 
                    gender=gender_const.GIRL, 
                    negative_style="no_glasses", 
                    body_style="upper body", 
                    seed=-1):
    # 生成girl图片
    prompt = ",".join([item for item in [gender, 
                                         name, 
                                         girl_prompts.base_prompt.format(body_style), 
                                         girl_prompts.light_prompt, 
                                         girl_prompts.background_prompt[style], 
                                         girl_prompts.clothing_prompt[style],
                                         girl_prompts.hair_prompt[style],
                                         girl_prompts.posing_prompt[style]]if item])
    adtailer_face_prompt_tmp = girl_prompts.adtailer_face_prompt.format(gender)
    adtailer_hand_prompt_tmp = girl_prompts.adtailer_hand_prompt.format(gender)
    with open(os.path.join(CUR_DIR, 'sd', 'jsontemp', 'solo.json'), 'r') as f:
        request_json = json.load(f)

    # set base
    request_json["steps"] = steps
    request_json["width"] = width
    request_json["height"] = height
    request_json["prompt"] = prompt
    request_json["negative_prompt"] = girl_prompts.negative_prompt[negative_style]
    request_json["batch_size"] = batch_size
    request_json["n_iter"] = n_iter
    request_json["seed"] = seed

    # set lora
    lora_names = [lora]
    for i in range(len(lora_names)):
        request_json["alwayson_scripts"]["additional networks for generating"]["args"][i * 4 + 3] = lora_names[i]

    # TODO: 如果是id_photo，可以考虑不要adtailer，但是最好加一个depth的controlnet，不然background不一定是纯色
    # set adtailer
    request_json["alwayson_scripts"]["adetailer"]["args"][1]["ad_prompt"] = adtailer_face_prompt_tmp
    request_json["alwayson_scripts"]["adetailer"]["args"][1]["ad_negative_prompt"] = girl_prompts.adtailer_face_negative_prompt[negative_style]
    request_json["alwayson_scripts"]["adetailer"]["args"][2]["ad_prompt"] = adtailer_hand_prompt_tmp
    request_json["alwayson_scripts"]["adetailer"]["args"][2]["ad_negative_prompt"] = girl_prompts.adtailer_hand_negative_prompt

    images_base64 = interfaces.txt2imgapi(request_json)
    imgs = [cv_common.base64_to_img(img_base64) for img_base64 in images_base64]
    return imgs


def gen_girl_solo(num_imgs, 
                  name, 
                  lora, 
                  style, 
                  save_dir, 
                  anchor_file_pattern, 
                  face_model_file, 
                  width=512, 
                  height=512, 
                  gender=gender_const.GIRL, 
                  negative_style='no_glasses', 
                  body_style='upper body', 
                  n_iter=10):
    # 单人girl
    # 1. 生成指定Lora的图片（Lora+ADtailer_face&hand）
    # 2. 后处理评价图片，从num_imgs*n_iter张图片中保留得分最高的num_imgs张图片
    if not (1 <= num_imgs <= 8):
        raise ValueError('only 1~8 images can be generated. If you want to generate more images, call gen_solo multiple times.')
    
    face_model = face_utils.get_face_model(face_model_file)
    face_anchor = face_utils.FaceIdentity(face_model, file_pattern=anchor_file_pattern)

    imgs = solo_girl_photo(name=name, 
                            gender=gender, 
                            lora=lora, 
                            style=style,
                            width=width,
                            height=height,
                            batch_size=num_imgs, 
                            n_iter=n_iter, 
                            negative_style=negative_style, 
                            body_style=body_style, 
                            seed=-1)
    good_face_photos = []
    bad_face_photos = []
    for img in imgs:
        face_photo = face_utils.PortraitPhoto(img=img)
        face_photo.evaluate(body_style=body_style)
        if face_photo.main_feedback == 'good':
            good_face_photos.append(face_photo)
        else:
            bad_face_photos.append(face_photo)
            
    # 这里good face必然只有一个face
    good_face_photos_scores = [face_anchor.similarity(pht.face_images[0]) for pht in good_face_photos]

    sorted_tuples = sorted(zip(good_face_photos_scores, good_face_photos), key=lambda x: x[0], reverse=True)
    good_face_photos_scores, good_face_photos = zip(*sorted_tuples)

    os.makedirs(best_save_dir := os.path.join(save_dir, 'best'), exist_ok=True)
    os.makedirs(dissim_save_dir := os.path.join(save_dir, 'dissim'), exist_ok=True)
    os.makedirs(bad_save_dir := os.path.join(save_dir, 'bad'), exist_ok=True)

    for i in range(len(good_face_photos)):
        good_photo = good_face_photos[i]
        good_photo_score = good_face_photos_scores[i]
        if i < num_imgs:
            good_photo.save(os.path.join(best_save_dir, f"{i}_{good_photo_score: .3f}.png"))
        else:
            good_photo.save(os.path.join(dissim_save_dir, f"{i}_{good_photo_score: .3f}.png"))
    
    for i, bad_photo in enumerate(bad_face_photos):
        os.makedirs(bad_save_dir_with_feedback := os.path.join(bad_save_dir, bad_photo.main_feedback.replace(' ', '_')), exist_ok=True)
        bad_photo.save(os.path.join(bad_save_dir_with_feedback, f"{i}.png"))


def couple_style_photo(style, 
                       steps=20,
                       width=512, 
                       height=512, 
                       batch_size=1, 
                       n_iter=1, 
                       gender1=gender_const.YOUNG_MAN, 
                       gender2=gender_const.GIRL, 
                       negative_style="no_glasses", 
                       body_style="upper body", 
                       seed=-1):
    # 生成couple的风格图片
    gender_prompt = GenderPrompt.merge(gender1, gender2, weight=1.5)
    prompt = ",".join([item for item in [gender_prompt, 
                                         couple_prompts.base_prompt.format(body_style), 
                                         couple_prompts.light_prompt, 
                                         couple_prompts.background_prompt[style], 
                                         couple_prompts.clothing_prompt[style],
                                         couple_prompts.hair_prompt[style],
                                         couple_prompts.posing_prompt[style]] if item])
    
    with open(os.path.join(CUR_DIR, 'sd', 'jsontemp', 'couple_style.json'), 'r') as f:
        request_json = json.load(f)

    # set base
    request_json["steps"] = steps
    request_json["width"] = width
    request_json["height"] = height
    request_json["prompt"] = prompt
    request_json["negative_prompt"] = couple_prompts.negative_prompt[negative_style]
    request_json["batch_size"] = batch_size
    request_json["n_iter"] = n_iter
    request_json["seed"] = seed

    print(f"generating...\nprompts: {request_json['prompt']}\n negative prompts: {request_json['negative_prompt']}")
    images_base64 = interfaces.txt2imgapi(request_json)
    imgs = [cv_common.base64_to_img(img_base64) for img_base64 in images_base64]
    return imgs


def couple_photo(name1, 
                 name2, 
                 lora1, 
                 lora2, 
                 style,
                 style_image,
                 style_rbg_mask,
                 steps=40, 
                 width=512, 
                 height=512, 
                 batch_size=1, 
                 n_iter=1, 
                 gender1=gender_const.YOUNG_MAN, 
                 gender2=gender_const.GIRL, 
                 order='left_to_right',
                 negative_style="no_glasses", 
                 body_style="upper body", 
                 seed=-1):
    # 根据style生成couple图片
    gender_prompt = GenderPrompt.merge(gender1, gender2, weight=1.5)
    prompt = ",".join([item for item in [gender_prompt, 
                                         name1,
                                         name2, 
                                         couple_prompts.base_prompt.format(body_style), 
                                         couple_prompts.light_prompt, 
                                         couple_prompts.background_prompt[style], 
                                         couple_prompts.clothing_prompt[style],
                                         couple_prompts.hair_prompt[style],
                                         couple_prompts.posing_prompt[style]] if item])
    
    adtailer_face_prompt1 = couple_prompts.adtailer_face_prompt.format(gender1, name1)
    adtailer_face_prompt2 = couple_prompts.adtailer_hand_prompt.format(gender2, name2)
    adtailer_hand_prompt_tmp = couple_prompts.adtailer_hand_prompt

    with open(os.path.join(CUR_DIR, 'sd', 'jsontemp', 'couple.json'), 'r') as f:
        request_json = json.load(f)

    # set base
    request_json["steps"] = steps
    request_json["width"] = width
    request_json["height"] = height
    request_json["prompt"] = prompt
    request_json["negative_prompt"] = girl_prompts.negative_prompt[negative_style]
    request_json["batch_size"] = batch_size
    request_json["n_iter"] = n_iter
    request_json["seed"] = seed

    # set lora
    # 这里要根据style image的男女顺序调整lora_names顺序
    if order == 'left_to_right':
        lora_names = [lora1, lora2]
    elif order == 'right_to_left':
        lora_names = [lora2, lora1]
    else:
        raise ValueError(f'unknown order={order}')

    for i in range(len(lora_names)):
        request_json["alwayson_scripts"]["additional networks for generating"]["args"][i * 4 + 3] = lora_names[i]
    
    # set lora RGB mask, should be an ndarray of shape=[h, w, 3], dtyle=np.uint8
    request_json["alwayson_scripts"]["additional networks for generating"]["args"][22] = style_rbg_mask.tolist()

    # set adtailer
    # 这里要根据style image的男女顺序调整ad_prompt顺序
    if order == 'left_to_right':
        request_json["alwayson_scripts"]["adetailer"]["args"][1]["ad_prompt"] = adtailer_face_prompt1
        request_json["alwayson_scripts"]["adetailer"]["args"][2]["ad_prompt"] = adtailer_face_prompt2
    elif order == 'right_to_left':
        request_json["alwayson_scripts"]["adetailer"]["args"][1]["ad_prompt"] = adtailer_face_prompt2
        request_json["alwayson_scripts"]["adetailer"]["args"][2]["ad_prompt"] = adtailer_face_prompt1
    else:
        raise ValueError(f'unknown order={order}')

    request_json["alwayson_scripts"]["adetailer"]["args"][1]["ad_negative_prompt"] = couple_prompts.adtailer_face_negative_prompt[negative_style]
    request_json["alwayson_scripts"]["adetailer"]["args"][2]["ad_negative_prompt"] = couple_prompts.adtailer_face_negative_prompt[negative_style]

    request_json["alwayson_scripts"]["adetailer"]["args"][3]["ad_prompt"] = adtailer_hand_prompt_tmp
    request_json["alwayson_scripts"]["adetailer"]["args"][3]["ad_negative_prompt"] = girl_prompts.adtailer_hand_negative_prompt

    style_image_base64 = cv_common.img_to_sdwebui_base64(style_image)
    # set controlnet openpose
    request_json["alwayson_scripts"]["ControlNet"]["args"][0]["input_image"] = style_image_base64
    # set controlnet canny
    request_json["alwayson_scripts"]["ControlNet"]["args"][1]["input_image"] = style_image_base64
    
    # start generate
    images_base64 = interfaces.txt2imgapi(request_json)
    # 最后一张图片是controlnet的输入图片，这里去掉
    images_base64 = images_base64[:batch_size]
    imgs = [cv_common.base64_to_img(img_base64) for img_base64 in images_base64]

    return imgs


def gen_couple(num_imgs,
               num_style_imgs,
               name1,
               name2,
               lora1,
               lora2,
               style, 
               save_dir,
               anchor_file_pattern1,
               anchor_file_pattern2,
               face_model_file,
               width=512, 
               height=512, 
               gender1=gender_const.YOUNG_MAN, 
               gender2=gender_const.GIRL, 
               negative_style="no_glasses", 
               body_style="upper body",
               n_iter=10):
    # 双人
    # 1. 生成style图片(ad修复face&hand,20步)，留下good的图片，直到生成完n_iter张图片为止
    # 2. 制作RGBmask
    # 3. 根据style图片，生成n_iter*num_imgs张图片(顺序从前往后取style)，Lora+RGNMask+Openpose+Canny+ADtailer(face&hand-20)
    # 4. 取得分最高的图片

    if not (1 <= num_imgs <= 8):
        raise ValueError('only 1~8 images can be generated. If you want to generate more images, call gen_solo multiple times.')
    
    face_model = face_utils.get_face_model(face_model_file)
    face_anchor1 = face_utils.FaceIdentity(face_model, file_pattern=anchor_file_pattern1)
    face_anchor2 = face_utils.FaceIdentity(face_model, file_pattern=anchor_file_pattern2)

    os.makedirs(good_style_save_dir := os.path.join(save_dir, style, 'style', 'good'), exist_ok=True)
    os.makedirs(bad_style_save_dir := os.path.join(save_dir, style, 'style', 'bad'), exist_ok=True)
    good_style_photos = []
    bad_style_photos = []
    with open(os.path.join(bad_style_save_dir, 'details.txt'), 'w') as f:
        while len(good_style_photos) < num_style_imgs:
            print(f'generating style images({len(good_style_photos)}/{num_style_imgs})...')
            style_imgs = couple_style_photo(style, 
                                            steps=20, 
                                            width=width, 
                                            height=height, 
                                            batch_size=8, 
                                            n_iter=1, 
                                            gender1=gender1, 
                                            gender2=gender2, 
                                            negative_style=negative_style, 
                                            body_style=body_style, 
                                            seed=-1)

            for style_img in style_imgs:
                style_photo = face_utils.PortraitPhoto(img=style_img, expected_face_num=2)
                feedbacks = style_photo.evaluate(body_style=body_style,
                                                 eye_mouth_degree_max=20,
                                                 upper_body_head_ratio_min=0.04,
                                                 upper_body_head_ratio_max=0.3,
                                                 genders_prompts=[gender1, gender2])
                print(feedbacks)
                if set(feedbacks) == {'good'}:
                    # 先保存，再append
                    style_photo.save(os.path.join(good_style_save_dir, f"{len(good_style_photos)}.png"))
                    good_style_photos.append(style_photo)
                    
                else:
                    # 先保存，再append
                    style_photo.save(os.path.join(bad_style_save_dir, f"{len(bad_style_photos)}.png"))
                    f.write(f"{len(bad_style_photos)}.png: {feedbacks}\n")
                    bad_style_photos.append(style_photo)

    print('generating couple images...')
    good_face_photos = []
    bad_face_photos = []
    # 多余的style images直接丢弃，只取前num_style_imgs个
    for ith_iter in range(n_iter):
        style_photo = good_style_photos[ith_iter % len(good_style_photos)]
        # mask总是从左到右呈现R-G-B顺序
        try:
            # 极小概率会出现rgb-mask的数量和预期的人数不相符
            style_rbg_mask = segmentation.segment_person_rgb(style_photo.img, expected_num=2)
        except segmentation.PersonRGBMaskNumberNotMatch as e:
            logging.error(f"rgb mask number not match. error={e}")
            continue
        # 根据生成的style图片，需要知道从左到右每个人的性别，用来设置order这个参数
        # order如果设置为left_to_right的意思是，按照style图片的人头，从左到右替换lora1和lora2
        # 所以要根据style图片的性别，和输入参数中gender1和gender2的性别，对号入座，不能搞错性别
        # 这里知道style_photo的两个gender，并且还需要知道左右关系才行
        face0 = style_photo.face_images[0]
        face1 = style_photo.face_images[1]
        xmin0, _, xmax0, _ = face0.box
        center_x0 = (xmin0 + xmax0) / 2
        xmin1, _, xmax1, _ = face1.box
        center_x1 = (xmin1 + xmax1) / 2
        if center_x0 < center_x1:
            # face0在左，face1在右边
            left_gender = face0.face_attr.gender
            right_gender = face1.face_attr.gender
        else:
            # face0在右，face1在左边
            left_gender = face1.face_attr.gender
            right_gender = face0.face_attr.gender

        lora_gender1 = gender_const.LABEL_MAP[gender1]
        lora_gender2 = gender_const.LABEL_MAP[gender2]

        # 经过前面对good style的筛选，这里的gender数量上能对上，left_gender和right_gender如果是1男1女，那么lora_gender1和lora_gender2也一定是1男1女
        if left_gender == lora_gender1 and right_gender == lora_gender2:
            order = "left_to_right"
        elif left_gender == lora_gender2 and right_gender == lora_gender1:
            order = "right_to_left"
        else:
            raise ValueError(f'impossible: {left_gender=}, {right_gender}, {lora_gender1=}, {lora_gender2=}')

        print(f"generating final images({ith_iter}/{n_iter})...")
        couple_imgs = couple_photo(name1=name1, 
                                   name2=name2, 
                                   lora1=lora1, 
                                   lora2=lora2, 
                                   style=style,
                                   style_image=style_photo.img,
                                   style_rbg_mask=style_rbg_mask,
                                   steps=40, 
                                   width=width, 
                                   height=height, 
                                   batch_size=num_imgs, 
                                   n_iter=1, 
                                   gender1=gender1, 
                                   gender2=gender2, 
                                   order=order,
                                   negative_style="no_glasses", 
                                   body_style="upper body", 
                                   seed=-1)
        for couple_img in couple_imgs:
            face_photo = face_utils.PortraitPhoto(img=couple_img, expected_face_num=2)
            feedbacks = face_photo.evaluate(body_style=body_style,
                                            eye_mouth_degree_max=20,
                                            upper_body_head_ratio_min=0.04,
                                            upper_body_head_ratio_max=0.3,
                                            genders_prompts=[gender1, gender2])
            if set(feedbacks) == {'good'}:
                good_face_photos.append(face_photo)
            else:
                bad_face_photos.append(face_photo)

    os.makedirs(best_save_dir := os.path.join(save_dir, style, 'final', 'best'), exist_ok=True)
    os.makedirs(dissim_save_dir := os.path.join(save_dir, style, 'final', 'dissim'), exist_ok=True)
    os.makedirs(bad_save_dir := os.path.join(save_dir, style, 'final', 'bad'), exist_ok=True)
    good_face_photos_scores = []
    for face_photo in good_face_photos:
        sim1 = face_anchor1.similarity(face_photo.get_face_by_mask(segmentation.rgb_mask_choose(style_rbg_mask, 0)))
        sim2 = face_anchor2.similarity(face_photo.get_face_by_mask(segmentation.rgb_mask_choose(style_rbg_mask, 1)))
        # 1/5的标准差为惩罚项，相似度不能差太远
        score = (sim1 + sim2) * 0.5 - 0.2 * np.std([sim1, sim2])
        good_face_photos_scores.append(score)
    
    sorted_tuples = sorted(zip(good_face_photos_scores, good_face_photos), key=lambda x: x[0], reverse=True)
    good_face_photos_scores, good_face_photos = zip(*sorted_tuples)

    for i in range(len(good_face_photos)):
        good_photo = good_face_photos[i]
        good_photo_score = good_face_photos_scores[i]
        if i < num_imgs:
            good_photo.save(os.path.join(best_save_dir, f"{i}_{good_photo_score: .3f}.png"))
        else:
            good_photo.save(os.path.join(dissim_save_dir, f"{i}_{good_photo_score: .3f}.png"))
    
    with open(os.path.join(bad_save_dir, 'details.txt'), 'w') as f:
        for i, bad_photo in enumerate(bad_face_photos):
            bad_photo.save(os.path.join(bad_save_dir, f"{i}.png"))
            f.write(f"{i}.png: {bad_photo.feedbacks}\n")


def gen_girl_pipeline():
    gen_girl_solo(
        num_imgs=8,
        n_iter=10,
        name='jiangnana',
        lora='jn_retouch_aug3_lora_sd15_v7(6d0770d74d49)',
        style='id_photo',
        width=512,
        height=640,
        save_dir=f'/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/pipeline_saved/id_photo/', 
        anchor_file_pattern="/home/zzy2/workspace/sd-webui-prettyu/prettyu/lora/data/raw/jiangnana/*.JPG", 
        face_model_file="/home/zzy2/workspace/insightface/model_zoo/webface_r50_pfc.onnx",
        negative_style='no_glasses', 
        body_style='upper body')


def gen_couple_pipeline():
    for style in couple_prompts.background_prompt.keys():
        gen_couple(num_imgs=8,
                num_style_imgs=8,
                name1='zhangzhenyu',
                name2='jiangnana',
                lora1='zzy_retouch_aug3_lora_sd15_v7(bcdb0302323a)',
                lora2='jn_retouch_aug3_lora_sd15_v7(6d0770d74d49)',
                style=style, 
                save_dir='/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/couple_saved', 
                anchor_file_pattern1='/home/zzy2/workspace/sd-webui-prettyu/prettyu/lora/data/raw/zhangzhenyu/*.JPG',
                anchor_file_pattern2='/home/zzy2/workspace/sd-webui-prettyu/prettyu/lora/data/raw/jiangnana/*.JPG',
                face_model_file="/home/zzy2/workspace/insightface/model_zoo/webface_r50_pfc.onnx",
                gender1=gender_const.YOUNG_MAN, 
                gender2=gender_const.GIRL, 
                negative_style="no_glasses", 
                body_style="upper body",
                n_iter=8)
