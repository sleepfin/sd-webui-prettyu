import os
import json
import logging
import numpy as np

from contextlib import closing
import modules.scripts
from modules import processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts, cmd_opts
import modules.shared as shared

from sd.sd_utils import init_default_script_args
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


def solo_photo_api(name, 
                    lora, 
                    style=None,
                    prompt="",
                    negative_prompt="",
                    steps=40, 
                    width=512, 
                    height=512, 
                    batch_size=1,
                    n_iter=1, 
                    gender=None, 
                    negative_style="no_glasses", 
                    body_style="upper body", 
                    seed=-1):
    if gender is None:
        raise ValueError(f'invalid gender={gender}')
    
    if style is not None:
        prompt = ",".join([item for item in [gender, 
                                             name, 
                                             girl_prompts.base_prompt.format(body_style), 
                                             girl_prompts.light_prompt, 
                                             girl_prompts.background_prompt[style], 
                                             girl_prompts.clothing_prompt[style],
                                             girl_prompts.hair_prompt[style],
                                             girl_prompts.posing_prompt[style]] if item])
        negative_prompt = girl_prompts.negative_prompt[negative_style]

    adtailer_face_prompt_tmp = girl_prompts.adtailer_face_prompt.format(gender)
    adtailer_hand_prompt_tmp = girl_prompts.adtailer_hand_prompt.format(gender)

    override_settings_texts = []
    override_settings = create_override_settings_dict(override_settings_texts)

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[],
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        sampler_name="DPM++ SDE Karras",
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=7,
        width=width,
        height=height,
        restore_faces=False,
        tiling=False,
        enable_hr=False,
        denoising_strength=None,
        hr_scale=2,
        hr_upscaler='Latent',
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        hr_sampler_name=None,
        hr_prompt='',
        hr_negative_prompt='',
        override_settings=override_settings,
        do_not_save_grid=True,
    )

    p.scripts = modules.scripts.scripts_txt2img
    args = init_default_script_args(p.scripts)

    for _scripts in p.scripts.alwayson_scripts:
        if str(type(_scripts)) == "<class '!adetailer.py.AfterDetailerScript'>":
            print("-------------ad--------------")
            print(args[_scripts.args_from: _scripts.args_to])
            print("-------------ad--------------") 
            args[_scripts.args_from] = True
            if isinstance(args[_scripts.args_from + 1], bool):
                args[_scripts.args_from + 1] = True
                offset = 2
            else:
                offset = 1

            # face adtailer
            args[_scripts.args_from + offset]['ad_mask_k_largest'] = 1
            args[_scripts.args_from + offset]['ad_model'] = 'face_yolov8n.pt'
            args[_scripts.args_from + offset]['ad_negative_prompt'] = girl_prompts.adtailer_face_negative_prompt[negative_style]
            args[_scripts.args_from + offset]['ad_noise_multiplier'] = 0.5
            args[_scripts.args_from + offset]['ad_prompt'] = adtailer_face_prompt_tmp
            args[_scripts.args_from + offset]['ad_sampler'] = 'DPM++ 2M SDE Karras'
            args[_scripts.args_from + offset]['ad_steps'] = 80
            args[_scripts.args_from + offset]['ad_use_noise_multiplier'] = True
            args[_scripts.args_from + offset]['ad_use_sampler'] = True
            args[_scripts.args_from + offset]['ad_use_steps'] = True
            # hand adtailer
            args[_scripts.args_from + offset + 1]['ad_mask_k_largest'] = 2
            args[_scripts.args_from + offset + 1]['ad_model'] = 'hand_yolov8n.pt'
            args[_scripts.args_from + offset + 1]['ad_negative_prompt'] = girl_prompts.adtailer_hand_negative_prompt
            args[_scripts.args_from + offset + 1]['ad_noise_multiplier'] = 0.5
            args[_scripts.args_from + offset + 1]['ad_prompt'] = adtailer_hand_prompt_tmp
            args[_scripts.args_from + offset + 1]['ad_sampler'] = 'DPM++ 2M SDE Karras'
            args[_scripts.args_from + offset + 1]['ad_steps'] = 80
            args[_scripts.args_from + offset + 1]['ad_use_noise_multiplier'] = True
            args[_scripts.args_from + offset + 1]['ad_use_sampler'] = True
            args[_scripts.args_from + offset + 1]['ad_use_steps'] = True
        elif str(type(_scripts)) == "<class 'additional_networks.py.Script'>":
            print("-------------lora--------------")
            print(args[_scripts.args_from: _scripts.args_to])
            print("-------------lora--------------")
            args[_scripts.args_from] = True
            args[_scripts.args_from + 2] = "LoRA"
            args[_scripts.args_from + 3] = lora
            args[_scripts.args_from + 4] = 1.0
            args[_scripts.args_from + 5] = 1.0
        else:
            pass

    p.script_args = args

    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed


def tile_fix(input_image, pre_seed, lora):
    prompt = "masterpiece,best quality,high quality,highres,ultra-detailed"
    negative_prompt = "NSFW,worst quality,low quality,normal quality"

    override_settings_texts = []
    override_settings = create_override_settings_dict(override_settings_texts)

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[],
        negative_prompt=negative_prompt,
        seed=pre_seed,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        sampler_name="DPM++ SDE Karras",
        batch_size=1,
        n_iter=1,
        steps=60,
        cfg_scale=7,
        width=width,
        height=height,
        restore_faces=False,
        tiling=False,
        enable_hr=True,
        denoising_strength=0.7,
        hr_scale=2,
        hr_upscaler='R-ESRGAN 4x+',
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        hr_sampler_name=None,
        hr_prompt='',
        hr_negative_prompt='',
        override_settings=override_settings,
        do_not_save_grid=True,
    )

    p.scripts = modules.scripts.scripts_txt2img
    args = init_default_script_args(p.scripts)


    print("==============alwayson_scripts==============")
    for _scripts in p.scripts.alwayson_scripts:
        print(f"=============={_scripts}:{_scripts.args_from}->{_scripts.args_to}==============")
        print(args[_scripts.args_from: _scripts.args_to])
    
    print("==============alwayson_scripts==============")


    for _scripts in p.scripts.alwayson_scripts:
        if str(type(_scripts)) == "<class 'additional_networks.py.Script'>":
            args[_scripts.args_from] = True
            args[_scripts.args_from + 2] = "LoRA"
            args[_scripts.args_from + 3] = lora
            args[_scripts.args_from + 4] = 1.0
            args[_scripts.args_from + 5] = 1.0
            print("-------------lora--------------")
            print(args[_scripts.args_from: _scripts.args_to])
            print("-------------lora--------------")
        elif str(type(_scripts)) == "<class 'controlnet.py.Script'>":
            # args[i] is an instance of `UiControlNetUnit`
            args[_scripts.args_from].enabled = True
            args[_scripts.args_from].module = "tile_resample"
            args[_scripts.args_from].model = "control_v11f1e_sd15_tile [a371b31b]"
            args[_scripts.args_from].image = input_image
            args[_scripts.args_from].pixel_perfect = True
            print("-------------controlnet--------------")
            print(args[_scripts.args_from: _scripts.args_to])
            print("-------------controlnet--------------")
        else:
            pass

    p.script_args = args

    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed


def gen_solo_api(num_imgs, 
                 name, 
                 lora, 
                 anchor_file_pattern, 
                 style=None,
                 prompt="",
                 negative_prompt="",
                 width=512, 
                 height=512, 
                 gender=gender_const.GIRL, 
                 negative_style='no_glasses', 
                 body_style='upper body', 
                 n_iter=10,
                 high_res=False):
    # 单人girl
    # 1. 生成指定Lora的图片（Lora+ADtailer_face&hand）
    # 2. 后处理评价图片，从num_imgs*n_iter张图片中保留得分最高的num_imgs张图片
    if not (1 <= num_imgs <= 8):
        raise ValueError('only 1~8 images can be generated. If you want to generate more images, call gen_solo multiple times.')
    
    face_model = face_utils.get_face_model()
    face_anchor = face_utils.FaceIdentity(face_model, file_pattern=anchor_file_pattern)

    processed = solo_photo_api(name=name, 
                               gender=gender, 
                               lora=lora, 
                               style=style,
                               prompt=prompt,
                               negative_prompt=negative_prompt,
                               width=width,
                               height=height,
                               batch_size=num_imgs, 
                               n_iter=n_iter, 
                               negative_style=negative_style, 
                               body_style=body_style, 
                               seed=-1)

    good_face_photos = []
    good_face_seeds = []
    bad_face_photos = []
    for img, seed in zip(processed.images, processed.all_seeds):
        img = np.asarray(img)
        face_photo = face_utils.PortraitPhoto(img=img, expected_face_num=1)
        face_photo.evaluate(body_style=body_style)
        if face_photo.main_feedback == 'good':
            good_face_photos.append(face_photo)
            good_face_seeds.append(seed)
        else:
            bad_face_photos.append(face_photo)
    
    if high_res:
        print('upscale to high resolution')
        fixed_good_face_photos = []
        for good_photo, seed in zip(good_face_photos, good_face_seeds):
            fixed_processed = tile_fix(np.asarray(good_photo.img), seed, lora)
            fixed_img = np.asarray(fixed_processed.images[0])
            fixed_photo = face_utils.PortraitPhoto(img=fixed_img, expected_face_num=1)
            fixed_good_face_photos.append(fixed_photo)
        
        good_face_photos = fixed_good_face_photos

    # 这里good face必然只有一个face
    good_face_photos_scores = [face_anchor.similarity(pht.face_images[0]) for pht in good_face_photos]

    if len(good_face_photos) > 0:
        sorted_tuples = sorted(zip(good_face_photos_scores, good_face_photos), key=lambda x: x[0], reverse=True)
        good_face_photos_scores, good_face_photos = zip(*sorted_tuples)
        good_num = min(num_imgs, len(good_face_photos))
        best_face_photos = good_face_photos[:good_num]
        dissim_face_photos = good_face_photos[good_num:]
    else:
        # all photos are bad
        best_face_photos, dissim_face_photos = [], []

    return best_face_photos, dissim_face_photos, bad_face_photos
