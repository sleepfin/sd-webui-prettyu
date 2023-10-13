import requests
import logging
import copy
import json


SDWEB_API_URL = "http://127.0.0.1:7861"
MODELS = {
    'sdvn6': "sdvn6Realxl_detailface.safetensors [777f31751a]",
    'majic': "majicmixRealistic_betterV2V25.safetensors [d7e2ac2f4a]"
}
TXT2IMG_BODY = {
  "enable_hr": False,
  "denoising_strength": 0,
  "firstphase_width": 0,
  "firstphase_height": 0,
  "hr_scale": 2,
  "hr_upscaler": "",
  "hr_second_pass_steps": 0,
  "hr_resize_x": 0,
  "hr_resize_y": 0,
  "hr_sampler_name": "",
  "hr_prompt": "",
  "hr_negative_prompt": "",
  "prompt": "",
  "styles": [],
  "seed": -1,
  "subseed": -1,
  "subseed_strength": 0,
  "seed_resize_from_h": -1,
  "seed_resize_from_w": -1,
  "sampler_name": "",
  "batch_size": 1,
  "n_iter": 1,
  "steps": 20,
  "cfg_scale": 7,
  "width": 512,
  "height": 512,
  "restore_faces": False,
  "tiling": False,
  "do_not_save_samples": False,
  "do_not_save_grid": False,
  "negative_prompt": "",
  "eta": 0,
  "s_min_uncond": 0,
  "s_churn": 0,
  "s_tmax": 0,
  "s_tmin": 0,
  "s_noise": 1,
  "override_settings": {},
  "override_settings_restore_afterwards": True,
  "script_args": [],
  "sampler_index": "Euler",
  "script_name": "",
  "send_images": True,
  "save_images": False,
  "alwayson_scripts": {}
}

OPTION_BODY={
  "samples_save": False,
  "samples_format": "png",
  "samples_filename_pattern": "",
  "save_images_add_number": True,
  "grid_save": False,
  "grid_format": "png",
  "grid_extended_filename": False,
  "grid_only_if_multiple": True,
  "grid_prevent_empty_spots": False,
  "grid_zip_filename_pattern": "",
  "n_rows": -1,
  "font": "",
  "grid_text_active_color": "#000000",
  "grid_text_inactive_color": "#999999",
  "grid_background_color": "#ffffff",
  "enable_pnginfo": True,
  "save_txt": False,
  "save_images_before_face_restoration": False,
  "save_images_before_highres_fix": False,
  "save_images_before_color_correction": False,
  "save_mask": False,
  "save_mask_composite": False,
  "jpeg_quality": 80,
  "webp_lossless": False,
  "export_for_4chan": True,
  "img_downscale_threshold": 4,
  "target_side_length": 4000,
  "img_max_size_mp": 200,
  "use_original_name_batch": True,
  "use_upscaler_name_as_suffix": False,
  "save_selected_only": True,
  "save_init_img": False,
  "temp_dir": "",
  "clean_temp_dir_at_start": False,
  "outdir_samples": "",
  "outdir_txt2img_samples": "outputs/txt2img-images",
  "outdir_img2img_samples": "outputs/img2img-images",
  "outdir_extras_samples": "outputs/extras-images",
  "outdir_grids": "",
  "outdir_txt2img_grids": "outputs/txt2img-grids",
  "outdir_img2img_grids": "outputs/img2img-grids",
  "outdir_save": "log/images",
  "outdir_init_images": "outputs/init-images",
  "save_to_dirs": True,
  "grid_save_to_dirs": True,
  "use_save_to_dirs_for_ui": False,
  "directories_filename_pattern": "[date]",
  "directories_max_prompt_words": 8,
  "ESRGAN_tile": 192,
  "ESRGAN_tile_overlap": 8,
  "realesrgan_enabled_models": [
    "R-ESRGAN 4x+",
    "R-ESRGAN 4x+ Anime6B"
  ],
  "upscaler_for_img2img": None,
  "face_restoration_model": "CodeFormer",
  "code_former_weight": 0.5,
  "face_restoration_unload": False,
  "show_warnings": False,
  "memmon_poll_rate": 8,
  "samples_log_stdout": False,
  "multiple_tqdm": True,
  "print_hypernet_extra": False,
  "list_hidden_files": True,
  "disable_mmap_load_safetensors": False,
  "unload_models_when_training": False,
  "pin_memory": False,
  "save_optimizer_state": False,
  "save_training_settings_to_txt": True,
  "dataset_filename_word_regex": "",
  "dataset_filename_join_string": " ",
  "training_image_repeats_per_epoch": 1,
  "training_write_csv_every": 500,
  "training_xattention_optimizations": False,
  "training_enable_tensorboard": False,
  "training_tensorboard_save_images": False,
  "training_tensorboard_flush_every": 120,
  "sd_model_checkpoint": "sdvn6Realxl_detailface.safetensors [777f31751a]",
  "sd_checkpoint_cache": 0,
  "sd_vae_checkpoint_cache": 0,
  "sd_vae": "Automatic",
  "sd_vae_as_default": True,
  "sd_unet": "Automatic",
  "inpainting_mask_weight": 1,
  "initial_noise_multiplier": 1,
  "img2img_color_correction": False,
  "img2img_fix_steps": False,
  "img2img_background_color": "#ffffff",
  "enable_quantization": False,
  "enable_emphasis": True,
  "enable_batch_seeds": True,
  "comma_padding_backtrack": 20,
  "CLIP_stop_at_last_layers": 1,
  "upcast_attn": False,
  "auto_vae_precision": True,
  "randn_source": "GPU",
  "sdxl_crop_top": 0,
  "sdxl_crop_left": 0,
  "sdxl_refiner_low_aesthetic_score": 2.5,
  "sdxl_refiner_high_aesthetic_score": 6,
  "cross_attention_optimization": "Automatic",
  "s_min_uncond": 0,
  "token_merging_ratio": 0,
  "token_merging_ratio_img2img": 0,
  "token_merging_ratio_hr": 0,
  "pad_cond_uncond": False,
  "experimental_persistent_cond_cache": False,
  "use_old_emphasis_implementation": False,
  "use_old_karras_scheduler_sigmas": False,
  "no_dpmpp_sde_batch_determinism": False,
  "use_old_hires_fix_width_height": False,
  "dont_fix_second_order_samplers_schedule": False,
  "hires_fix_use_firstpass_conds": False,
  "interrogate_keep_models_in_memory": False,
  "interrogate_return_ranks": False,
  "interrogate_clip_num_beams": 1,
  "interrogate_clip_min_length": 24,
  "interrogate_clip_max_length": 48,
  "interrogate_clip_dict_limit": 1500,
  "interrogate_clip_skip_categories": [],
  "interrogate_deepbooru_score_threshold": 0.5,
  "deepbooru_sort_alpha": True,
  "deepbooru_use_spaces": True,
  "deepbooru_escape": True,
  "deepbooru_filter_tags": "",
  "extra_networks_show_hidden_directories": True,
  "extra_networks_hidden_models": "When searched",
  "extra_networks_default_multiplier": 1,
  "extra_networks_card_width": 0,
  "extra_networks_card_height": 0,
  "extra_networks_card_text_scale": 1,
  "extra_networks_card_show_desc": True,
  "extra_networks_add_text_separator": " ",
  "ui_extra_networks_tab_reorder": "",
  "textual_inversion_print_at_load": False,
  "textual_inversion_add_hashes_to_infotext": True,
  "sd_hypernetwork": "None",
  "localization": "None",
  "gradio_theme": "Default",
  "img2img_editor_height": 720,
  "return_grid": True,
  "return_mask": False,
  "return_mask_composite": False,
  "do_not_show_images": False,
  "send_seed": True,
  "send_size": True,
  "js_modal_lightbox": True,
  "js_modal_lightbox_initially_zoomed": True,
  "js_modal_lightbox_gamepad": False,
  "js_modal_lightbox_gamepad_repeat": 250,
  "show_progress_in_title": True,
  "samplers_in_dropdown": True,
  "dimensions_and_batch_together": True,
  "keyedit_precision_attention": 0.1,
  "keyedit_precision_extra": 0.05,
  "keyedit_delimiters": ".,\\/!?%^*;:{}=`~()",
  "keyedit_move": True,
  "quicksettings_list": [
    "sd_model_checkpoint"
  ],
  "ui_tab_order": [],
  "hidden_tabs": [],
  "ui_reorder_list": [],
  "hires_fix_show_sampler": False,
  "hires_fix_show_prompts": False,
  "disable_token_counters": False,
  "add_model_hash_to_info": True,
  "add_model_name_to_info": True,
  "add_user_name_to_info": False,
  "add_version_to_infotext": True,
  "disable_weights_auto_swap": True,
  "infotext_styles": "Apply if any",
  "show_progressbar": True,
  "live_previews_enable": True,
  "live_previews_image_format": "png",
  "show_progress_grid": True,
  "show_progress_every_n_steps": 10,
  "show_progress_type": "Approx NN",
  "live_preview_content": "Prompt",
  "live_preview_refresh_period": 1000,
  "hide_samplers": [],
  "eta_ddim": 0,
  "eta_ancestral": 1,
  "ddim_discretize": "uniform",
  "s_churn": 0,
  "s_tmin": 0,
  "s_noise": 1,
  "k_sched_type": "Automatic",
  "sigma_min": 0,
  "sigma_max": 0,
  "rho": 0,
  "eta_noise_seed_delta": 0,
  "always_discard_next_to_last_sigma": False,
  "uni_pc_variant": "bh1",
  "uni_pc_skip_type": "time_uniform",
  "uni_pc_order": 3,
  "uni_pc_lower_order_final": True,
  "postprocessing_enable_in_main_ui": [],
  "postprocessing_operation_order": [],
  "upscaling_max_images_in_cache": 5,
  "disabled_extensions": [
    "LDSR",
    "ScuNET",
    "SwinIR",
    "canvas-zoom-and-pan",
    "extra-options-section",
    "mobile",
    "prompt-bracket-checker"
  ],
  "disable_all_extensions": "none",
  "restore_config_state_file": "",
  "sd_checkpoint_hash": "777f31751ae6f9bddf9744a2c644d8fbf2eb94384d0dcd5377ac56afc079aacd",
  "sd_lora": "None",
  "lora_preferred_name": "Alias from file",
  "lora_add_hashes_to_infotext": True,
  "lora_show_all": False,
  "lora_hide_unknown_for_versions": [],
  "lora_functional": False
}


def switch_model(model_name):    
    option_payload = {
        "sd_model_checkpoint": MODELS[model_name],
    }
    response = requests.post(url=f'{SDWEB_API_URL}/sdapi/v1/options', json=option_payload)
    if response.status_code < 300:
        return True
    else:
        logging.error(response.text)
        return False


def set_clip_skip(clip_skip):
    option_payload = {
        "CLIP_stop_at_last_layers": clip_skip,
    }
    response = requests.post(url=f'{SDWEB_API_URL}/sdapi/v1/options', json=option_payload)
    if response.status_code < 300:
        return True
    else:
        logging.error(response.text)
        return False


def txt2img(prompt="", negative_prompt="", batch_size=1, width=512, height=512, steps=20, cfg_scale=7, sampler='DPM++ 2M SDE Karras', seed=-1):
    payload = copy.deepcopy(TXT2IMG_BODY)
    payload['prompt'] = prompt
    payload['negative_prompt'] = negative_prompt
    payload['batch_size'] = batch_size
    payload['sampler_index'] = sampler
    payload['cfg_scale'] = cfg_scale
    payload['steps'] = steps
    payload['width'] = width
    payload['height'] = height
    payload['seed'] = seed
    return txt2imgapi(payload)


def txt2imgapi(payload):
    response = requests.post(url=f'{SDWEB_API_URL}/sdapi/v1/txt2img', json=payload)
    if response.status_code < 300:
        resp_json = json.loads(response.text)
        images_base64 = resp_json['images']
        return images_base64
    else:
        logging.error(response.text)
        return None


def progress():
    response = requests.get(url=f'{SDWEB_API_URL}/sdapi/v1/progress')
    if response.status_code < 300:
        resp_json = json.loads(response.text)
        return resp_json['progress']
    else:
        logging.error(response.text)
        return None