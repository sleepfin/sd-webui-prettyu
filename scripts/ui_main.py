import os
import sys
import glob
import shutil
import psutil
import gradio as gr

from modules import sd_models
from modules import script_callbacks, shared

from scripts.ph_config import models_path
from scripts.train_pipeline import train_sd15_lora_pipeline, DataFileAlreadyExistsError
from scripts.ph_utils import maybe_import, hash_lora_file, custom_embeddings_dir


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
GENDER_CHOICE = ["1girl", "a little girl", "1woman", "1boy", "a young man", "1man", "1bady"]
gradio_compat = True

maybe_import(ROOT_DIR, "prettyu")
girl_prompts = maybe_import(os.path.join(ROOT_DIR, "prettyu", "prompts"), "girl_prompts")
ai_photo_pipeline_api = maybe_import(os.path.join(ROOT_DIR, "prettyu"), "ai_photo_pipeline_api")
additional_networks_dir = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 'sd-webui-additional-networks')
adtailer_dir = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 'adetailer')
controlnet_dir = os.path.join(os.path.dirname(os.path.dirname(CUR_DIR)), 'sd-webui-controlnet')
proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'prettyu')
lora_data_dir = os.path.join(proj_dir, 'lora', 'data')

try:
    from distutils.version import LooseVersion
    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", 
                         elem_classes=kwargs.pop('elem_classes', []) + ["cnet-toolbutton"], 
                         **kwargs)

    def get_block_name(self):
        return "button"


def create_ui_checkpoint_selector():
    def _get_latest_checkpoints():
        return filter(lambda fp: fp.endswith(("pth", "safetensors", "ckpt")), os.listdir(os.path.join(models_path, "Stable-diffusion")))

    def _refresh_checkpoints():
        checkpoints = _get_latest_checkpoints()
        return gr.update(choices=checkpoints)
    
    with gr.Row():
        checkpoints = _get_latest_checkpoints()
        checkpoint_selector = gr.Dropdown(value="majicmixRealistic_betterV2V25.safetensors",
                                          choices=checkpoints, 
                                          label="[Required] Stable Diffusion Pretrained Model(选择基础SD模型). majicmixRealistic is recommended(推荐使用majicmixRealistic): https://civitai.com/models/43331/majicmix-realistic", 
                                          visible=True)

        checkpoint_refresher = ToolButton(value="\U0001f504")
        checkpoint_refresher.click(
            fn=_refresh_checkpoints,
            inputs=[],
            outputs=[checkpoint_selector]
        )

    return checkpoint_selector, checkpoint_refresher


def check_train_info(checkpoint_info, unique_name, gender, train_images):
    if not (os.path.exists(additional_networks_dir) and os.path.exists(os.path.join(additional_networks_dir, 'scripts', 'additional_networks.py'))):
        return False, gr.update(value=[("sd-webui-additional-networks is not installed. please install this plugin first.", "Error")], visible=True)

    if not (os.path.exists(controlnet_dir) and os.path.exists(os.path.join(controlnet_dir, 'scripts', 'controlnet.py'))):
        return False, gr.update(value=[("controlnet is not installed. please install this plugin first.", "Error")], visible=True)
    
    if not (os.path.exists(adtailer_dir) and os.path.exists(os.path.join(adtailer_dir, 'scripts', '!adetailer.py'))):
        return False, gr.update(value=[("adetailer is not installed. please install this plugin first.", "Error")], visible=True)

    if not checkpoint_info:
        return False, gr.update(value=[("Stable Diffusion Pretrained Model is not seleceted", "Error")], visible=True)

    if not unique_name:
        return False, gr.update(value=[("Unique Name is empty", "Error")], visible=True)

    if (not gender) or (gender not in GENDER_CHOICE):
        return False, gr.update(value=[("Gender is empty or not valid", "Error")], visible=True)

    if (not train_images) or (len(train_images) < 1):
        return False, gr.update(value=[("Not enough photos", "Error")], visible=True)
    
    return True, None


def check_and_update_status(unique_name, gender, train_images, progress=gr.Progress()):
    checkpoint_info = sd_models.select_checkpoint()
    is_valid, gr_error = check_train_info(checkpoint_info, unique_name, gender, train_images)
    if not is_valid:
        return gr_error, gr.update(interactive=True)
    
    return gr.update(value=[("Start to training", "Running")], visible=True), gr.update(interactive=False)


def start_train(unique_name, gender, train_images, progress=gr.Progress()):
    checkpoint_info = sd_models.select_checkpoint()
    is_valid, gr_error = check_train_info(checkpoint_info, unique_name, gender, train_images)
    if not is_valid:
        return gr_error, gr.update(interactive=True)

    try:
        train_sd15_lora_pipeline(checkpoint_info.filename, 
                                 unique_name, 
                                 gender, 
                                 train_images, 
                                 train_batch_size=shared.opts.train_batch_size, 
                                 xformers=shared.opts.xformers, 
                                 mixed_precision=shared.opts.mixed_precision,
                                 cache_latents=shared.opts.cache_latents,
                                 res=shared.opts.train_image_size,
                                 gradient_checkpointing=shared.opts.gradient_checkpointing,
                                 max_data_loader_n_workers=shared.opts.max_data_loader_n_workers,
                                 progress=progress)
    except DataFileAlreadyExistsError:
        return gr.update(value=[(f"Digital clone name={unique_name} already exists. You can remove it at `Generate Photos` tab.", "Error")], visible=True), gr.update(interactive=True)

    return gr.update(value=[("End of training", "Success")], visible=True), gr.update(interactive=True)


def stop_train(status_text):
    parent = psutil.Process(os.getpid())
    # for child in parent.children(recursive=True):
    #     print(child.cmdline())

    stop_success = False
    for child in parent.children(recursive=False):
        cmdline = child.cmdline()
        if ("prepare_lora_data.py" in " ".join(cmdline)) or ("train_sd15_lora.py" in " ".join(cmdline)):
            # kill grandchild
            for grandchild in child.children(recursive=True):
                if grandchild.pid != child.pid:
                    try:
                        print(f'[Interrupt] killing pid={grandchild.pid}')
                        grandchild.kill()
                    except psutil.NoSuchProcess:
                        pass
            # kill child
            try:
                print(f'[Interrupt] killing pid={child.pid}')
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
            stop_success = True
    
    if stop_success:
        return gr.update(value=[("Training is interrupted", "Error")], visible=True), gr.update(interactive=True)
    else:
        return gr.update(value=status_text), gr.update(interactive=True)


def check_upload_files(training_images):
    if len(training_images) > 0:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)


def clear_upload_files():
    return gr.update(interactive=False)


def get_lora_model_list():
    model_list = []
    for lock_file_path in glob.glob(os.path.join(lora_data_dir, "*.lock")):
        lock_name = os.path.basename(lock_file_path)
        unique_name = lock_name[:lock_name.rfind('(')]
        hash_id = lock_name[lock_name.rfind('(') + 1: lock_name.rfind(')')]
        model_list.append(f'{unique_name}({hash_id})')
    
    return model_list


def update_lora_model_list():
    model_list = get_lora_model_list()
    # 从sys.modules中找到additional-networks的model_util模块，并更新其中的lora_models这个全局变量
    if 'scripts.model_util' in sys.modules:
        addnet_model_util = sys.modules['scripts.model_util']
        addnet_model_util.update_models()
    else:
        raise ModuleNotFoundError('additional-networks is not installed or imported.')

    return gr.update(choices=model_list)


def remove_selected_lora_model(selected_models):
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'prettyu')
    lora_data_dir = os.path.join(proj_dir, 'lora', 'data')
    lora_model_dir = os.path.join(proj_dir, 'lora', 'models')

    for model_name in selected_models:
        print(f'removing lora {model_name}')
        hash_id = model_name[model_name.rfind('(') + 1: model_name.rfind(')')]
        raw_data_dir = os.path.join(lora_data_dir, 'raw', hash_id)
        print(f'removing lora raw data {raw_data_dir}')
        if os.path.exists(raw_data_dir):
            shutil.rmtree(raw_data_dir)

        train_data_dir = os.path.join(lora_data_dir, f'train_{hash_id}')
        print(f'removing lora train data {raw_data_dir}')
        if os.path.exists(train_data_dir):
            shutil.rmtree(train_data_dir)

        model_link = os.path.join(additional_networks_dir, 'models', 'lora', f"{hash_id}.safetensors")
        print(f'removing symbol link {model_link}')
        if os.path.exists(model_link):
            os.unlink(model_link)

        model_data_dir = os.path.join(lora_model_dir, f'sd15_{hash_id}')
        print(f'removing lora model {raw_data_dir}')
        if os.path.exists(model_data_dir):
            shutil.rmtree(model_data_dir)

        lock_file_name = os.path.join(lora_data_dir, model_name + '.lock')
        print(f'removing lock file {lock_file_name}')
        if os.path.exists(lock_file_name):
            os.remove(lock_file_name)

    return update_lora_model_list()


def on_style_change(style_option):
    if style_option == "Builtin (预设)":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif style_option == "Custom (自定义)":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    else:
        return


def start_generation(lora_model_list, 
                     ui_style_option,
                     builtin_style_choice,
                     custom_style_prompt,
                     custom_style_negative_prompt,
                     resolution, 
                     batch_size):
    checkpoint_info = sd_models.select_checkpoint()
    if not checkpoint_info:
        raise ValueError('pretrained model is None')
        
    if len(lora_model_list) < 1:
        raise ValueError('One digital clone should be selected')
    if len(lora_model_list) > 1:
        raise ValueError('Only one digital clone can be selected')
    
    lock_name = lora_model_list[0]
    lock_file_path = os.path.join(lora_data_dir, f"{lock_name}.lock")
    with open(lock_file_path, 'r') as f:
        gender = f.read()
    
    if gender not in GENDER_CHOICE:
        raise ValueError(f'unknown gender={gender}')

    lora_hash_id = lock_name[lock_name.rfind('(') + 1: lock_name.rfind(')')]
    if resolution == '512x512':
        high_res = False
    elif resolution == '1024x1024':
        high_res = True
    else:
        raise ValueError('resolution invalid')
    
    lora_model_file = os.path.join(additional_networks_dir, 'models', 'lora', f"{lora_hash_id}.safetensors")
    lora_hash = hash_lora_file(lora_model_file)
    lora_name = f"{lora_hash_id}({lora_hash})"

    if ui_style_option == "Builtin (预设)":
        style_option = builtin_style_choice
    elif ui_style_option == "Custom (自定义)":
        style_option = None
    else:
        raise ValueError(f'unknown ui_style_option={ui_style_option}')

    with custom_embeddings_dir():
        good_photos, dissim_photos, bad_photots = ai_photo_pipeline_api.gen_solo_api(
            batch_size, 
            name=lora_hash_id, 
            lora=lora_name, 
            anchor_file_pattern=os.path.join(lora_data_dir, 'raw', lora_hash_id, '*.*'), 
            style=style_option,
            prompt=custom_style_prompt,
            negative_prompt=custom_style_negative_prompt,
            width=512, 
            height=512, 
            gender=gender, 
            negative_style=shared.opts.negative_style, 
            body_style=shared.opts.body_style, 
            n_iter=shared.opts.redundancy,
            high_res=high_res)
    
    return gr.update(value=[p.img for p in good_photos]), gr.update(value=[p.img for p in dissim_photos]), gr.update(value=[p.img for p in bad_photots])


def init_ui():
    with gr.Blocks(analytics_enabled=False) as ui_tabs:
        with gr.TabItem('Train Lora (训练模型)'):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(value="## Upload your photos (20-35 recommended)")
                    gr.Markdown(value="Single person, half-body (单人半身照)<br>Try not to tilt your head (尽量不歪头)<br>Try not to show your teeth or open your mouth (尽量不露牙/不张嘴)<br>Better without glasses (最好不带眼睛)<br>If it is taken by the front camera of your phone, please look at the camera instead of the screen (如果是手机前置摄像头拍摄，请看镜头，不要看手机屏幕)<br>Simle softly (可以适当微笑)")
                    ui_training_images = gr.File(file_count='multiple', file_types=['image'], show_label=False)

                with gr.Column():
                    gr.Markdown(value="## Training Settings")
                    ### Use Global Checkpoint Setting
                    ui_unique_name = gr.Textbox(label="[Required] Your Unique Name(English)  [必须] 输入英文名或姓名拼音", value="")
                    ui_gender = gr.Dropdown(label="[Required] Choose the most properate gender description  [必须] 选择一个最合适的性别描述", value="",choices=GENDER_CHOICE)
                    with gr.Row():
                        ui_start_train_button = gr.Button('Start Training', variant='primary', interactive=False)
                        ui_stop_train_button = gr.Button('Stop Training', variant='stop')

                    ui_status_text = gr.HighlightedText(label="Status  当前状态", value=[("No workload", "Ready")], visible=True, show_label=True).style(color_map={"Ready": "blue", "Error": "red", "Success": "green", "Running": "yellow"})
                
            # bind 2 function to click, they will be executed by binding order and update `outputs` twice
            ui_start_train_button.click(
                fn=check_and_update_status,
                inputs=[
                    ui_unique_name,
                    ui_gender,
                    ui_training_images
                ],
                outputs=[ui_status_text, ui_start_train_button]
            )
            ui_start_train_button.click(
                fn=start_train,
                inputs=[
                    ui_unique_name,
                    ui_gender,
                    ui_training_images,
                ],
                outputs=[ui_status_text, ui_start_train_button]
            )
            ui_stop_train_button.click(
                fn=stop_train,
                inputs=[ui_status_text], 
                outputs=[ui_status_text, ui_start_train_button]
            )

            ui_training_images.upload(fn=check_upload_files, inputs=[ui_training_images], outputs=[ui_start_train_button])
            ui_training_images.clear(fn=clear_upload_files, outputs=[ui_start_train_button])


        with gr.TabItem('Generate Photos (生成照片)'):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('Generation Settings (设置)')
                        ui_style_option = gr.Radio(label="Style (风格)", choices=["Builtin (预设)", "Custom (自定义)"], value="Builtin (预设)")
                        # for builtin
                        ui_builtin_style_choice = gr.Dropdown(label="", choices=list(girl_prompts.background_prompt.keys()), visible=True)
                        # for custom
                        ui_custom_style_prompt = gr.Textbox(label='Prompt', visible=False)
                        ui_custom_style_negative_prompt = gr.Textbox(label='negative Prompt', visible=False)
                        # other settings
                        ui_resolution = gr.Dropdown(label='Resolution (分辨率)', choices=["512x512", "1024x1024"])
                        ui_batch_size = gr.Slider(label='Batch Size (生成数量)', minimum=1, maximum=8, step=1)
                        ui_start_gen_button = gr.Button('Generate', variant='primary')
                        
                    with gr.Column():
                        ui_lora_model_list = gr.CheckboxGroup(label="Lora Models (选择模型)", choices=get_lora_model_list(), interactive=True)
                        with gr.Row():
                            ui_refresh_lora_models = gr.Button('Refresh Models')
                            ui_remove_lora_models = gr.Button('Remove Selected Models', variant='stop')
                    
                good_gallery = gr.Gallery(label='photos').style(columns=[8], rows=[2], object_fit="contain", height="auto")
                with gr.Accordion("Show more photos", open=False):
                    with gr.Row():
                        dissim_gallery = gr.Gallery(label='normal').style(columns=[4], rows=[2], object_fit="contain", height="auto")
                        bad_gallery = gr.Gallery(label='bad').style(columns=[4], rows=[2], object_fit="contain", height="auto")
 
            ui_refresh_lora_models.click(fn=update_lora_model_list, inputs=[], outputs=[ui_lora_model_list])
            ui_remove_lora_models.click(fn=remove_selected_lora_model, inputs=[ui_lora_model_list], outputs=[ui_lora_model_list])
            ui_style_option.change(fn=on_style_change, inputs=[ui_style_option], outputs=[ui_builtin_style_choice, ui_custom_style_prompt, ui_custom_style_negative_prompt])
            ui_start_gen_button.click(
                fn=start_generation, 
                inputs=[
                    ui_lora_model_list, 
                    ui_style_option,
                    ui_builtin_style_choice,
                    ui_custom_style_prompt,
                    ui_custom_style_negative_prompt,
                    ui_resolution, 
                    ui_batch_size
                ], 
                outputs=[
                    good_gallery, 
                    dissim_gallery, 
                    bad_gallery
                ])

    return [(ui_tabs, "PrettyU", f"prettyu_tabs")]


script_callbacks.on_ui_tabs(init_ui)
