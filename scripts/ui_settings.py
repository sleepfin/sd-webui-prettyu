
import gradio as gr

from modules import script_callbacks, shared


def init_ui_settings():
    section = ('PrettyU', "prettyu")
    shared.opts.add_option("redundancy", shared.OptionInfo(
        3, "Generating photos will generate additional photos of this multiple and select the best photos from them", 
        gr.Slider, {"minimum": 1, "maximum": 8, "step": 1}, section=section))
    shared.opts.add_option("negative_style", shared.OptionInfo(
        "no_glasses", "Default negative prompt type", 
        gr.Radio, {"choices": ["no_glasses", "normal"]}, section=section))
    shared.opts.add_option("body_style", shared.OptionInfo(
        "upper body", "Specify whether the photo should be half-length or full-length", 
        gr.Radio, {"choices": ["upper body", "full body"]}, section=section))
    shared.opts.add_option("xformers", shared.OptionInfo(
        True, "Enable xformers when training lora model", 
        gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("train_batch_size", shared.OptionInfo(
        2, "Traning mini-batch size.", 
        gr.Slider, {"minimum": 1, "maximum": 2, "step": 1}, section=section))
    shared.opts.add_option("mixed_precision", shared.OptionInfo(
        "no", "mixed_precision", 
        gr.Radio, {"choices": ["no", "fp16", "bf16"]}, section=section))
    shared.opts.add_option("cache_latents", shared.OptionInfo(
        True, "Enable cache_latents when training lora model", 
        gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("train_image_size", shared.OptionInfo(
        512, "Image size when training lora model", 
        gr.Radio, {"choices": [512, 1024]}, section=section))
    shared.opts.add_option("gradient_checkpointing", shared.OptionInfo(
        False, "Enable gradient_checkpointing to save GPU memory usage in cost of longer training time", 
        gr.Checkbox, {"interactive": True}, section=section))
    

script_callbacks.on_ui_settings(init_ui_settings)