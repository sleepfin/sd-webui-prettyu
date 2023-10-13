

import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from controlnet_aux import OpenposeDetector
import torch


from safetensors.torch import load_file


class ControlnetOpenpose:
    def __init__(self, controlnet_conditioning_scale=0.5, device='cuda'):
        self.scale = controlnet_conditioning_scale
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device)
        self.model = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0")

    def preprocess(self, image, height, width):
        openpose_image = self.openpose(image)
        openpose_image = openpose_image.resize((height, width))
        return openpose_image


class ControlnetCanny:
    def __init__(self, controlnet_conditioning_scale=0.5):
        self.scale = controlnet_conditioning_scale  # recommended for good generalization
        self.model = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")
    
    def preprocess(self, image, low=100, high=200):
        image = np.array(image)
        image = cv2.Canny(image, low, high)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image


class StableDiffusionGenerator:
    def __init__(self, controlnet=None, lora_paths=None, device='cuda'):
        # TODO: for now, SDXL Pipeline not support multi Controlnet
        self.controlnet = controlnet
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet.model,
        ).to(device)

        vae = self.pipe.vae
        text_encoder = self.pipe.text_encoder
        unet = self.pipe.unet

        if lora_paths:
            # load lora networks
            print(f"loading lora networks...")
            import sys
            sys.path.insert(0, "/home/zzy2/workspace/sd-scripts")
            from networks.lora import create_network_from_weights

            for lora_path in lora_paths:
                sd = load_file(lora_path)   # If the file is .ckpt, use torch.load instead.
                lora_network, sd = create_network_from_weights(0.5, None, vae, text_encoder,unet, sd)
                lora_network.apply_to(text_encoder, unet)
                lora_network.load_state_dict(sd)
                lora_network.to(device)

            print(f"successfully loaded lora networks.")


    def __call__(self, prompt, negative_prompt, steps=20, height=1024, width=1024, batch_size=1, guidance_scale=7.0, controlnet_input=None, controlnet_preprocess_kwargs={}):
        controlnet_kwargs = {}
        if controlnet_input is not None and self.controlnet is not None:
            if isinstance(self.controlnet, ControlnetOpenpose):
                controlnet_preprocess_kwargs.update({'height': height, 'width': width})

            preprocessed_input = self.controlnet.preprocess(controlnet_input, **controlnet_preprocess_kwargs)
            controlnet_kwargs.update({
                "image": preprocessed_input,
                "controlnet_conditioning_scale": self.controlnet.scale})

        # TODO: How to change sampler
        images = self.pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=steps,
            num_images_per_prompt=batch_size, 
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            **controlnet_kwargs).images

        # A list, each image is in pillow format
        return images  


