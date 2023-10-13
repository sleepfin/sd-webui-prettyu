# PrettyU

PrettyU allows you to train a digital avatar of your own and use it to generate various rich portrait photos.

Advantages:
- No need to set complex training parameters, easy-to-use, and quick to get started
- The generated photos have greater diversity, and no template images are required during generation
- Supports 1024x1024 photos (better performance, longer time)
- Supports low-memory GPUs (minimum support is around 10GB)

![Alt text](images/demo.png)

# Requirements
1. Linux with Ubuntu/Centos
2. Nvidia-GPU with CUDA >= 11.x
3. gcc/g++ >= 6.0

# Preparation
- Install [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Download base Stable-Diffusion-Checkpoint and copy it to `stable-diffusion-webui/models/Stable-diffusion`
- Install extension [sd-webui-additional-networks](https://github.com/kohya-ss/sd-webui-additional-networks)
  - Extensions -> Install from URL (`https://github.com/kohya-ss/sd-webui-additional-networks.git`) -> Install
- Install extension [adtailer](https://github.com/Bing-su/adetailer)
  - Extensions -> Install from URL (`https://github.com/Bing-su/adetailer.git`) -> Install
- Install extension [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)
  - Extensions -> Install from URL (`https://github.com/Mikubill/sd-webui-controlnet.git`) -> Install

For high-resolution 1024x1024
- Download [controlnet](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) to directory `stable-diffusion-webui/extensions/sd-webui-controlnet/models/`
  - Download [control_v11f1e_sd15_tile.pth](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11f1e_sd15_tile.pth)

# Installation
1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/sleepfin/sd-webui-prettyu.git` to "URL for extension's git repository".
4. Press "Install" button.
5. It may take few minutes (tensorflow/mmcv installation is time-consuming)
6. You will see the message "Installed into stable-diffusion-webui\extensions\xxx. Use Installed tab to restart".

# Usage
1. Training a Lora model
    - Click `Train Lora` Tab
    - Upload 20-35 photos
    - Fill in the name and gender description
    - Click `Start Training` botton

2. Wait until model is finish training
   
3. Generate photos
    - Click `Generate Photos` Tab
    - Click `Refresh Models`
    - Choose one `Lora Models`
    - Choose `Style` and `Resolution`
    - Click `Generate`

4. Generated Photos will be showed in `photos`, you can also click `Show more photos` to check more photos which is classified as low quality (There is a chance that all photos are classified as low quality)

# Suggestions
We highly recommend you to use [majicMIX realistic](https://civitai.com/models/43331?modelVersionId=126470) as base checkpoint (v2.5 or v7 version). You can also optionally set clip_skip to 2 (Settings->Stable Diffusion->Clip skip, but according to my tests, there is not much difference)

# Plans
- [x] Support higher resolution of 1024x1024 (Controlnet-tile hyper-res)
- [ ] Support Windows
- [ ] Support generating photos based on reference images
- [ ] Support 2 people photos

# Contribute

Welcome to contribute to this repo
Even if you do not know any coding, if you find any amazing prompt, you can still contribute to 
# Q&A

### GPU out of memory:
If GPU memory is less than 14GB, change settings as followed:
   - Settings -> prettyu -> Set `Traning mini-batch size` to 1
   - Settings -> prettyu -> Check `Enable gradient_checkpointing to save GPU memory usage in cost of longer training time`

### RuntimeError: cutlassF: no kernel found to launch!
Maybe caused not supported xformers on your GPU, change settings as followed:
   - Settings -> prettyu -> Uncheck `Enable xformers when training lora model`

### ONNXRuntimeError
`onnxruntime-gpu` not supported your CUDA version. Reinstall `onnxruntime-gpu` according to [THIS Document](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

### gcc/g++ version error
We need gcc/g++ (>=6.0.0, <11.0) to compile mmcv-full, follow [THIS Document](https://stackoverflow.com/questions/55345373/how-to-install-gcc-g-8-on-centos).
After you upgrade your gcc/g++, you have to reinstall mmcv-full:
```shell
pip uninstall -y mmcv-full
pip install mmcv-full==1.7.1 --no-cache-dir
```


