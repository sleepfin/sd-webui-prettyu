import os
import sys
import importlib


sdwebui_root_dir = "/home/zzy2/workspace/stable-diffusion-webui"
sdwebui_controlnet_root_dir = os.path.join(sdwebui_root_dir, "extensions", "sd-webui-controlnet")


def import_module(module_dir, module_name):
    if module_name in sys.modules:
        existed_module = sys.modules[module_name].__file__
        existed_module = os.path.splitext(existed_module)[0]
        if os.path.exists(
            os.path.join(module_dir, '%s.py' % module_name)) or os.path.exists(
            os.path.join(module_dir, '%s.pyc' % module_name)):
            new_module = os.path.join(module_dir, '%s' % module_name)
        else:
            new_module = os.path.join(module_dir, module_name, '__init__')

        if existed_module == new_module:
            return sys.modules[module_name]
        else:
            raise ValueError('conflict import. Already defined: %s. Try to define: %s' % (existed_module, new_module))

    if module_dir not in sys.path:
        sys.path.append(module_dir)

    module = importlib.import_module(module_name)
    return module


import_module(sdwebui_root_dir, 'modules')
import_module(sdwebui_controlnet_root_dir, "annotator")
openpose_module = import_module(os.path.join(sdwebui_controlnet_root_dir, "annotator"), "openpose")
model_openpose = openpose_module.OpenposeDetector()


def face_kpss_detect(img):
    return model_openpose.detect_poses(
            img,
            include_hand=False,
            include_face=True)
