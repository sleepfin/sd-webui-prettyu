
import os
import glob
from cv_utils import face_utils
from cv_utils import common





face_model = "/home/zzy2/workspace/insightface/model_zoo/webface_r50_pfc.onnx"
face_anchor = "/home/zzy2/workspace/sd-webui-prettyu/prettyu/lora/data/raw/jiangnana/*.JPG"
face_query = "/home/zzy2/workspace/sd-webui-prettyu/prettyu/images/generated/*.png"



face_data = face_utils.FaceIdentity(face_anchor, face_model)
for face_img in face_data.imgs_info:
    eva = face_img.evaluate()
    if eva != 'good':
        print(eva, face_img.image_path)
