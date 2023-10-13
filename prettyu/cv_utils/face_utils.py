import glob
from collections import Counter

import numpy as np
import huggingface_hub

from .onnx_utils import ONNXModel
from . import pointing
from . import common
from . import resizing
from . import pointing
from . import face_attr
from . import gender_const


def get_cos_similar(v0, v1):
    num = float(np.dot(v0, v1))
    denom = np.linalg.norm(v0) * np.linalg.norm(v1)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    

class FaceMatchBase(object):
    def __init__(self):
        self.anchor_features = []

    def extract_feature(img):
        raise NotImplementedError()
    

class FaceMatchONNX(FaceMatchBase):
    def __init__(self, pretrained_model):
        super(FaceMatchONNX, self).__init__()
        self.model = ONNXModel(pretrained_model)

    def preprocess(self, img):
        # img = cv2.resize(img, (112, 112))
        img = resizing.resize_and_pad(img, target_size=112)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        return img

    def extract_feature(self, img):
        img = self.preprocess(img)
        out = self.model.forward(img)
        out = out[0][0]
        # original out is [array(shape=(1, 512))], we return array(shape=(512, )) here
        return out


# class FaceMatchTorch(FaceMatchBase):
#     def __init__(self, pretrained_model, insightface_root_dir):
#         super(FaceMatchTorch, self).__init__()
#         sys.path.insert(0, os.path.join(insightface_root_dir, 'recognition'))
#         from arcface_torch.backbones import get_model
#         self.model = get_model('r100', fp16=False)
#         self.model.load_state_dict(torch.load(pretrained_model))
#         self.model.eval()

#     def preprocess(self, img):
#         # img = cv2.resize(img, (112, 112))
#         img = resizing.resize_and_pad(img, target_size=112)
#         img = np.transpose(img, (2, 0, 1))
#         img = torch.from_numpy(img).unsqueeze(0).float()
#         img.div_(255).sub_(0.5).div_(0.5)
#         return img

#     @torch.no_grad()
#     def extract_feature(self, img):
#         img = self.preprocess(img)
#         out = self.model(img).numpy()
#         return out[0]


class FaceImage(object):
    def __init__(self, img, kps, box):
        self.img = img
        self.kps = kps
        self.box = box
        xmin, ymin, xmax, ymax = box
        self.face_img = img[ymin: ymax, xmin: xmax]
        self.face_attr = face_attr.FacialFeature(img, self.face_img, box, kps)
        self.feat = None
        self.feedback = None

    def fill_feature(self, face_model: FaceMatchBase):
        self.feat = face_model.extract_feature(self.face_img)

    def evaluate(
            self,
            body_style='upper body',
            eye_mouth_degree_max=15, 
            eye_mouth_degree_gap=5,
            eye_mouth_ratio_max=1.42, 
            eye_mouth_ratio_min=1.1,
            full_body_head_ratio_min=0.015,
            full_body_head_ratio_max=0.1,
            upper_body_head_ratio_min=0.06,
            upper_body_head_ratio_max=0.4):
        # 单人人脸评价
        # 1) 只有1个人脸
        # 2) 人脸5个关键点可见
        # 3）两眼之间的角度和嘴角之间的角度相近（差距小于10度）
        # 4）瞳距和嘴长的比例合理（瞳距/嘴长=1.1~1.42）
        # 5）人脸占图片比例大小合理（半身照比例在0.06~0.4之间，全身在0.015~0.1之间）
        # 6) 人脸是否摆正没有过度歪斜（歪斜角度不超过15度）
        if self.feedback:
            return self.feedback
        
        if body_style not in ['upper body', 'full body']:
            raise ValueError('body_style should be `upper body` or `full body`')

        if len(self.kps) != 5:
            self.feedback = "num features not match"
            return self.feedback
        
        facial = self.face_attr
        if abs(facial.eye_degree - facial.mouth_degree) > eye_mouth_degree_gap:
            self.feedback = "eye mouth degree not match"
            return self.feedback
        
        if facial.eye_degree > eye_mouth_degree_max:
            self.feedback = "eye degree too tilt"
            return self.feedback
        
        if facial.mouth_degree > eye_mouth_degree_max:
            self.feedback = "mouth degree too tilt"
            return self.feedback

        if facial.eye_mouth_ratio > eye_mouth_ratio_max:
            self.feedback = "mouth distance too small or eye distance too large"
            return self.feedback
        
        if facial.eye_mouth_ratio < eye_mouth_ratio_min:
            self.feedback = "mouth distance too large or eye distance too small"
            return self.feedback
        
        if body_style == 'upper body':
            if facial.head_area_ratio > upper_body_head_ratio_max:
                self.feedback = "head too close"
                return self.feedback
            
            if facial.head_area_ratio < upper_body_head_ratio_min:
                self.feedback = "head to far"
                return self.feedback

        if body_style == 'full body':
            if facial.head_area_ratio > full_body_head_ratio_max:
                self.feedback = "head too close"
                return self.feedback
            
            if facial.head_area_ratio < full_body_head_ratio_min:
                self.feedback = "head to far"
                return self.feedback
            
        self.feedback = 'good'
        return self.feedback

    def __repr__(self):
        return f"ImageInfo(face.shape={self.face_img.shape}, origin.shape={self.img.shape}, filename={self.image_path}, face_attrs={self.face_attrs},)"


class PortraitPhoto(object):
    def __init__(self, image_path=None, img=None, expected_face_num=1):
        if image_path is None and img is None:
            raise ValueError('one of image_path or img should be given')
        
        if image_path is not None and img is not None:
            raise ValueError('only one of image_path or img should be given')
        
        if image_path is not None:
            self.image_path = image_path
            self.img = common.read_img_rgb(image_path)
        else:
            self.image_path = None
            self.img = img

        self.expected_face_num = expected_face_num
        self.kpss, self.boxes = pointing.face_points_detect(self.img, return_box=True, relative=False, single=False, sort_by='area')
        self.face_images = []
        for kps, box in zip(self.kpss, self.boxes):
            self.face_images.append(FaceImage(self.img, kps, box))

        if len(self.face_images) > 0:
            # 找到图中最大区域的人脸，最为main face
            self.main_idx = np.argmax([face_img.face_attr.head_area for face_img in self.face_images])
            self.main_face_img = self.face_images[self.main_idx]
        else:
            self.main_idx = 0
            self.main_face_img = None

        # 这里的feat是main_idx的
        self.main_feat = None
        # 这里的feedback是全部的
        self.feedbacks = None

    def fill_feature(self, face_model):
        if len(self.face_images) < 1:
            return
        
        for face_image in self.face_images:
            face_image.fill_feature(face_model)

        self.main_feat = self.face_images[self.main_idx].feat

    def get_face_by_mask(self, mask):
        if len(self.face_images) < 1:
            return None
        
        # mask.shape=(h, w), value should be 0/1 or 0/255
        if len(mask.shape) != 2:
            raise ValueError('mask shape not with format (h, w)')
        
        unique_vals = np.unique(mask).tolist()
        if set(unique_vals) != {0, 1}:
            raise ValueError('mask values not 0/1')

        # 找到和mask重合度最高的face是哪个
        max_hit, tgt_idx = -1, None
        for i in range(len(self.boxes)):
            xmin, ymin, xmax, ymax = self.boxes[i]
            hit_num = np.sum(mask[ymin: ymax, xmin: xmax])
            if hit_num > max_hit:
                max_hit = hit_num
                tgt_idx = i
        
        return self.face_images[tgt_idx]

    def evaluate(self, 
                 body_style='upper body',
                 eye_mouth_degree_max=15, 
                 eye_mouth_degree_gap=5,
                 eye_mouth_ratio_max=1.42, 
                 eye_mouth_ratio_min=1.1,
                 full_body_head_ratio_min=0.015,
                 full_body_head_ratio_max=0.1,
                 full_body_head_ratio_gap_max=0.016,
                 upper_body_head_ratio_min=0.06,
                 upper_body_head_ratio_max=0.4,
                 upper_body_head_ratio_gap_max=0.016,
                 genders_prompts=None):
        """
        最后一个genders_prompts参数，输入一个列表，如["1girl", "1man", "a little girl"]，说明希望photo中有2男1女
        注意这里的gender必须是gender_const.py中定义的字符串
        evaluate函数会观察所有的face_images，统计是否是2男1女，如果符合，那就是good，否则不是good
        """
        if self.feedbacks is not None:
            return self.feedbacks

        if len(self.kpss) != self.expected_face_num:
            self.feedbacks = ["face number not match"] * self.expected_face_num
            return self.feedbacks
        
        if genders_prompts:
            real_gender_list = [face_img.face_attr.gender for face_img in self.face_images]
            if any([x is None for x in real_gender_list]):
                self.feedbacks = ["gender is None"] * self.expected_face_num
                return self.feedbacks

            real_genders = Counter({'Male': 0, 'Female': 0})
            real_genders.update(real_gender_list)
            expected_genders = Counter({'Male': 0, 'Female': 0})
            expected_genders.update([gender_const.LABEL_MAP[x] for x in genders_prompts])

            if (expected_genders['Male'] != real_genders['Male']) or (expected_genders['Female'] != real_genders['Female']):
                self.feedbacks = ["gender not match prompts"] * self.expected_face_num
                return self.feedbacks
            
        # 如果有多个肖像，判断这些肖像的head_ratio是否接近
        head_ratios = [face_img.face_attr.head_area_ratio for face_img in self.face_images]
        if body_style == 'upper body':
            head_gap_max = upper_body_head_ratio_gap_max 
        elif body_style == 'full body':
            head_gap_max = full_body_head_ratio_gap_max 
        else:
            raise ValueError(f'unknown body_style={body_style}')

        if max(head_ratios) - min(head_ratios) > head_gap_max:
            self.feedbacks = ["head size gap too big"] * self.expected_face_num
            return self.feedbacks
        
        # 如果有多个肖像，这些肖像的head在y坐标上不能有太大的高度落差，差距不大于一个head的距离
        max_head_area = max([face_img.face_attr.head_area for face_img in self.face_images])
        eye_y = [face_img.face_attr.eye_center[1] for face_img in self.face_images]
        if max(eye_y) - min(eye_y) > max_head_area:
            self.feedbacks = ["head y-coord gap too large"] * self.expected_face_num
            return self.feedbacks

        self.feedbacks = [
            face_img.evaluate(
                body_style=body_style,
                eye_mouth_degree_max=eye_mouth_degree_max,
                eye_mouth_degree_gap=eye_mouth_degree_gap,
                eye_mouth_ratio_max=eye_mouth_ratio_max,
                eye_mouth_ratio_min=eye_mouth_ratio_min,
                full_body_head_ratio_min=full_body_head_ratio_min,
                full_body_head_ratio_max=full_body_head_ratio_max,
                upper_body_head_ratio_min=upper_body_head_ratio_min,
                upper_body_head_ratio_max=upper_body_head_ratio_max
            ) for face_img in self.face_images]
        return self.feedbacks
    
    @property
    def main_feedback(self):
        if not self.feedbacks:
            raise ValueError('not evaluated yet')
    
        return self.feedbacks[self.main_idx]

    def save(self, file_path):
        common.save_img(self.img, file_path)

    def __repr__(self):
        return "\n".join(str(face_img) for face_img in self.face_images)


class FaceIdentity(object):
    # 通过N张图片，建立一个人脸特征库，这个类包含一个独立identity的人脸图片、特征、特征向量等
    def __init__(self, face_model, file_pattern=None, imgs=None):
        if file_pattern is None and imgs is None:
            raise ValueError('one of file_pattern or imgs should be given')
        
        if file_pattern is not None and imgs is not None:
            raise ValueError('only one of file_pattern or imgs should be given')
        
        self.face_model = face_model
        if file_pattern is not None:
            self.file_pattern = file_pattern
            self.anchor_photos = []
            for image_path in glob.glob(file_pattern):
                face_photo = PortraitPhoto(image_path=image_path)
                face_photo.fill_feature(self.face_model)
                self.anchor_photos.append(face_photo)
        else:
            self.file_pattern = None
            self.anchor_photos = []
            for img in imgs:
                face_photo = PortraitPhoto(img=img)
                face_photo.fill_feature(self.face_model)
                self.anchor_photos.append(face_photo)

    def similarity_many(self, query_photo: FaceImage, sort_by_sim=True):
        if query_photo.feat is None:
            query_photo.fill_feature(self.face_model)
    
        sim = [(get_cos_similar(query_photo.feat, anchor_photo.main_feat), anchor_photo) for anchor_photo in self.anchor_photos]
        if sort_by_sim:
            sim = sorted(sim, key=lambda x: x[0])

        return sim
    
    def similarity(self, query_photo: FaceImage):
        if query_photo.feat is None:
            query_photo.fill_feature(self.face_model)

        sim = np.mean([get_cos_similar(query_photo.feat, anchor_photo.main_feat) for anchor_photo in self.anchor_photos])
        return sim


# def get_face_sim(face_recogniztion_model, anchor_file_pattern, query_file_pattern):
#     face_data = FaceIdentity(anchor_file_pattern, face_recogniztion_model)
#     for img_path in glob.glob(query_file_pattern):
#         query_img_info = FaceImage(img_path)
#         ret = face_data.similarity(query_img_info)
#         print(ret, img_path)


# def get_face_sim_each(face_recogniztion_model, anchor_file_pattern, query_file):
#     face_data = FaceIdentity(anchor_file_pattern, face_recogniztion_model)
#     query_img_info = FaceImage(query_file)
#     ret = face_data.similarity_many(query_img_info)
#     print(ret)


def get_face_model():
    pretrained_model = huggingface_hub.hf_hub_download(repo_id="Cathy0908/insight-face", filename="webface_r50_pfc.onnx")
    global _face_model_instance
    if _face_model_instance is None:
        _face_model_instance = FaceMatchONNX(pretrained_model=pretrained_model)
    
    return _face_model_instance
    

_face_model_instance = None
