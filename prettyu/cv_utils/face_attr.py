import math

import numpy as np

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

FACE_ATTR_GENDER_MALE = 'Male'
FACE_ATTR_GENDER_FEMALE = 'Female'


def normalize_vector(v):
    # 归一化向量
    return v / np.linalg.norm(v)


def get_angle_of_two_vector(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    ret = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    ret = ret * 180 / np.pi
    return ret


def get_distance_of_two_vector(v1, v2):
    return math.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)


def get_vector_through_two_points(start_point, end_point):
    # 获取start_point指向end_point的向量(以start_point为圆心，返回end_point的向量)
    return np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])


def get_angle_to_horizon(start_point, end_point):
    v = get_vector_through_two_points(start_point, end_point)
    h = get_vector_through_two_points([0, 0], [1, 0])
    angle = get_angle_of_two_vector(v, h)
    if angle <= 90:
        return angle, 'clockwise'
    else:
        return 180 - angle, 'anticlockwise'


def get_center_of_two_vector(v1, v2):
    return int((v1[0] + v2[0]) / 2), int((v1[1] + v2[1]) / 2)


class FaceAttrPipeline(object):
    def __init__(self, task=Tasks.face_attribute_recognition, model='damo/cv_resnet34_face-attribute-recognition_fairface'):
        self.pipeline = pipeline(task, model)

    def __call__(self, img):
        """
        return value example:
        {
            'scores': [[g1, g2], [a1, a2, a3, a4, a5, a6, a7, a8, a9]], 
            'labels': [['Male', 'Female'], ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']]
        }
        """
        result = self.pipeline(img)
        return result


def get_gender_and_age(face_img, gender_threshold=0.8):
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = FaceAttrPipeline()
    
    result = _PIPELINE(face_img)
    if result['scores'] is None or not result['scores']:
        return None, None
    
    gender_scores, age_scores = result['scores']
    gender_labels, age_labels = result['labels']
    gid = np.argmax(gender_scores)
    # gender的得分如果小于gender_threshold，则认为性别是不清楚的，提高性别的辨识度
    if gender_scores[gid] < gender_threshold:
        gender = 'Unknown'
    else:
        gender = gender_labels[gid]

    age = age_labels[np.argmax(age_scores)]
    return gender, age


class FacialFeature(object):
    def __init__(self, img, face_img, box, kps):
        self._img = img
        self._face_img = face_img
        self._box = box
        self._kps = kps

        xmin, ymin, xmax, ymax = box
        self.head_area = (xmax - xmin) * (ymax - ymin)
        img_area = img.shape[0] * img.shape[1]
        self.head_area_ratio = self.head_area / img_area
        left_eye, right_eye, nose, left_mouth, right_mouth = kps
        self.eye_center = get_center_of_two_vector(left_eye, right_eye)
        self.eye_degree, _ = get_angle_to_horizon(left_eye, right_eye)
        self.mouth_degree, _ = get_angle_to_horizon(left_mouth, right_mouth)
        self.eye_distance = get_distance_of_two_vector(left_eye, right_eye)
        self.mouth_distance = get_distance_of_two_vector(left_mouth, right_mouth)
        self.eye_mouth_ratio = self.eye_distance / (self.mouth_distance + 0.0001)

        # 这些属性需要耗时操作，取值的时候再动态初始化
        self._gender = None
        self._age = None
        self._is_male = None

    def _build(self):
        # 判断性别时，由于face的box仅框了脸，向外扩张一圈，让头发元素进来，提高性别判断准确率
        h, w = self._img.shape[:2]
        xmin, ymin, xmax, ymax = self._box
        dx, dy = int(0.2 * (xmax - xmin)), int(0.2 * (ymax - ymin))
        xmin = max(0, xmin - dx)
        xmax = min(w, xmax + dx)
        ymin = max(0, ymin - dy)
        ymax = min(h, ymax + dy)
        face_img_expanded = self._img[ymin: ymax, xmin: xmax]

        gender, age = get_gender_and_age(face_img_expanded)
        self._gender = gender
        self._age = age
        if self._gender is not None:
            self._is_male = (self.gender == FACE_ATTR_GENDER_MALE)

    @property
    def gender(self):
        if self._gender is not None:
            return self._gender
        
        self._build()
        return self._gender

    @property
    def is_male(self):
        if self._is_male is not None:
            return self._is_male
        
        self._build()
        return self._is_male
    
    @property
    def age(self):
        if self._age is not None:
            return self._age
        
        self._build()
        return self._age
    

_PIPELINE = None
