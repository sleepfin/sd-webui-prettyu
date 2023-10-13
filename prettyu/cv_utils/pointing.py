import cv2
import numpy as np
from functools import reduce

from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks


class MoreThanOneFaceError(ValueError):
    pass


class FacePointsDetection:
    def __init__(self):
        self.pipeline = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd-34G')

    def __call__(self, img):
        # {'scores': [score], 'boxes': [[x_min, y_min, x_max, y_max]], 'keypoints': [[x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]]}
        result = self.pipeline(img)
        return result


def face_points_detect(img, return_box=True, relative=False, single=True, sort_by='area'):
    """
    return_box: If True, return keypoints and boxes. If False, only return keypoints
    relative: If False, return absolute x/y of keypoints. If True, return relative x/y of keypoints to boxes
    single: If True, only return the first face detected, if sort_by_score, the first face is also the highest scored one.
    sort_by: sorted results by `area` or `score`, If None, sort will not be applied.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = FacePointsDetection()

    result = _PIPELINE(img)
    
    # import uuid
    # from modelscope.utils.cv.image_utils import draw_face_detection_result
    # img_uid = uuid.uuid4().hex
    # cv2.imwrite(f'srcImg_{img_uid}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # img_draw = draw_face_detection_result(f'srcImg_{img_uid}.jpg', result)
    # cv2.imwrite(f'dstImg_{img_uid}.jpg', img_draw)

    kpss, boxes = [], []
    for i in range(len(result['keypoints'])):
        kps = result['keypoints'][i]
        box = result['boxes'][i]
        x_min, y_min, _, _ = box

        if relative:
            kps = [(kps[i] - x_min) if (i % 2 == 0) else (kps[i] - y_min) for i in range(len(kps))]

        kps = [int(x) for x in kps]
        kps = [[kps[0], kps[1]], [kps[2], kps[3]], [kps[4], kps[5]], [kps[6], kps[7]], [kps[8], kps[9]]]
        kpss.append(kps)
        boxes.append([int(x) for x in box])
    
    scores = result['scores']
    
    if len(kpss) > 0:
        if sort_by == 'score':
            sorted_tuples = sorted(zip(scores, kpss, boxes), key=lambda x: x[0] ,reverse=True)
            scores, kpss, boxes = zip(*sorted_tuples)
        elif sort_by == 'area':
            sorted_tuples = sorted(zip(scores, kpss, boxes), key=lambda x: (x[2][2] - x[2][0]) * (x[2][3] - x[2][1]), reverse=True)
            scores, kpss, boxes = zip(*sorted_tuples)
        elif sort_by is None:
            pass
        else:
            raise ValueError(f'unknown sort_by={sort_by}')

    if not return_box:
        if single:
            if len(kpss) > 0:
                return kpss[0]
            else:
                return None
        else:
            return kpss
    else:
        if single:
            if len(kpss) > 0:
                return kpss[0], boxes[0]
            else:
                return None, None
        else:
            return kpss, boxes


def get_head(img):
    _, box = face_points_detect(img, return_box=True, single=True)
    xmin, ymin, xmax, ymax = box
    head_box = img[ymin: ymax, xmin: xmax]
    return head_box


_PIPELINE = None
