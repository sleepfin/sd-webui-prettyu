from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys


class SkinRetouching:
    def __init__(self):
        self.pipeline = pipeline(Tasks.skin_retouching, model='damo/cv_unet_skin-retouching')

    def __call__(self, img):
        return self.pipeline(img)


def skin_retouch(img):
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SkinRetouching()
        
    result = _PIPELINE(img)
    return result[OutputKeys.OUTPUT_IMG]


_PIPELINE = None
