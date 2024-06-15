
from typing import List
import os
import datetime
from PIL import Image
import numpy as np
import folder_paths
import tensorflow as tf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from .utils import *
from .merge_utils import png_to_mask

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")
model_path = os.path.join(comfy_path, "models","modelscope","cv_unet_image-matting")

class AliOffline_Seg_Obj:
   
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
               {   
                "image":("IMAGE", {"default": "","multiline": False}),
                "return_form":   (["whiteBK", "mask"],{"default":"mask"} ),   
              }
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("image","mask",)
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH"

    def sample(self,image,return_form):
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        model = pipeline(
            Tasks.portrait_matting, model=model_path
        )
        result = model(tensor2pil(image))
        ndarray = cv2.cvtColor(result[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGBA)
        outImage = array2image(ndarray)
        outTensor = pil2tensor(outImage)
        if return_form == "whiteBK":
          outMask = pil2tensor(png_to_mask(outImage))
        else:
          outMask = pil2tensor(png_to_mask(outImage,False,0))
        sess.close()
        return (outTensor,outMask)