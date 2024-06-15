# from .mine_nodes import *
from .AliOffline_Seg_Obj import * 
from .AliOffline_Ddcolor import * 

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AliOffline_Seg_Obj":AliOffline_Seg_Obj,
    "AliOffline_Ddcolor":AliOffline_Ddcolor,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AliOffline_Seg_Obj":"阿里离线抠图",
    "AliOffline_Ddcolor":"阿里离线上色",
}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']