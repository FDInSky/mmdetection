from prettytable import PrettyTable
from collections import OrderedDict
import numpy as np 
import torch 
from mmcv import Config 
from mmdet.models.builder import build_backbone, build_detector

def yolov42me():
    # 1. load yolov4.pt in yolov4 dir
    # 2. save model state dict into new pth
    # 3. use new pth here 
    # with open("../pretrain/yolov4-pacsp-mish_.weights", 'rb') as f:
    #     darknet_weights = np.fromfile(f, dtype=np.float32)
    
    y4_ckpt = torch.load("../pretrain/yolov4_pacsp_mish.pth")
    y4_ws_old = []
    for k,v in y4_ckpt.items():
        if 'anchor' in k:
            continue
        y4_ws_old.append((k,v.size()))
    print("Old Weights: ", len(y4_ws_old))

    cfg = Config.fromfile("mmdet/projects/yolo/config/yolov4l_coco.py")
    yolov4 = build_detector(cfg.model)
    y4st = yolov4.state_dict()
    y4_ws_new = []
    for k, v in y4st.items():
        if 'anchor' in k:
            continue
        y4_ws_new.append((k, v.size()))
    print("New Weights: ", len(y4_ws_new))

    table = PrettyTable(['name', 'shape', 'y4_shape', 'y4_name'])
    table.align['old_name'] = 'l'
    table.align['old_shape'] = 'l'
    table.align['new_shape'] = 'l'
    table.align['new_name'] = 'l'
    
    # new_weights = {}
    # new_weights['model'] = OrderedDict()
    # new_weights['matching_heuristics'] =True
    for y4_o, y4_n in zip(y4_ws_old, y4_ws_new):
        # if y4[1] == y4[1]:
        #     # y4st[y4[0]] = y4_ckpt[y4[0]]
        #     new_weights['model'][y4[0]] = y4_ckpt[y4[0]]
        table.add_row([y4_o[0], y4_o[1], y4_n[1], y4_n[0]])
    print(table)
    # torch.save(new_weights, "../pretrain/mmdet_yolov4.pth")

    
if __name__ == "__main__":
    yolov42me()
