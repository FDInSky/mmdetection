from prettytable import PrettyTable
from collections import OrderedDict
import numpy as np 
import torch 
from mmcv import Config 
from mmdet.models.builder import build_backbone, build_detector


def yolov52mmdet():
    # 1. load yolov5.pt in yolov5 dir
    # 2. save model state dict into new pth
    # 3. use new pth here 
    # with open("../pretrain/yolov5-pacsp-mish_.weights", 'rb') as f:
    #     darknet_weights = np.fromfile(f, dtype=np.float32)
    import sys 
    sys.path.insert(0, '../yolov5')
    y5_ckpt = torch.load("../pretrain/yolov5l.pt")
    print("debug: ", y5_ckpt.keys(), type(y5_ckpt['model']))
    y5_ckpt = y5_ckpt['model'].state_dict()
    y5_ws_old = []
    for k,v in y5_ckpt.items():
        if 'anchor' in k:
            continue
        y5_ws_old.append((k,v.size()))
    print("Old Weights: ", len(y5_ws_old))

    cfg = Config.fromfile("mmdet/projects/yolo/config/yolov5l_coco.py")
    yolov5 = build_detector(cfg.model)
    y5st = yolov5.state_dict()
    y5_ws_new = []
    for k, v in y5st.items():
        if 'anchor' in k:
            continue
        y5_ws_new.append((k, v.size()))
    print("New Weights: ", len(y5_ws_new))

    table = PrettyTable(['old_name', 'old_shape', 'new_shape', 'new_name'])
    table.align['old_name'] = 'l'
    table.align['old_shape'] = 'l'
    table.align['new_shape'] = 'l'
    table.align['new_name'] = 'l'
    
    new_weights = OrderedDict()
    for y5_o, y5_n in zip(y5_ws_old, y5_ws_new):
        if y5_o[1] == y5_n[1]:
            k = y5_n[0]
            if 'backbone' in k or 'neck' in k:
                new_weights[k] = y5_ckpt[y5_o[0]]
                table.add_row([y5_o[0], y5_o[1], y5_n[1], y5_n[0]])
        else:
            continue
        
    print(table)
    save_ckpt = {
        'state_dict': new_weights
    }
    torch.save(save_ckpt, "../pretrain/mmdet_yolov5l.pth")

    
if __name__ == "__main__":
    yolov52mmdet()
