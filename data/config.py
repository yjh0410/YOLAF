# config.py
import os.path

## widerface
# 640 for TinyYOLAF and MiniYOLAF with 3 level feature
TINY_MULTI_ANCHOR_SIZE_WDF = [[3.41, 5.95], [6.45, 12.08], [11.22, 20.98],     
                              [18.62, 34.15], [30.23, 53.54], [52.25, 83.88],     
                              [84.57, 142.49], [151.82, 224.33], [268.92, 355.28]]   

# 640 for XXX model with 4 level feature
MULTI_ANCHOR_SIZE_WDF = [[3.27, 5.7], [6.01, 11.27], [10.2, 18.74],     
                         [15.62, 29.9], [24.91, 42.54], [33.87, 65.1],     
                         [56.07, 78.07], [64.08, 129.93], [126.46, 120.83],
                         [101.85, 208.92], [194.59, 267.97], [308.13, 405.41]]   


IGNORE_THRESH = 0.5

# wider_face
WF_config = {
    'num_classes': 1,
    'lr_epoch': (120, 160),
    'max_epoch': 200,
    'min_dim': 640,
    'scale_thresh': ((0, 50), (50, 300), (300, 1e4)),
    'name': 'widerface',
}
