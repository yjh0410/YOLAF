# YOLAF
You Only Look At Face

This is a demo about face detection. 

No paper.

No sota model.

I provide two models, TinyYOLAF(anchor-based) and CenterYOLAF(anchor-free).

They are both fast enough (50-60 FPS on GTX-1060-mobile-3G GPU) and effective.

The AP on widerface val dataset:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>          <td bgcolor=white> size </td><td bgcolor=white> Easy </td><td bgcolor=white> Medium  </td><td bgcolor=white> Hard </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> TinyYOLAF</th>  <td bgcolor=white> 640 </td><td bgcolor=white> 0.784 </td><td bgcolor=white> 0.827 </td><td bgcolor=white> 0.771 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> CenterYOLAF</th><td bgcolor=white> 640 </td><td bgcolor=white> 0.889 </td><td bgcolor=white> 0.850 </td><td bgcolor=white> 0.720 </td></tr>
</table></tbody>

# TinyYOLAF
TinyYOLAF is very simple. Its backbone network is darknet_tiny which is designed by myself.

![Image](https://github.com/yjh0410/YOLAF/blob/master/img_file/darknet_tiny.png)

Since it is an anchor-based method, I design some anchor boxes with kmeans used in YOLOv3. You can open ```data/config.py``` to check them.


# CenterYOLAF
CenterYOLAF is also very simple. Following CenterFace and CenterNet, I use ResNet-18 as backbone and several deconv to get a heatmap. 

However, there are some differences:

1. In CenterNet, it copies the codes from CornerNet to generate a radius for Gauss Kernel who will create a groudtruth heatmap. But, I can't understand why this method
is suitable for center. So I apply a different method that I use weight and height of a bounding box to calculate sigma_w and sigma_h. For more details, you can open 
```tools.py``` to see.

2. In CenterNet, it uses L1 to learn offset while I use Sigmoid and BCELoss as the offset is between 0 and 1. Just like YOLOv3.


# WiderFace
I only download widerface dataset, and evaluate my model on Val dataset.


# Train on widerface
```Shell
python train.py -v TinyYOLAF --cuda -hr --num_workers 8

python train.py -v CenterYOLAF --cuda -hr --num_workers 8
```

# Eval on widerface
```Shell
python widerface_val.py -v TinyYOLAF --trained_model [path_to_model]

python widerface_val.py -v CenterYOLAF --trained_model [path_to_model]
```

# Demo
```Shell
python demo.py -v [select a model] --cuda --trained_model [path_to_model] --mode [camera/image/video]
```

CenterYOLAF:

![Image](https://github.com/yjh0410/YOLAF/blob/master/img_file/000001.jpg)
![Image](https://github.com/yjh0410/YOLAF/blob/master/img_file/000004.jpg)
![Image](https://github.com/yjh0410/YOLAF/blob/master/img_file/000006.jpg)
![Image](https://github.com/yjh0410/YOLAF/blob/master/img_file/000010.jpg)
![Image](https://github.com/yjh0410/YOLAF/blob/master/img_file/000014.jpg)
![Image](https://github.com/yjh0410/YOLAF/blob/master/img_file/000043.jpg)
![Image](https://github.com/yjh0410/YOLAF/blob/master/img_file/000057.jpg)


