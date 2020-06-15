from __future__ import print_function 
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import config, WIDERFace_ROOT
from data import WIDERFaceDetection, WIDERFaceAnnotationTransform, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform
import tools
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
import tqdm
import pickle
from scipy.io import loadmat
from IPython import embed


parser = argparse.ArgumentParser(description='FDNet: Face Detector')
parser.add_argument('-v', '--version', default='FDNet',
                    help='TinyYOLAF')
parser.add_argument('--trained_model', default='weights/widerface/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='pred/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of WIDERFACE root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

os.makedirs(args.save_folder, exist_ok=True)


# Part-1: test model, and save all detections
def detect_face(net, image, transform, device):
    width = image.shape[1]
    height = image.shape[0]

    # to tensor
    x = torch.from_numpy(transform(image)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
    x = x.unsqueeze(0).to(device)
    # x = Variable(x.cuda(), volatile=True)

    bbox_pred, scores_pred = net(x)

    # scale each detection back up to the image
    scale = np.array([[width, height,
                       width, height]])
    # map the boxes to origin image scale
    bbox_pred *= scale

    boxes=[]
    scores = []
    for i, box in enumerate(bbox_pred):
        x1, y1, x2, y2 = box
        score = scores_pred[i]
        if score > 0.01:
            boxes.append([x1, y1, x2, y2])
            scores.append(score)

    # for i in range(detections.size(1)):
    #     j = 0
    #     while detections[0,i,j,0] >= 0.01:
    #         score = detections[0,i,j,0]
    #         pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
    #         boxes.append([pt[0],pt[1],pt[2],pt[3]])
    #         scores.append(score)
    #         j += 1
    #         if j >= detections.size(2):
    #             break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det = np.column_stack((boxes, det_conf))
    return det


def write_to_txt(f, det , event, im_name):
    f.write('{:s}\n'.format(str(event[0][0])[2:-1] + '/' + im_name + '.jpg'))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]

        #f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
        #        format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))

        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(np.floor(xmin), np.floor(ymin), np.ceil(xmax - xmin + 1), np.ceil(ymax - ymin + 1), score))


def test_widerface(net, testset, device, transform):
    # evaluation
    save_path = args.save_folder
    num_images = len(testset)

    for i in range(0, num_images):
        image = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        event = testset.pull_event(i)
        print('Testing image {:d}/{:d} {}....'.format(i+1, num_images , img_id))

        det = detect_face(net, image, transform, device)  # origin test
        
        if not os.path.exists(save_path + event):
            os.makedirs(save_path + event)
        f = open(save_path + event + '/' + img_id.split(".")[0] + '.txt', 'w')
        write_to_txt(f, det , event, img_id)


if __name__=="__main__":
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = config.WF_config
    input_size = cfg['min_dim']
    transform = BaseTransform(input_size)

    # load net
    if args.version == 'TinyYOLAF':
        from models.TinyYOLAF import TinyYOLAF
        anchor_size = tools.get_total_anchor_size(name='widerface', version=args.version)

        net = TinyYOLAF(device, input_size=input_size, trainable=False, anchor_size=anchor_size)
        print('Let us eval TinyYOLAF......')

    elif args.version == 'SlimYOLAF':
        from models.SlimYOLAF import SlimYOLAF
        anchor_size = tools.get_total_anchor_size(name=args.dataset, version=args.version)

        net = SlimYOLAF(device, input_size=input_size, trainable=False, anchor_size=anchor_size)
        print('Let us test SlimYOLAF......')

    elif args.version == 'MiniYOLAF':
        from models.MiniYOLAF import MiniYOLAF
        anchor_size = tools.get_total_anchor_size(name=args.dataset, version=args.version)

        net = MiniYOLAF(device, input_size=input_size, trainable=False, anchor_size=anchor_size)
        print('Let us eval MiniYOLAF......')

    else:
        print('Unknown version !!!')
        exit()

    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()
    print('Finished loading model!')

    # load data

    testset = WIDERFaceDetection(args.widerface_root, 'val' , None, WIDERFaceAnnotationTransform())
    # testset = WIDERFaceDetection(args.widerface_root, 'test' , None, WIDERFaceAnnotationTransform())

    test_widerface(net, testset, device, transform)