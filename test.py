import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from data import WIDERFaceDetection, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform
from data import config
import numpy as np
import cv2
import tools
import time


parser = argparse.ArgumentParser(description='Face Detection')
parser.add_argument('-v', '--version', default='FDNet',
                    help='TinyYOLAF, MiniYOLAF')
parser.add_argument('-d', '--dataset', default='widerface',
                    help='widerface dataset')
parser.add_argument('--trained_model', default='weights/widerface/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--vis_thresh', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda.')
parser.add_argument('--dataset_root', default=WIDERFace_ROOT, 
                    help='Location of widerface root directory')

args = parser.parse_args()

print("----------------------------------------Face Detection--------------------------------------------")

def test_net(net, device, testset, transform, thresh):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img = testset.pull_image(index)
        # img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        dets = net(x)      # forward pass
        print("detection time used ", time.time() - t0, "s")
        bbox_pred, scores = dets

        # scale each detection back up to the image
        scale = np.array([[img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]]])
        # map the boxes to origin image scale
        bbox_pred *= scale

        class_color = (255, 0, 0)
        for i, box in enumerate(bbox_pred):
            xmin, ymin, xmax, ymax = box
            # print(xmin, ymin, xmax, ymax)
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color, 2)
        cv2.imshow('face detection', img)
        cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite(str(index).zfill(6) +'.jpg', img)
        # if index > 5:
        #     break



def test():
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load net
    cfg = config.WF_config
    input_size = cfg['min_dim']

    # dataset
    if args.dataset == 'widerface':
        testset = WIDERFaceDetection(args.dataset_root, image_sets='val')

    else:
        print('Only support Wider-Face dataset !!')
        exit(0)

    # build model
    if args.version == 'TinyYOLAF':
        from models.TinyYOLAF import TinyYOLAF
        anchor_size = tools.get_total_anchor_size(name=args.dataset, version=args.version)

        net = TinyYOLAF(device, input_size=input_size, trainable=False, anchor_size=anchor_size)
        print('Let us test TinyYOLAF......')

    elif args.version == 'CenterYOLAF':
        from models.CenterYOLAF import CenterYOLAF

        net = CenterYOLAF(device, input_size=input_size, trainable=False)
        print('Let us test CenterYOLAF......')

    else:
        print('Unknown version !!!')
        exit()


    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()
    print('Finished loading model!')

    net = net.to(device)

    # evaluation
    test_net(net, device, testset,
             BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
             thresh=args.vis_thresh)

if __name__ == '__main__':
    test()
