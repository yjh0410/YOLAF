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
                    help='FDNet, TinyYOLAF')
parser.add_argument('--setup', default='widerface',
                    type=str, help='widerface')
parser.add_argument('--mode', default='image',
                    type=str, help='Use the data from image, video or camera')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('--path_to_img', default='data/demo/Images/',
                    type=str, help='The path to image files')
parser.add_argument('--path_to_vid', default='data/demo/video/',
                    type=str, help='The path to video files')
parser.add_argument('--path_to_saveVid', default='data/video/result.avi',
                    type=str, help='The path to save the detection results video')
parser.add_argument('--trained_model', default='weights/widerface/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--vis_thresh', default=0.2, type=float,
                    help='Final confidence args.vis_threshold')

args = parser.parse_args()

print("----------------------------------------Face Detection--------------------------------------------")
def detect(net, device, transform, mode='image', path_to_img=None, path_to_vid=None, path_to_save=None, setup='VOC'):
    # ------------------------- Camera ----------------------------
    # I'm not sure whether this 'camera' mode works ...
    if mode == 'camera':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('current frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
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
                if scores[i] > args.vis_thresh:
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color, 2)
            cv2.imshow('face detection', img)
            cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        save_path = 'test_results'
        os.makedirs(save_path, exist_ok=True)
        for index, file_name in enumerate(os.listdir(path_to_img)):
            img = cv2.imread(path_to_img + '/' + file_name, cv2.IMREAD_COLOR)
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
                if scores[i] > args.vis_thresh:
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color, 2)
            # cv2.imshow('face detection', img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output000.avi',fourcc, 40.0, (1280,720))
        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
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
                    if scores[i] > args.vis_thresh:
                        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color, 2)
                cv2.imshow('face detection', img)
                cv2.waitKey(0)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load net
    cfg = config.WF_config
    input_size = cfg['min_dim']

    # build model
    if args.version == 'FDNet':
        from models.FDNet import FDNet

        net = FDNet(device, input_size=input_size, trainable=False)
        print('Let us test FDNet......')

    elif args.version == 'TinyYOLAF':
        from models.TinyYOLAF import TinyYOLAF
        anchor_size = tools.get_total_anchor_size(name=args.setup)

        net = TinyYOLAF(device, input_size=input_size, trainable=False, anchor_size=anchor_size)
        print('Let us test TinyYOLAF......')

    else:
        print('Unknown version !!!')
        exit()


    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()
    print('Finished loading model!')

    net = net.to(device)

    # run
    if args.mode == 'image':
        detect(net, device, BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)), 
                    mode=args.mode, path_to_img=args.path_to_img, setup=args.setup)
    elif args.mode == 'video':
        detect(net, device, BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                    mode=args.mode, path_to_vid=args.path_to_vid, path_to_save=args.path_to_saveVid, setup=args.setup)

if __name__ == '__main__':
    run()
