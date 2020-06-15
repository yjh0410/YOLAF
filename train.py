from data import *
from utils.augmentations import SSDAugmentation
import tools
import torch.backends.cudnn as cudnn
import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument('-v', '--version', default='FDNet',
                        help='TinyYOLAF, MiniYOLAF')
    parser.add_argument('-d', '--dataset', default='widerface',
                        help='widerface dataset')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=5,
                        help='The upper bound of warm-up')
    parser.add_argument('--dataset_root', default=WIDERFace_ROOT, 
                        help='Location of VOC root directory')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=0, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--save_folder', default='weights/widerface/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--resume', type=str, default=None,
                        help='fine tune the model trained on MSCOCO.')

    return parser.parse_args()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# setup_seed(20)
def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    hr = False  
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True

    cfg = WF_config

    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # use multi-scale trick
    if args.multi_scale:
        print('use multi-scale trick.')
        input_size = 640
        dataset = WIDERFaceDetection(root=args.dataset_root, transform=SSDAugmentation(640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))

    else:
        input_size = cfg['min_dim']
        dataset = WIDERFaceDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))

    # build model
    if args.version == 'TinyYOLAF':
        from models.TinyYOLAF import TinyYOLAF
        anchor_size = tools.get_total_anchor_size(name=args.dataset, version=args.version)

        net = TinyYOLAF(device, input_size=input_size, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train TinyYOLAF......')

    elif args.version == 'SlimYOLAF':
        from models.SlimYOLAF import SlimYOLAF
        anchor_size = tools.get_total_anchor_size(name=args.dataset, version=args.version)

        net = SlimYOLAF(device, input_size=input_size, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train SlimYOLAF......')

    elif args.version == 'MiniYOLAF':
        from models.MiniYOLAF import MiniYOLAF
        anchor_size = tools.get_total_anchor_size(name=args.dataset, version=args.version)

        net = MiniYOLAF(device, input_size=input_size, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train MiniYOLAF......')

    else:
        print('Unknown version !!!')
        exit()

    # finetune the model trained on COCO 
    if args.resume is not None:
        print('finetune COCO trained ')
        net.load_state_dict(torch.load(args.resume, map_location=device), strict=False)


    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/widerface/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    print("----------------------------------------Face Detection--------------------------------------------")
    model = net
    model.to(device)

    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    # loss counters
    print("----------------------------------------------------------")
    print('Training on:', dataset.name)
    print('The dataset size:', len(dataset))
    print('Initial learning rate: ', args.lr)
    print("----------------------------------------------------------")

    epoch_size = len(dataset) // args.batch_size
    max_epoch = cfg['max_epoch']
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    t0 = time.time()

    # start training
    for epoch in range(max_epoch):
        
        # use cos lr
        if args.cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            set_lr(optimizer, tmp_lr)

        elif args.cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        
        # use step lr
        else:
            if epoch in cfg['lr_epoch']:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets) in enumerate(data_loader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
                    
            targets = [label.tolist() for label in targets]
            # vis_data(images, targets, input_size)

            # make train label
            targets = tools.multi_gt_creator_ab(input_size, net.stride, label_lists=targets, name=args.dataset, version=args.version)

            # to device
            images = images.to(device)
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, txtytwth_loss, total_loss = model(images, target=targets)
                     
            # backprop and update
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), txtytwth_loss.item(), total_loss.item(), input_size, t1-t0),
                        flush=True)

                t0 = time.time()

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                if epoch >= max_epoch - 10:
                    size = 416
                else:
                    size = random.randint(10, 20) * 32
                input_size = size

                # change input dim
                # But this operation will report bugs when we use more workers in data loader, so I have to use 0 workers.
                # I don't know how to make it suit more workers, and I'm trying to solve this question.
                data_loader.dataset.reset_transform(SSDAugmentation(input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))

        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth')  
                    )


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    cv2.imwrite('1.jpg', img)

    img_ = cv2.imread('1.jpg')
    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img_)
    cv2.waitKey(0)
    print(targets)


if __name__ == '__main__':
    train()