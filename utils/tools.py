import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


class BCELoss(nn.Module):
    def __init__(self,  weight=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets):
        pos_id = (targets==1.0).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -pos_id * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * torch.log(1.0 - inputs + 1e-14)
        pos_num = torch.sum(pos_id, 1)
        neg_num = torch.sum(neg_id, 1)
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss

class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets):
        pos_id = (targets==1.0).float()
        neg_id = 1 - pos_id.float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        pos_num = torch.sum(pos_id, 1)
        neg_num = torch.sum(neg_id, 1)
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss

class BCE_focal_loss(nn.Module):
    def __init__(self,  weight=None, gamma=2, reduction='mean'):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        pos_id = (targets==1.0).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -pos_id * (1.0-inputs)**self.gamma * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * (inputs)**self.gamma * torch.log(1.0 - inputs + 1e-14)

        if self.reduction == 'mean':
            return torch.mean(torch.sum(pos_loss+neg_loss, 1))
        else:
            return pos_loss+neg_loss
  
def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1e-28 or box_h < 1e-28:
        # print('A dirty data !!!')
        return False    

    # map center point of box to the grid cell
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # compute the (x, y, w, h) for the corresponding grid cell
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    weight = 2.0 - (box_w / w) * (box_h / h)
    box_area = (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight, box_area

def multi_gt_creator(input_size, strides, scale_thresh, label_lists=[], name='widerface'):
    """creator multi scales gt"""
    assert len(strides) == len(scale_thresh)
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h, w = input_size, input_size
    gt_tensor = []

    # generate gt datas
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, 1+4+1]))

    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # map center point of box to the grid cell
            for index, s in enumerate(strides):
                thresh = scale_thresh[index]
                results = generate_dxdywh(gt_label, w, h, s)
                if not results:
                    break
                else:
                    grid_x, grid_y, tx, ty, tw, th, weight, box_area = results
                    area_ratio = box_area

                    if area_ratio > thresh[0] and area_ratio <= thresh[1]:
                        gt_tensor[index][batch_index, grid_y, grid_x, 0] = 1.0
                        gt_tensor[index][batch_index, grid_y, grid_x, 1:5] = np.array([tx, ty, tw, th])
                        gt_tensor[index][batch_index, grid_y, grid_x, 5] = weight

                        break
                    else:
                        continue
                
    gt_tensor = [gt.reshape(batch_size, -1, 1+4+1) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)
    return gt_tensor

def loss(pred_obj, pred_txtytwth, label):
    obj = 5.0
    noobj = 1.0

    # create loss_f
    obj_loss_function = MSELoss(reduction='mean')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_obj = torch.sigmoid(pred_obj[:, :, 0])
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]
        
    gt_obj = label[:, :, 0].float()
    gt_txtytwth = label[:, :, 1:-1].float()
    gt_scale_weight = label[:, :, -1]

    # objectness loss
    pos_loss, neg_loss = obj_loss_function(pred_obj, gt_obj)
    obj_loss = obj * pos_loss + noobj * neg_loss
        
    # box loss
    txty_loss = torch.mean(torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2) * gt_scale_weight * gt_obj, 1))
    twth_loss = torch.mean(torch.sum(torch.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2) * gt_scale_weight * gt_obj, 1))

    txtytwth_loss = txty_loss + twth_loss

    total_loss = obj_loss + txtytwth_loss

    return obj_loss, txtytwth_loss, total_loss

# IoU and its a series of variants
def IoU(pred, label):
    """
        Input: pred  -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
               label -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
        Output: IoU  -> [iou_1, iou_2, ...],    size=[B, H*W]
    """

    return

def GIoU(pred, label):
    """
        Input: pred  -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
               label -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
        Output: GIoU -> [giou_1, giou_2, ...],    size=[B, H*W]
    """

    return

def DIoU(pred, label):
    """
        Input: pred  -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
               label -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
        Output: DIoU -> [diou_1, diou_2, ...],    size=[B, H*W]
    """

    return

def CIoU(pred, label):
    """
        Input: pred  -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
               label -> [xmin, ymin, xmax, ymax], size=[B, H*W, 4]
        Output: CIoU -> [ciou_1, ciou_2, ...],    size=[B, H*W]
    """

    return

def compute_miss(input_size, stride, fpn=False, scale_thresholds=None):
    dataset = VOCDetection(root=VOC_ROOT, transform=BaseTransform(input_size, MEANS))
    data_size = len(dataset)

    data_loader = data.DataLoader(dataset, 32,
                                num_workers=8,
                                shuffle=True, collate_fn=detection_collate,
                                pin_memory=True)
    batch_iterator = iter(data_loader)

    # print('Total image number : ', data_size)
    miss_r = []
    total_b_number = 0
    total_a_number = 0
    for _, targets in batch_iterator:
        for batch_index in range(len(targets)):
            total_b_number += len(targets[batch_index])

        # make gt
        targets = [label.tolist() for label in targets]
        if not fpn:
            targets = gt_creator(input_size=input_size, stride=stride, label_lists=targets)
        else:
            targets = multi_gt_creator(input_size=input_size, strides=stride, 
                                        scale_thresholds=scale_thresholds, label_lists=targets)
        
        total_a_number += np.sum(targets[:, :, 0])
    print('Total gt number before making gt : ', total_b_number)
    print('Total gt number after making gt : ', total_a_number)
    print('Found ratio : ', total_a_number / total_b_number)
    print('Miss ratio : ', 1.0 - total_a_number / total_b_number)

    return total_a_number / total_b_number, 1.0 - total_a_number / total_b_number

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    input_size = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    sl_fr = []
    ml_fr = []
    for size in input_size:
        print('Input size : ', [size, size])
        # single level
        fr_, mr_ = compute_miss(input_size=[size, size], stride=32)
        sl_fr.append(fr_)
        # multi scale
        fr_, mr_ = compute_miss(input_size=[size, size], stride=[8, 16, 32], 
                                fpn=True, scale_thresholds=[[0, 0.046], [0.046, 0.227], [0.227, 1.0]])
        ml_fr.append(fr_)
    
    plt.plot(input_size, sl_fr, c='r', marker='o', label='single-level')
    plt.plot(input_size, ml_fr, c='b', marker='*', label='multi-level')
    plt.ylabel('found ratio (fr)')
    plt.xlabel('input size')
    plt.legend(loc='lower right')
    plt.show()