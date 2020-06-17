import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from data import *

ignore_thresh = IGNORE_THRESH

class BCELoss(nn.Module):
    def __init__(self,  weight=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, obj_w, noobj_w):
        pos_id = (targets==1.0).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -pos_id * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * torch.log(1.0 - inputs + 1e-14)
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return obj_w * pos_loss + noobj_w * neg_loss
        else:
            return obj_w * pos_loss + noobj_w * neg_loss


class BCE_focal_loss(nn.Module):
    def __init__(self,  weight=None, gamma=2, reduction='mean'):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        pos_id = (targets==1.0).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -pos_id * (1.0-inputs)**self.gamma * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * (inputs)**self.gamma * torch.log(1.0 - inputs + 1e-14)

        if self.reduction == 'mean':
            return torch.mean(torch.sum(pos_loss + neg_loss, 1))
        else:
            return pos_loss + neg_loss
  

def get_total_anchor_size(name='widerface', version=None):
    if name == 'widerface':
        if version == 'TinyYOLAF' or version == 'MiniYOLAF':
            all_anchor_size = TINY_MULTI_ANCHOR_SIZE_WDF
        else:
            print('Unknown Version !!')
            exit(0)
    else:
        print('Unknown Dataset !!')
        exit(0)


    return all_anchor_size


def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, anchor_w, anchor_h] ->  [xmin, ymin, xmax, ymax]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


def multi_gt_creator_ab(input_size, strides, label_lists=[], name='widerface', version=None):
    """creator multi scales gt with anchor boxes"""
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h = w = input_size
    num_scale = len(strides)
    gt_tensor = []

    # generate gt datas
    all_anchor_size = get_total_anchor_size(name=name, version=version)
    anchor_number = len(all_anchor_size) // num_scale
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, anchor_number, 1+4+1]))
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            xmin, ymin, xmax, ymax = gt_label[:-1]
            # compute the center, width and height
            c_x = (xmax + xmin) / 2 * w
            c_y = (ymax + ymin) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            if box_w < 1e-28 or box_h < 1e-28:
                # print('A dirty data !!!')
                continue    

            # compute the IoU
            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                # compute gt labels
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                weight = 2.0 - (box_w / w) * (box_h / h)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1:5] = np.array([tx, ty, tw, th])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 5] = weight
            
            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.
                
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        # s_indx, ab_ind = index // num_scale, index % num_scale
                        s_indx = index // anchor_number
                        ab_ind = index - s_indx * anchor_number
                        # get the corresponding stride
                        s = strides[s_indx]
                        # get the corresponding anchor box
                        p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                        # compute the gride cell location
                        c_x_s = c_x / s
                        c_y_s = c_y / s
                        grid_x = int(c_x_s)
                        grid_y = int(c_y_s)
                        # compute gt labels
                        tx = c_x_s - grid_x
                        ty = c_y_s - grid_y
                        tw = np.log(box_w / p_w)
                        th = np.log(box_h / p_h)
                        weight = 2.0 - (box_w / w) * (box_h / h)

                        if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1:5] = np.array([tx, ty, tw, th])
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 5] = weight


    gt_tensor = [gt.reshape(batch_size, -1, 1+4+1) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)
    
    return gt_tensor


def loss(pred_conf, pred_txtytwth, label):
    # create loss_f
    conf_loss_function = nn.CrossEntropyLoss(reduction='none') # nn.BCEWithLogitsLoss(reduction='none') 
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_conf = pred_conf.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]
        
    gt_cls = label[:, :, 0].long()
    gt_txtytwth = label[:, :, 1:-1].float()
    gt_box_scale_weight = label[:, :, -1]
    gt_mask = (gt_box_scale_weight > 0.).float()

    # objectness loss
    conf_loss = torch.mean(torch.sum(conf_loss_function(pred_conf, gt_cls), 1))
    # conf_loss = conf_loss_function(pred_conf, gt_cls)
        
    # box loss
    txty_loss = torch.mean(torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight * gt_mask, 1))
    twth_loss = torch.mean(torch.sum(torch.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight * gt_mask, 1))

    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + txtytwth_loss

    return conf_loss, txtytwth_loss, total_loss


if __name__ == "__main__":
    dataset = WIDERFaceDetection(root=WIDERFace_ROOT, transform=BaseTransform(640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))
