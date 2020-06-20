import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d, SPP, DeConv2d
from backbone import *
import numpy as np
import tools

class CenterYOLAF(nn.Module):
    def __init__(self, device, input_size=None, trainable=False, conf_thresh=0.3, nms_thresh=0.3, topk=200, hr=False):
        super(CenterYOLAF, self).__init__()
        self.device = device
        self.input_size = input_size
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 4
        self.topk = topk
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size, input_size, input_size, input_size]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

        # backbone
        self.backbone = resnet18(pretrained=trainable)

        # neck: deconv + FPN
        self.deconv5 = DeConv2d(512, 256, ksize=2, stride=2) # 32 -> 16
        self.deconv4 = DeConv2d(256, 128, ksize=2, stride=2) # 16 -> 8
        self.deconv3 = DeConv2d(128, 64, ksize=2, stride=2)  #  8 -> 4

        self.smooth5 = Conv2d(512, 512, ksize=3, padding=1)
        self.smooth4 = Conv2d(256, 256, ksize=3, padding=1)
        self.smooth3 = Conv2d(128, 128, ksize=3, padding=1)
        self.smooth2 = Conv2d( 64,  64, ksize=3, padding=1)

        # head: conf, txty, twth
        self.conf_pred = nn.Sequential(
            Conv2d(64, 32, ksize=3, padding=1, leakyReLU=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.txty_pred = nn.Sequential(
            Conv2d(64, 32, ksize=3, padding=1, leakyReLU=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )
       
        self.twth_pred = nn.Sequential(
            Conv2d(64, 32, ksize=3, padding=1, leakyReLU=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )


    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
        return grid_xy

    def set_grid(self, input_size):
        self.grid_cell, self.stride_tensor = self.create_grid(input_size)
        self.scale = np.array([[[input_size, input_size, input_size, input_size]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_boxes(self, pred):
        """
        input box :  [delta_x, delta_y, sqrt(w), sqrt(h)]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        pred[:, :, :2] = (torch.sigmoid(pred[:, :, :2]) + self.grid_cell) * self.stride
        pred[:, :, 2:] = (torch.exp(pred[:, :, 2:])) * self.stride

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
        
        return output

    def forward(self, x, target=None):
        # backbone
        c2, c3, c4, c5 = self.backbone(x)
        B = c2.size(0)

        # FPN deconv
        c5 = self.smooth5(c5)
        c4 = self.smooth4(c4 + self.deconv5(c5))
        c3 = self.smooth3(c3 + self.deconv4(c4))
        c2 = self.smooth2(c2 + self.deconv3(c3))

        # head
        conf_pred = self.conf_pred(c2)
        txty_pred = self.txty_pred(c2)
        twth_pred = self.twth_pred(c2)

        # train
        if self.trainable:
            # [B, H*W, num_classes]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, H*W, 2]
            txty_pred = txty_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
            # [B, H*W, 2]
            twth_pred = twth_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
            # [B, H*W, 4]
            txtytwth_pred = torch.cat([txty_pred, twth_pred], dim=2)


            # compute loss
            conf_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_txtytwth=txtytwth_pred, label=target, version='CenterYOLAF')

            return conf_loss, txtytwth_loss, total_loss       

        # test
        else:
            with torch.no_grad():
                txtytwth_pred = torch.cat([txty_pred, twth_pred], dim=1)

                # decode
                conf_pred = torch.sigmoid(conf_pred[:, :1, :, :])
                txtytwth_pred = txtytwth_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                # simple nms
                hmax = F.max_pool2d(conf_pred, 5, stride=1, padding=2)
                keep = (hmax == conf_pred).float()
                conf_pred *= keep
                # threshold
                conf_pred *= (conf_pred >= self.conf_thresh).float()
                # [B, C, H, W] -> [H, W]
                score = conf_pred[0, 0, :, :] 
                # top K
                topk_scores, topk_inds = torch.topk(score.view(-1), self.topk)
                # decode bbox
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_bbox = all_bbox[topk_inds]

                # separate box pred and class conf
                scores = topk_scores.cpu().numpy()
                bboxes = all_bbox.cpu().numpy()

                return bboxes, scores