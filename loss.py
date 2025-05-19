"""
 @Time    : 2021/7/6 14:31
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : loss.py
 @Function: Loss
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.metrics import SSIM
from pytorch_msssim import SSIM
import py_sod_metrics
###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)




class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1.  # 平滑项，避免分母为零
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))    #(pred + target).sum(dim=(2, 3)) - intersection
        dice_coefficient = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_coefficient
        return dice_loss.mean()



class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):        
        pred = torch.sigmoid(pred)
        #ssim=SSIM()        
        ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        ssim_loss = 1 - ssim(pred,target)
        return ssim_loss.mean()


    
class objectLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(objectLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        """
        Calculate the object score.
        """
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = gt.mean()
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        object_loss=1-object_score
        return object_loss

    def s_object(self, pred, gt):
        x = pred[gt == 1].mean()
        sigma_x = pred[gt == 1].std(unbiased=True)
        score = 2 * x / (x**2 + 1 + sigma_x + self.epsilon)
        return score
###################################################################
# #################### structure loss #############################
###################################################################
class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)
