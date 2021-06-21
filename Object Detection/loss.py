import torch
import torch.nn as nn
import math
from PIL import Image


class EIoU_Loss(nn.Module):
    """ Efficient IOU Loss.
    Original Paper: https://arxiv.org/abs/2101.08158.
    Code By: Mohammad Sadil Khan"""
    
    def __init__(self):
      super(EIoU_Loss,self).__init__()
     
    def forward(self,pred,true):
      pred=torch.clamp(pred,min=0)
      true=torch.clamp(true,min=0)

      rows = pred.shape[0]
      cols = true.shape[0]
      dious = torch.zeros((rows, cols))
      if rows * cols == 0:
          return dious
      exchange = False
      if pred.shape[0] > true.shape[0]:
          pred, true = true, pred
          dious = torch.zeros((cols, rows))
          exchange = True

      w1 = pred[:, 2] - pred[:, 0]
      h1 = pred[:, 3] - pred[:, 1]
      w2 = true[:, 2] - true[:, 0]
      h2 = true[:, 3] - true[:, 1]

      area1 = w1 * h1
      area2 = w2 * h2
      center_x1 = (pred[:, 2] + pred[:, 0]) / 2
      center_y1 = (pred[:, 3] + pred[:, 1]) / 2
      center_x2 = (true[:, 2] + true[:, 0]) / 2
      center_y2 = (true[:, 3] + true[:, 1]) / 2

      inter_max_xy = torch.min(pred[:, 2:],true[:, 2:])
      inter_min_xy = torch.max(pred[:, :2],true[:, :2])
      
      # Bottom right corner of the enclosing box
      out_max_xy = torch.max(pred[:, 2:],true[:, 2:]) 
      # Top left corner of the enclosing box
      out_min_xy = torch.min(pred[:, :2],true[:, :2]) 
      # Width of the Smallest enclosing box
      C_w=(out_max_xy[:,0]-out_min_xy[:,0])
      # Height of the smallest enclosing box
      C_h=(out_max_xy[:,1]-out_min_xy[:,1])

      inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
      inter_area = inter[:, 0] * inter[:, 1]
      inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
      outer = torch.clamp((out_max_xy - out_min_xy), min=0)
      outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
      union = area1+area2-inter_area
      dious = inter_area / union - (inter_diag) / outer_diag
      asp= torch.clamp((w2-w1)**2,min=0)/(C_w**2) + torch.clamp((h2-h1)**2,min=0)/(C_h**2)
      eiou=dious-asp
      eiou_loss=torch.mean(1-eiou)
      return eiou_loss
  
  
  class GIoU_Loss(nn.Module):
    """ Generalized Intersection Over Union Loss.
    Original Paper: https://arxiv.org/pdf/1902.09630.pdf
    Code By: Mohammad Sadil Khan
    """

    def __init__(self):
        super(GIoU_Loss,self).__init__()
    
    def forward(self,pred,true):

        # Find Area of the Intersection
        #pred=pred.type(torch.float32).clone().detach().requires_grad_(True)
        ints_x_min=torch.max(true[:,0],pred[:,0])
        ints_y_min=torch.max(true[:,1],pred[:,1])
        ints_x_max=torch.min(true[:,2],pred[:,2])
        ints_y_max=torch.min(true[:,3],pred[:,3])

        width=torch.max((ints_x_max-ints_x_min),torch.tensor([0]).unsqueeze(0))
        height=torch.max((ints_y_max-ints_y_min),torch.tensor([0]).unsqueeze(0))

        area_intersection=torch.max(width*height,
        torch.tensor([0]).unsqueeze(0))

        # Find Area of the Box True
        area_true=torch.mul((true[:,2]-true[:,0]),(true[:,3]-true[:,1]))

        # Find Area of the Box Pred
        area_pred=torch.mul((pred[:,2]-pred[:,0]),(pred[:,3]-pred[:,1]))

        # Find Area of the Union
        area_union=area_true+area_pred-area_intersection

        # Find Area of the Smallest Enclosing Box
        box_x_min=torch.min(true[:,0],pred[:,0])
        box_y_min=torch.min(true[:,1],pred[:,1])
        box_x_max=torch.max(true[:,2],pred[:,2])
        box_y_max=torch.max(true[:,3],pred[:,3])

        area_c=(box_x_max-box_x_min)*(box_y_max-box_y_min)

        # Calculate IOU
        iou=area_intersection/area_union
        #iou.requires_grad=True

        # Calculate GIOU
        giou=iou-(area_c-area_union)/area_c
        #giou.requires_grad=True

        # Calculate Loss
        giou_loss=torch.mean(1-giou)
        #giou_loss.requires_grad=True
 
        return giou_loss
