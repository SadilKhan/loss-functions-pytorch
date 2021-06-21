import torch
from torch import nn

class GIoU_Loss(nn.Module):
    """ Generalized Intersection Over Union Loss.
    Original Paper: https://arxiv.org/pdf/1902.09630.pdf
    By: Mohammad Sadil Khan
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


if __name__== "__main__":
    giou=GIoU_Loss()
    model = nn.Linear(4, 4,bias=False)
    x = torch.randn(1, 4)
    target = torch.randn(1, 4)
    output = model(x)
    loss = giou(output, target)
    loss.backward()
    print(model.weight.grad)