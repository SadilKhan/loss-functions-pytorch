import os
import torch
import math
import torch.nn as nn
import torchvision


def calculate_iou(pred,true):
	"""Functions to calculate IoU """
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

	# Calculate IoU
	iou=area_intersection/area_union

	return iou,area_intersection,area_union


class IoULoss(nn.Module):
    """Intersection over Union Loss"""

    def __init__(self,losstype="giou"):
		super(IoULoss, self).__init__()
		"""losstype --> str. Type of IoU based Loss. "iou","giou","diou","ciou","eiou" are available """
		self.losstype =losstype

    def forward(self, pred, true):
		pred=torch.clamp(pred,min=0)
		true=torch.clamp(true,min=0)

		if self.losstype == "iou":
			loss=torch.mean(1-calculate_iou(pred,true)[0])

		elif self.losstype == "giou":
			l_giou=1-calculate_iou(pred,true)[0]+self.penalty_giou(pred,true)
			loss=torch.mean(l_giou)

		elif self.losstype == "diou":
			l_diou=1-calculate_iou(pred,true)[0]+self.penalty_diou(pred,true)
			loss=torch.mean(l_diou)

		elif self.losstype == "ciou":
			l_ciou=1-calculate_iou(pred,true)[0]+self.penalty_ciou(pred,true)
			loss=torch.mean(l_ciou)

		elif self.losstype == "eiou":
			l_eiou=1-calculate_iou(pred,true)[0]+self.penalty_eiou(pred,true)
			loss=torch.mean(l_eiou)

		return loss
    
    def penalty_giou(self, pred, true):
		# Find Area of the Smallest Enclosing Box
		box_x_min=torch.min(true[:,0],pred[:,0])
		box_y_min=torch.min(true[:,1],pred[:,1])
		box_x_max=torch.max(true[:,2],pred[:,2])
		box_y_max=torch.max(true[:,3],pred[:,3])

		area_c=(box_x_max-box_x_min)*(box_y_max-box_y_min)

		return (area_c-calculate_iou(pred,true)[2])/area_c
    def penalty_diou(self, pred, true):
		# Center point of the predicted bounding box
		center_x1 = (pred[:, 2] + pred[:, 0]) / 2
		center_y1 = (pred[:, 3] + pred[:, 1]) / 2

		# Center Point of the ground truth box
		center_x2 = (true[:, 2] + true[:, 0]) / 2
		center_y2 = (true[:, 3] + true[:, 1]) / 2
		inter_max_xy = torch.min(pred[:, 2:],true[:, 2:])
		inter_min_xy = torch.max(pred[:, :2],true[:, :2])
		
		# Bottom right corner of the enclosing box
		out_max_xy = torch.max(pred[:, 2:],true[:, 2:]) 
		# Top left corner of the enclosing box
		out_min_xy = torch.min(pred[:, :2],true[:, :2]) 
			
			# Distance between the center points of the ground truth and the predicted box
		inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
		outer = torch.clamp((out_max_xy - out_min_xy), min=0)
		outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)

		return inter_diag/outer_diag


    def penalty_ciou(self, pred, true):
		w1 = pred[:, 2] - pred[:, 0]
		h1 = pred[:, 3] - pred[:, 1]
		w2 = true[:, 2] - true[:, 0]
		h2 = true[:, 3] - true[:, 1]
		v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
		with torch.no_grad():
			S = 1 - calculate_iou(pred,true)[0]
			alpha = v / (S + v)
		
		return self.penalty_diou(pred,true)+alpha*v
        
    def penalty_eiou(self, pred, true):
		w1 = pred[:, 2] - pred[:, 0]
		h1 = pred[:, 3] - pred[:, 1]
		w2 = true[:, 2] - true[:, 0]
		h2 = true[:, 3] - true[:, 1]

		# Bottom right corner of the enclosing box
		out_max_xy = torch.max(pred[:, 2:],true[:, 2:]) 
		# Top left corner of the enclosing box
		out_min_xy = torch.min(pred[:, :2],true[:, :2]) 
		# Width of the Smallest enclosing box
		C_w=(out_max_xy[:,0]-out_min_xy[:,0])
		# Height of the smallest enclosing box
		C_h=(out_max_xy[:,1]-out_min_xy[:,1])

		asp= torch.clamp((w2-w1)**2,min=0)/(C_w**2) + torch.clamp((h2-h1)**2,min=0)/(C_h**2)

		return self.penalty_diou(pred,true)+asp
