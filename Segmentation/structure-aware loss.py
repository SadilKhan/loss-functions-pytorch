import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# For point cloud segmentation with batch 1

class SALoss(nn.Module):
    def __init__(self,alpha=0.7,beta=1.5,weight=None):
        super().__init__()
        self.alpha=0.7
        self.beta=1.5
        self.weight=weight

    def forward(self,points,true,embedding):
        """
        points: shape (B,N,3)
        true: shape (B,N,C)
        embedding: shape (B,N,k)
        
        """
        M=len(torch.unique(true)) # Number of Labels
        N=points.size(1) # Number of points
        intraLoss=0
        interLoss=0
        mean_emb=[] # Mean embeddings for organs

        # Calculate Intra Loss
        for i in range(1,M):
            pos=(true==i).nonzero()[:,1]
            emb_i=embedding[:,pos] # (B,N1,k)
            point_i=points[:,pos]
            g=torch.sigmoid(torch.sum(point_i**2,dim=-1)**0.5) # points are already normalized
            mean_emb_i=torch.mean(embedding[:,pos],dim=1) # (B,k)
            mean_emb.append(mean_emb_i)
            if self.weight:
                intraLoss+=torch.mean(self.weight[i]*g*torch.square(torch.clamp(torch.sum((emb_i-mean_emb_i)**2,dim=-1)**0.5-self.alpha,min=0)))
            else:
                intraLoss+=torch.mean(g*torch.square(torch.clamp(torch.sum((emb_i-mean_emb_i)**2,dim=-1)**0.5-self.alpha,min=0)))
        mean_emb=torch.stack(mean_emb,dim=1)

        # Calculate Inter Loss
        for i in range(1,M):
            for j in range(1,M):
                if (i!=j):
                    interLoss+=torch.square(torch.clamp(self.beta-torch.sum((mean_emb[:,i-1]-mean_emb[:,j-1])**2,dim=-1)**0.5,min=0))
        
        return intraLoss/M+interLoss/(M*(M-1))
