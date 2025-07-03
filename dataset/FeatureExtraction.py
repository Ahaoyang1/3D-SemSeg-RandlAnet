
import torch
import torch.nn as nn
import torch.nn.functional as F
import network.pytorch_utils as pt_utils



def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
    # gather the coordinates or features of neighboring points
    batch_size = pc.shape[0]
    num_points = pc.shape[1]
    d = pc.shape[2]
    index_input = neighbor_idx.reshape(batch_size, -1)
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
    features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
    return features

class FeatureExtraction(nn.Module):
    def __int__(self,d_in,d_out):
        super(FeatureExtraction, self).__int__()
        self.mlp1_2d=pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp1_1d=pt_utils.Conv1d(d_out, d_out, kernel_size=1, bn=True,activation=None)

        self.mlp2_2d = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp2_1d = pt_utils.Conv1d(d_out, d_out, kernel_size=1, bn=True, activation=None)

        self.mlp3_2d = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp3_1d = pt_utils.Conv1d(d_out, d_out, kernel_size=1, bn=True, activation=None)

        self.mlp4_2d = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp4_1d = pt_utils.Conv1d(d_out, d_out, kernel_size=1, bn=True, activation=None)

    def forward(self,x,idx):

        x1=gather_neighbour(x,idx)
        x1=self.mlp1_2d(x1)
        max1=torch.max(x1,dim=-1)
        mean1=torch.mean(x1,dim=-1)
        x1=torch.cat([max1,mean1],dim=-1)
        x1=self.mlp1_1d(x1)

        tmp=F.leaky_relu(x1,negative_slope=0.2)
        x2=gather_neighbour(tmp,idx)
        x2 = self.mlp2_2d(x2)
        max2 = torch.max(x2, dim=-1)
        mean2 = torch.mean(x2, dim=-1)
        x2 = torch.cat([max2, mean2], dim=-1)
        x2 = self.mlp2_1d(x2)

        tmp=F.leaky_relu(x2+x1,negative_slope=0.2)
        x3=gather_neighbour(tmp,idx)
        x3 = self.mlp3_2d(x3)
        max3 = torch.max(x3, dim=-1)
        mean3 = torch.mean(x3, dim=-1)
        x3 = torch.cat([max3, mean3], dim=-1)
        x3 = self.mlp3_1d(x3)

        tmp=F.leaky_relu(x3+x2,negative_slope=0.2)
        x4=gather_neighbour(tmp,idx)
        x4 = self.mlp4_2d(x4)
        max4 = torch.max(x4, dim=-1)
        mean4 = torch.mean(x4, dim=-1)
        x4 = torch.cat([max4, mean4], dim=-1)
        x4 = self.mlp4_1d(x4)

        return F.leaky_relu(x1+x2+x3+x4,negative_slope=0.2)
