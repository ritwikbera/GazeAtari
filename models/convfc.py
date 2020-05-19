import torch
import torch.nn as nn 
import torch.nn.functional as F

class convFC(nn.Module):
    def __init__(self, dims, num_outputs=512):
        super(convFC, self).__init__()
        h,w = dims[2],dims[3]
        self.conv_fc1 = nn.Conv2d(in_channels=h, out_channels=1, kernel_size=1) 
        self.conv_fc2 = nn.Conv2d(in_channels=w, out_channels=1, kernel_size=1)

    def forward(self, y):
        y = y.rename('B','C','H','W')
        y = self.conv_fc1(y.align_to('B','H','C','W').rename(None)) 
        y = y.rename('B','H','C','W')
        y = self.conv_fc2(y.align_to('B','W','H','C').rename(None))
        y = y.view(y.size()[0],-1)

        return y