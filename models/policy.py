from .convfc import *

import torch
import torch.nn as nn 
import torch.nn.functional as F

class AtariPolicy(nn.Module):
    def __init__(self):
        super(AtariPolicy,self).__init__()
        self.conv1_1 = nn.Conv2d(3,32,(8,8),stride=4,padding=0)
        self.conv2_1 = nn.Conv2d(32,64,(4,4),stride=2,padding=0)
        self.conv3_1 = nn.Conv2d(64,64,(3,3),stride=1,padding=0)

        self.conv1_2 = nn.Conv2d(3,32,(8,8),stride=4,padding=0)
        self.conv2_2 = nn.Conv2d(32,64,(4,4),stride=2,padding=0)
        self.conv3_2 = nn.Conv2d(64,64,(3,3),stride=1,padding=0)

        self.conv_fc = None

        self.conv_scores = nn.Conv2d(64,18,(1,1))

    def forward(self, X):
        x, y = X
        y = torch.mul(x,y)
        x = self.conv3_1(self.conv2_1(self.conv1_1(x)))
        y = self.conv3_2(self.conv2_2(self.conv1_2(y)))
        z = (x+y)/2.0

        if not self.conv_fc:
            self.conv_fc = convFC(y.size())
        z = self.conv_scores(self.conv_fc(z)[:,:,None,None])
        z = F.softmax(z, dim=1).view(z.size(0),-1)

        return z
