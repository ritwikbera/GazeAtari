import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class convFC(nn.Module):
    def __init__(self, y, num_outputs=512):
        super(convFC, self).__init__()
        b,c,h,w = y.size()
        self.conv_fc1 = nn.Conv2d(in_channels=h, out_channels=num_outputs//c, kernel_size=1) 
        self.conv_fc2 = nn.Conv2d(in_channels=w, out_channels=1, kernel_size=1)

    def forward(self, y):
        y = y.rename('B','C','H','W')
        y = self.conv_fc1(y.align_to('B','H','C','W').rename(None)) 
        y = y.rename('B','H','C','W')
        y = self.conv_fc2(y.align_to('B','W','H','C').rename(None))
        y = y.view(y.size()[0],-1)

        return y

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

        self.fc = nn.Linear(512,18)

    def forward(self, X):
        x, y = X
        y = torch.mul(x,y)
        x = self.conv3_1(self.conv2_1(self.conv1_1(x)))
        y = self.conv3_2(self.conv2_2(self.conv1_2(y)))
        z = (x+y)/2.0

        if not self.conv_fc:
            self.conv_fc = convFC(y)
        z = F.softmax(self.fc(self.conv_fc(z)),dim=-1)

        return z

if __name__=='__main__':
    h,w =84,84
    h,w =210,160
    img = torch.randn(1,3,h,w)
    gaze = torch.randn(1,1,h,w)
    model = AtariPolicy()
    print(model((img, gaze)).size())