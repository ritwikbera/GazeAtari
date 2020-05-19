from .graphLayer import GCU
from .convfc import convFC

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import resnet18

class GazePred(nn.Module):

    def __init__(self, V=8, outfeatures=12, env='atari'): # sample batch BCHW
        super(GazePred, self).__init__()
        self.layers_set = False
        self.V = V
        self.outfeatures = outfeatures
        self.env = env

        if self.env == 'airsim':
            self.resnet = resnet18(pretrained=True)

    def set_layers(self, sample_batch):
        # self.conv1 = nn.Conv2d(in_channels=sample_batch.size(1), out_channels=16, kernel_size=8, stride=4, padding=2)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        
        bs,h,w,ch = sample_batch.size()
        self.gcu = GCU(V=self.V, h=h, w=w, d=ch, batch=bs, outfeatures=self.outfeatures)

        output_dim = 2

        self.conv = nn.Conv2d(in_channels=self.outfeatures, out_channels=output_dim, kernel_size=3, padding=(1,1))
        self.conv_fc = convFC(dims=[bs,ch,h,w], num_outputs=output_dim)

      
    def forward(self, x):
        if not self.layers_set:
            self.set_layers(x)
            self.layers_set = True

        if self.env == 'airsim':
            x = resnet(x.permute(0,2,3,1)).permute(0,3,1,2) # keep x in (B,H,W,C), due to PIL processing it like that

        x = (x-127.5)/127.5
        y = (self.gcu(x) + x).permute(0,3,1,2) # GCU and input x operate in (B,H,W,C)
        y = self.conv_fc(self.conv(y))

        return y
