from .convfc import *
from .gauss import *
from .gaze_model import *
from utils import denormalize

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
        z = z.view(z.size(0),-1)

        return z

class AtariPolicyFull(nn.Module):
    def __init__(self, gaze_model_path=None, V=8, outfeatures=12, env='atari'):
        super(AtariPolicyFull,self).__init__()
        self.policynet = AtariPolicy()
        self.gazenet = GazePred(V=V, outfeatures=outfeatures, env=env)
        self.gaze_model_path = gaze_model_path
        self.gaze_model_loaded = False
        self.gauss = get_gaussian_kernel()

    def forward(self, x):
        if not self.gazenet.layers_set:
            _ = self.gazenet(x)
            assert self.gazenet.layers_set == True

        if self.gaze_model_path is not None and not self.gaze_model_loaded:
            self.gazenet.load_state_dict(torch.load(self.gaze_model_path))
            self.gaze_model_loaded = True
        
        gazemap = torch.zeros(x.size(0),1,x.size(1),x.size(2)) # x is in BHWC format

        with torch.no_grad():
            coords = denormalize(self.gazenet(x), xlim=x.size(2), ylim=x.size(1)).long()
            bselect = torch.arange(x.size(0), dtype=torch.long)
            gazemap[bselect, :, coords[:, 0], coords[:, 1]] = 1.0
            gazemap = self.gauss(gazemap)

        x = x.permute(0,3,1,2)
        return self.policynet((x[:,-3:,:,:],gazemap))


