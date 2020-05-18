import torch.nn as nn
from .graphLayer import GCU
import torch


class GazePred(nn.Module):

    def __init__(self, V=8, outfeatures=12): # sample batch BCHW
        super(GazePred, self).__init__()
        self.layers_set = False
        self.V = V
        self.outfeatures = outfeatures

    def set_layers(self, sample_batch):
        # self.conv1 = nn.Conv2d(in_channels=sample_batch.size(1), out_channels=16, kernel_size=8, stride=4, padding=2)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        
        bs,h,w,ch = sample_batch.size()
        self.gcu = GCU(V=self.V, h=h, w=w, d=ch, batch=bs, outfeatures=self.outfeatures)

        output_dim = 2

        self.conv = nn.Conv2d(in_channels=self.outfeatures, out_channels=output_dim, kernel_size=3, padding=(1,1))
        self.conv_fc1 = nn.Conv2d(in_channels=h, out_channels=1, kernel_size=1) 
        self.conv_fc2 = nn.Conv2d(in_channels=w, out_channels=1, kernel_size=1)

      
    def forward(self, x):
        if not self.layers_set:
            self.set_layers(x)
            self.layers_set = True
        x = (x-127.5)/127.5
        y = self.gcu(x)
        y = y + x.permute(0,3,1,2) # residual connection
        y = self.conv(y)
        y = y.rename('B','C','H','W')
        # FC layers fully replaced by 2 fully conv layers
        y = self.conv_fc1(y.align_to('B','H','C','W').rename(None)) 
        y = y.rename('B','H','C','W')
        y = self.conv_fc2(y.align_to('B','W','H','C').rename(None))
        y = y.view(y.size()[0],-1)
        return y

if __name__=='__main__':
    img = torch.randn(2,16,16,3*4)
    model = GazePred()
    out = model(img)
    print(out.size())