from models import *
from utils import *

# a = torch.randn(1,210,160,12)
# model = GazePred()

# model = AtariPolicy()
# a = torch.randn(1,210,160,3)
# a = a.permute(0,3,1,2)
# a = (a,a)

# print(model(a).size())

a = AtariMSDL('dataset/alien/overfit',4)
print(len(a))
batch = next(iter(a))
print(batch[0].size(), batch[1].size(), batch[2].size())
