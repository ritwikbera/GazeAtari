import torch
import torch.nn as nn
from glob import glob 
import re
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 

from dataloader import *
from model import *
from utils import *

# batch size 1 for evaluation
config = {'game':'alien', 'batch_size':1, 'mode':'overfit'}

train_loader, size = get_loader(config)
model = init_model(train_loader)
writer = create_summary_writer(config, model, train_loader)

cp_path = 'trial3.pth'
model.load_state_dict(torch.load(cp_path))
model.eval()

def print_param(model):
	for param in model.named_parameters():
		if re.search('gc.*weight',param[0]):
			print(param, param[1].numel())
			writer.add_histogram(param[0], param[1].detach().squeeze().numpy(), bins='auto')
# print(list(model.parameters()))

for i, batch in enumerate(train_loader):
	# if i%1000 > 0:
		# continue
	if i > size:
		break
	item, _ = depickle(batch)
	print(model(item))
	# writer.add_figure('predictions vs actuals', model(item))

writer.flush()
writer.close()