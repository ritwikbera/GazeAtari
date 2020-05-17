import torch
import torch.nn as nn
from glob import glob 
import re
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np 

from dataloader import *
from model import *
from utils import *
from data_utils import gazemap_to_heatmap, create_gazemap

# batch size 1 for evaluation
config = {'game':'alien', 'batch_size':1, 'mode':'overfit', 'task':'gazepred'}

model = GazePred()
train_loader, size = get_loader(config)
model = init_model(model, train_loader)
writer = create_summary_writer(config, model, train_loader)

# cp_path = 'checkpoint/trial.pth'
cp_path = 'checkpoint/checkpoint_model_800.pth'
model.load_state_dict(torch.load(cp_path))
model.eval()

def print_param(model):
	for param in model.named_parameters():
		if re.search('gc.*weight',param[0]):
			print(param, param[1].numel())
			writer.add_histogram(param[0], param[1].detach().squeeze().numpy(), bins='auto')
# print(list(model.parameters()))

print_param(model)



for i, batch in enumerate(train_loader):
	# if i%1000 > 0:
		# continue
	if i > size: # need to include this for an interative dataloader
		break
	item, outputs = depickle(batch, config)
	preds = model(item)
	h, w = item.size(1), item.size(2)

	assert config['task'] == 'gazepred'
	
	outputs = list(denormalize(outputs, xlim=w, ylim=h).detach().numpy())
	preds = list(denormalize(preds, xlim=w, ylim=h).detach().numpy())

	for j, out in enumerate(outputs):
		gazemap = np.zeros_like(item.detach().numpy()[0])[:,:,0]
		img_1 = gazemap_to_heatmap(create_gazemap(gazemap, preds[j])[0])

		gazemap = np.zeros_like(item.detach().numpy()[0])[:,:,0]
		img_2 = gazemap_to_heatmap(create_gazemap(gazemap, out)[0])

		writer.add_image('prediction', img_1, i*1+j, dataformats='HWC')
		writer.add_image('ground_truth', img_2, i*1+j, dataformats='HWC')

writer.flush()
writer.close()