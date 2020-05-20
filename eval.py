import torch
import torch.nn as nn
from glob import glob 
import re
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np 
import json

import models
import utils
from models import *
from utils import *

config = json.load(open('config.json'))
config['dataloader']['args']['batch_size'] = 1

train_loader = get_instance(utils, 'dataloader', config)
model = get_instance(models, 'arch', config)

model = init_model(model, train_loader)

writer = create_summary_writer(config, model, train_loader)
batch_size = config['dataloader']['args']['batch_size']

if config['mode'] == 'eval' or config['resume'] == 1:
    model.load_state_dict(torch.load(config['ckpt_path']))

model.eval()

def print_param(model):
	for param in model.named_parameters():
		if re.search('gc.*weight',param[0]):
			print(param, param[1].numel())
			writer.add_histogram(param[0], param[1].detach().squeeze().numpy(), bins='auto')

# print_param(model)



for i, batch in enumerate(train_loader):
	# if i%1000 > 0:
		# continue
	if i > len(train_loader): # need to include this for an interative dataloader
		break
	item, outputs = func(batch)
	preds = model(item)
	h, w = item.size(1), item.size(2)
	action_dict = build_commands()

	if config['task'] == 'gazepred':
	
		outputs = list(denormalize(outputs, xlim=w, ylim=h).detach().numpy())
		preds = list(denormalize(preds, xlim=w, ylim=h).detach().numpy())

		for j, out in enumerate(outputs):
			gazemap = np.zeros_like(item.detach().numpy()[0])[:,:,0]
			img_1 = gazemap_to_heatmap(create_gazemap(gazemap, preds[j])[0])

			gazemap = np.zeros_like(item.detach().numpy()[0])[:,:,0]
			img_2 = gazemap_to_heatmap(create_gazemap(gazemap, out)[0])

			writer.add_image('prediction', img_1, i*1+j, dataformats='HWC')
			writer.add_image('ground_truth', img_2, i*1+j, dataformats='HWC')

	elif config['task'] == 'actionpred':
		preds = np.argmax(preds.detach().numpy(), axis=1)
		outputs = outputs.detach().numpy()  # outputs already converted from one-hot to labels
		print(action_dict[preds[0]], action_dict[outputs[0]])
	else:
		pass

writer.flush()
writer.close()