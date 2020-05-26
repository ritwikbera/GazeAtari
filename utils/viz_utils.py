import torch
import torch.nn as nn
from glob import glob 
import re
import numpy as np 
from .train_utils import *
from .build_commands import *
from .data_utils import *

def viz_param(writer, model, global_step=None, layer='gc'):
    for param in model.named_parameters():
        if re.search(layer+'.*weight',param[0]):
            # print(param, param[1].numel())
            writer.add_histogram(param[0], param[1].cpu().detach().squeeze().numpy(), global_step=global_step,\
                bins='auto')


def visualize_outputs(writer, state, task):

    item, preds, outputs = state.output 
    b, h, w = item.size(0), item.size(1), item.size(2)
    action_dict = build_commands()

    i = state.iteration

    if task == 'gazepred':
    
        outputs = list(denormalize(outputs, xlim=w, ylim=h).cpu().detach().numpy())
        preds = list(denormalize(preds, xlim=w, ylim=h).cpu().detach().numpy())

        for j, out in enumerate(outputs):
            gazemap = np.zeros_like(item.cpu().detach().numpy()[0])[:,:,0]
            img_1 = gazemap_to_heatmap(create_gazemap(gazemap, preds[j])[0])

            gazemap = np.zeros_like(item.cpu().detach().numpy()[0])[:,:,0]
            img_2 = gazemap_to_heatmap(create_gazemap(gazemap, out)[0])

            writer.add_image('prediction', img_1, (i-1)*b+(j+1), dataformats='HWC')
            writer.add_image('ground_truth', img_2, (i-1)*b+(j+1), dataformats='HWC')

    elif task == 'actionpred':
        preds = np.argmax(preds.cpu().detach().numpy(), axis=1)
        outputs = outputs.cpu().detach().numpy()  # outputs already converted from one-hot to labels
        print(action_dict[preds[0]], action_dict[outputs[0]])
    else:
        pass