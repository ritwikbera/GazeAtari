import os
import math
import json
import logging
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


def normalize(data, xlim, ylim):
    return (data - torch.Tensor([[xlim/2,ylim/2]]))/torch.Tensor([xlim,ylim])

def denormalize(data, xlim, ylim):
    return (data * torch.Tensor([[xlim,ylim]]))+torch.Tensor([xlim/2,ylim/2])

def depickle(batch, config):
    frames = torch.cat(list(map(lambda x: torch.load(x)['frame_stack'], batch)),0)
    gaze_data = torch.cat(list(map(lambda x: torch.load(x)['gaze_point'], batch)),0)
    actions = torch.cat(list(map(lambda x: torch.load(x)['action'], batch)),0)
    h,w = frames.size(1), frames.size(2)
    gaze_data = normalize(gaze_data, xlim=w, ylim=h)

    if config['task'] == 'gazepred':
        return frames, gaze_data
    else:
        return frames, actions

def init_model(model, train_loader):
    batch_ = next(iter(train_loader))
    frames_ = torch.cat(list(map(lambda x: torch.load(x)['frame_stack'], batch_)),0)
    _ = model(frames_)
    return model

def create_summary_writer(config, model, data_loader):
    writer = SummaryWriter(log_dir=config['log_dir'])

    x, y = depickle(next(iter(data_loader)), config)

    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1
        
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ModelPrepper:
    def __init__(self, model, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    def _prepare_device(self, n_gpu_use):
        """ 
        setup GPU device if available, move model into configured device
        """ 
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    @property
    def out(self):
        return self.model, self.device