import os
import math
import json
import logging
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataloader import *

def depickle(batch):
    frames = torch.cat(list(map(lambda x: torch.load(x)['frame_stack'], batch)),0)
    outputs = torch.cat(list(map(lambda x: torch.load(x)['gaze_point'], batch)),0)
    h,w = 210,160
    outputs = (outputs - torch.Tensor([h/2,w/2])[None,:])/torch.Tensor([h,w])[None,:]

    return frames, outputs

def init_model(train_loader):
    model = GazePred()
    batch_ = next(iter(train_loader)) 
    frames_ = torch.cat(list(map(lambda x: torch.load(x)['frame_stack'], batch_)),0)
    _ = model(frames_)
    return model

def create_summary_writer(config, model, data_loader):
    log_dir = 'dataset/'+config['game']+'/logs/'

    writer = SummaryWriter(log_dir=log_dir)

    x, y = depickle(next(iter(data_loader)))

    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def get_loader(config):
    path = 'dataset/'+config['game']+'/data/*'
    
    # if overfitting on a batch
    if config['mode'] == 'overfit':
        path = 'dataset/'+config['game']+'/sandbox/*'

    if config['mode'] == 'eval':
        config['batch_size'] = 1

    data_list = [glob(trial+'/*.pth') for trial in glob(path)[:]]
    size = recursive_len(data_list)
    print('Dataset Size {}'.format(size))
    
    datasets = MyIterableDataset.split_datasets(data_list, batch_size=config['batch_size'], max_workers=1)
    train_loader = MultiStreamDataLoader(datasets)

    return train_loader, size

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
        return self.model