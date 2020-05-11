import os
import math
import json
import logging
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


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