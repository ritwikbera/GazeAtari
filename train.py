import torch
from torch import nn
from torch.optim import Adam, Rprop
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import json 
import os

from ignite.engine import Events, Engine
from ignite.metrics import Loss, RunningAverage
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint

from tqdm import tqdm

import models
import utils
from models import *
from utils import *

current_dir = os.getcwd()
torch.manual_seed(0)
np.random.seed(0)

def run(config):
    train_loader = get_instance(utils, 'dataloader', config)
    model = get_instance(models, 'arch', config)

    model = init_model(model, train_loader)
    model, device = ModelPrepper(model, config).out

    loss_fn = getattr(nn,config['loss_fn']['type'])()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)

    writer = create_summary_writer(config, model, train_loader)
    batch_size = config['dataloader']['args']['batch_size']

    if config['mode'] == 'eval' or config['resume'] == 1:
        model.load_state_dict(torch.load(config['ckpt_path']))

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))

    def process_batch(engine, batch):
        inputs, outputs = func(batch)

        if config['mode'] in ['train','overfit']:
            model.train()
        else:
            model.eval()

        preds = model(inputs)
        loss = loss_fn(outputs.to(device), preds)*100

        if config['mode'] in ['train','overfit']:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        return loss.item()

    trainer = Engine(process_batch)
    trainer.logger = setup_logger("trainer")

    if config['mode'] in ['train','overfit']:
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
        training_saver = ModelCheckpoint(config['checkpoint_dir'],
                                     filename_prefix='checkpoint_'+config['task'],
                                     n_saved=1,
                                     atomic=True,
                                     save_as_state_dict=True,
                                     create_dir=True, require_empty=False)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, training_saver, 
                              {'model': model})

    @trainer.on(Events.ITERATION_COMPLETED)
    def tb_log(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(batch_size)

        if config['mode'] in ['train','overfit']:
            writer.add_scalar('training/avg_loss', engine.state.metrics['loss'] ,engine.state.iteration)
        else:
            writer.add_scalar('testing/avg_loss', engine.state.output[0] ,engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_trainer_logs(engine):
        pbar.refresh()
        avg_loss = engine.state.metrics['loss']
        tqdm.write('Trainer Results - Epoch {} - Avg loss: {:.2f}'.format(engine.state.epoch, avg_loss))
        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=config['epochs'], epoch_length=len(train_loader)/batch_size)
    pbar.close()
    
if __name__ == "__main__":
    config = json.load(open('config.json'))
    run(config)