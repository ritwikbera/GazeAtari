import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import json 

from ignite.engine import Events, Engine
from ignite.metrics import Loss, RunningAverage
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint

from tqdm import tqdm
from model import *
from dataloader import *
from utils import *

torch.manual_seed(0)
np.random.seed(0)

def run(config):
    path = 'dataset/'+config['game']+'/data/*'
    log_dir = 'dataset/'+config['game']+'/logs/'
    writer = SummaryWriter(log_dir=log_dir)

    data_list = [glob(trial+'/*.pth') for trial in glob(path)[:]]
    datasets = MyIterableDataset.split_datasets(data_list, batch_size=config['batch_size'], max_workers=1)
    train_loader = MultiStreamDataLoader(datasets)

    model = GazePred()
    loss_fn = nn.MSELoss()

    batch_ = next(iter(train_loader)) 
    frames_ = torch.cat(list(map(lambda x: torch.load(x)['frame_stack'], batch_)),0)
    _ = model(frames_)

    model = ModelPrepper(model, config).out
    optimizer = Adam(model.parameters(), lr=config['lr'])


    def process_batch(engine, batch):
        frames = torch.cat(list(map(lambda x: torch.load(x)['frame_stack'], batch)),0)
        outputs = torch.cat(list(map(lambda x: torch.load(x)['gaze_point'], batch)),0)
        h,w = 210,160
        outputs = (outputs - torch.Tensor([h/2,w/2])[None,:])/torch.Tensor([h,w])[None,:]

        preds = model(frames)
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(outputs, preds)
        loss.backward()
        optimizer.step()

        # print(frames.size(), outputs.size())
        # print(pred.size())

        # print(preds, outputs)
        # print(loss.item())

        return loss.item()

    trainer = Engine(process_batch)
    trainer.logger = setup_logger("trainer")

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    training_saver = ModelCheckpoint("checkpoint",
                                 filename_prefix="checkpoint",
                                 n_saved=1,
                                 atomic=True,
                                 save_as_state_dict=True,
                                 create_dir=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, training_saver, 
                          {'model': model})

    @trainer.on(Events.ITERATION_COMPLETED)
    def tb_log(engine):
        print(engine.state.metrics['loss'])
        writer.add_scalar('training/avg_loss', engine.state.metrics['loss'] ,engine.state.iteration)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_trainer_logs(engine):
        avg_loss = engine.state.metrics['loss']
        print('Trainer Results - Epoch {} - Avg loss: {:.2f}'.format(engine.state.epoch, avg_loss))
    
    trainer.run(train_loader, max_epochs=config['epochs'], epoch_length=100)

if __name__ == "__main__":
    config = {'game':'alien', 'batch_size':4, 'epochs':40, 'lr':5e-3, 'log_interval':100, 'n_gpu':2}
    run(config)