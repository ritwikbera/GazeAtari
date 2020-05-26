import torch
from torch import nn
from torch.optim import Adam, Rprop
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import json 
import os
from math import ceil

from ignite.engine import Events, Engine
from ignite.metrics import Loss, RunningAverage, Accuracy, MeanSquaredError
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint
# from ignite.contrib.handlers.tqdm_logger import ProgressBar

from tqdm import tqdm

import models
import utils
from models import *
from utils import *

current_dir = os.getcwd()
torch.manual_seed(0)
np.random.seed(0)

def run(config):
    train_loader = get_instance(utils, 'dataloader', config, 'train')
    val_loader = get_instance(utils, 'dataloader', config, 'val')

    model = get_instance(models, 'arch', config)

    model = init_model(model, train_loader)
    model, device = ModelPrepper(model, config).out

    loss_fn = get_instance(nn, 'loss_fn', config)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)

    writer = create_summary_writer(config, model, train_loader)
    batch_size = config['dataloader']['args']['batch_size']

    if config['mode'] == 'eval' or config['resume']:
        model.load_state_dict(torch.load(config['ckpt_path']))

    epoch_length = int(ceil(len(train_loader)/batch_size))
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=epoch_length, desc=desc.format(0))

    def process_batch(engine, batch):
        inputs, outputs = func(batch)
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, outputs.to(device))
        
        a = list(model.parameters())[0].clone()
        
        loss.backward()
        optimizer.step() 
        
        # check if training is happening
        b = list(model.parameters())[0].clone()
        try:
            assert not torch.allclose(a.data, b.data), 'Model not updating anymore'
        except AssertionError:
            plot_grad_flow(model.named_parameters())

        return loss.item()

    def predict_on_batch(engine, batch):
        inputs, outputs = func(batch)
        model.eval()
        with torch.no_grad():
            y_pred = model(inputs)

        return inputs, y_pred, outputs.to(device)

    trainer = Engine(process_batch)
    trainer.logger = setup_logger("trainer")
    evaluator = Engine(predict_on_batch)
    evaluator.logger = setup_logger("evaluator")

    if config['task'] == 'actionpred':
        Accuracy(output_transform=lambda x: (x[1], x[2])).attach(evaluator, 'val_acc')

    if config['task'] == 'gazepred':
        MeanSquaredError(output_transform=lambda x: (x[1], x[2])).attach(evaluator, 'val_MSE')

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
        pbar.update(1)
        writer.add_scalar('training/avg_loss', engine.state.metrics['loss'] ,engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_trainer_logs(engine):
        pbar.refresh()
        
        avg_loss = engine.state.metrics['loss']
        tqdm.write('Trainer Results - Epoch {} - Avg loss: {:.2f} \n'.format(engine.state.epoch, avg_loss))
        viz_param(writer=writer, model=model, global_step=engine.state.epoch)

        pbar.n = pbar.last_print_n = 0


    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_result(engine):
        try:
            print('Evaluator Results - Accuracy {} \n'.format(engine.state.metrics['val_acc']))
        except KeyError:
            print('Evaluator Results - MSE {} \n'.format(engine.state.metrics['val_MSE']))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def viz_outputs(engine):    
        visualize_outputs(writer=writer, state=engine.state, task=config['task'])

    if config['mode'] == 'train':
        trainer.run(train_loader, max_epochs=config['epochs'], epoch_length=epoch_length)
    
    
    pbar.close()
    
    evaluator.run(val_loader, max_epochs=1, epoch_length=int(ceil(len(val_loader)/batch_size)))


    writer.flush()
    writer.close()
    
if __name__ == "__main__":
    config = json.load(open('config.json'))
    run(config)