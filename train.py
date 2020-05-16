import torch
from torch import nn
from torch.optim import Adam, Rprop
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
    loss_fn = nn.MSELoss()

    train_loader, size = get_loader(config)
    model = init_model(train_loader)
    writer = create_summary_writer(config, model, train_loader)


    model, device = ModelPrepper(model, config).out
    if config['optimizer']=='Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    else:
        optimizer = Rprop(model.parameters())

    if config['mode'] == 'eval' or config['resume'] == 1:
        model.load_state_dict(torch.load(config['ckpt_path']))

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=size, desc=desc.format(0))

    def process_batch(engine, batch):
        frames, outputs = depickle(batch)

        if config['mode'] in ['train','overfit']:
            model.train()
            preds = model(frames)
            optimizer.zero_grad()
            loss = loss_fn(outputs.to(device), preds)*100
            loss.backward()
            optimizer.step()
            # print(model.gcu.W.grad)
            # print(model.gcu.weight.grad)
            # print(model.gcu.variance.grad)
        elif config['mode'] == 'eval':
            model.eval()
            preds = model(frames)
            loss = loss_fn(outputs.to(device), preds)*100
        else:
            raise NotImplementedError

        # print(frames.size(), outputs.size())
        # print(pred.size())

        # print(preds, outputs)
        # print(loss.item())

        return loss.item()

    trainer = Engine(process_batch)
    trainer.logger = setup_logger("trainer")

    if config['mode'] in ['train','overfit']:
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
        training_saver = ModelCheckpoint("checkpoint",
                                     filename_prefix="checkpoint",
                                     n_saved=1,
                                     atomic=True,
                                     save_as_state_dict=True,
                                     create_dir=True, require_empty=False)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, training_saver, 
                              {'model': model})

    @trainer.on(Events.ITERATION_COMPLETED)
    def tb_log(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(config['batch_size'])

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

    trainer.run(train_loader, max_epochs=config['epochs'], epoch_length=size/config['batch_size'])
    pbar.close()
    
if __name__ == "__main__":
    config = json.load(open('config.txt'))
    run(config)