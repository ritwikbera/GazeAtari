import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from ignite.engine import Events, Engine
from ignite.metrics import Loss
from ignite.utils import setup_logger

from tqdm import tqdm
from model import *
from dataloader import *

torch.manual_seed(0)
np.random.seed(0)

def run(batch_size, epochs, lr, log_interval):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    game = 'alien'
    path = 'dataset/'+game+'/data/*'
    data_list = [glob(trial+'/*.pth') for trial in glob(path)[:]]
    datasets = MyIterableDataset(data_list, batch_size=4).split_datasets(data_list, batch_size=4, max_workers=1)
    train_loader = iter(MultiStreamDataLoader(datasets))

    model = GazePred()
    loss_fn = nn.MSELoss()

    batch_ = next(iter(MultiStreamDataLoader(datasets))) 
    frames_ = torch.cat(list(map(lambda x: torch.load(x)['frame_stack'], batch_)),0)
    _ = model(frames_)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)


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

        print(preds, outputs)

        print(loss.item())

    trainer = Engine(process_batch)
    trainer.run(train_loader, max_epochs=epochs, epoch_length=100)
    trainer.logger = setup_logger("trainer")
    

if __name__ == "__main__":
    kwargs = {'batch_size':4, 'epochs':1, 'lr':5e-3, 'log_interval':100}
    run(**kwargs)