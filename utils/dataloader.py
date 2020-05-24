from .train_utils import *

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import random
import time
from itertools import islice, chain, cycle
from functools import reduce
from glob import glob 

class MyIterableDataset(IterableDataset):
    def __init__(self, data_list, batch_size, sequential_data=True):        
        self.data_list = data_list        
        self.batch_size = batch_size    
        self.sequential_data = sequential_data

    # shuffling not possible by DataLoader in streaming dataloaders.
    @property
    def shuffled_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):  
        for x in data:      
            worker = torch.utils.data.get_worker_info()            
            worker_id = id(self) if worker is not None else -1  
            start = time.time()            
            time.sleep(0.001)            
            end = time.time()            
            yield x

    def process_data_(self, data):
        return data

    def get_stream(self, data_list): 
        if self.sequential_data: 
            return chain.from_iterable(map(self.process_data, cycle(data_list))) # if recurrence order has to be maintained
        else:
            return map(self.process_data_, cycle(random.sample(list(chain(*data_list)), sum(len(x) for x in data_list))))

    # randomization happens every time shuffle_data_list is called
    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()    

    @classmethod
    def split_datasets(cls, data_list, batch_size, max_workers):        
        for n in range(max_workers, 0, -1):
            if batch_size % n ==0:                
                num_workers = n
                break        
        split_size = batch_size // num_workers
        return [cls(data_list, batch_size=split_size) for _ in range(num_workers)]

class MultiStreamDataLoader:
    def __init__(self, datasets):        
        self.datasets = datasets
    
    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset, num_workers=1, batch_size=None, collate_fn=atariCollate) \
            for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield reduce(lambda a,b: [torch.cat((a[i],b[i]),0) for i in range(len(a))],batch_parts)

def atariCollate(batch): 
    data = ['frame_stack','gaze_point','action']
    out = [torch.cat(list(map(lambda x: torch.load(x)[key], batch)),0) for key in data]

    h,w = out[0].size(1), out[0].size(2)
    out[1] = normalize(out[1], xlim=w, ylim=h)
    out[2] = torch.max(out[2], 1)[1]

    return out

class AtariMSDL:
    def __init__(self, mode, path, batch_size, train_size):

        def file_set(trial):
            files = glob(trial+'/*.pth')
            split = int(train_size*len(files)/100)
            return files[:split] if mode == 'train' else files[split:]

        data_list = [file_set(trial) for trial in glob(path+'/*')[:]]
        
        self.size = recursive_len(data_list)
        print('Dataset Size {}'.format(self.size))
    
        datasets = MyIterableDataset.split_datasets(data_list, batch_size=batch_size, max_workers=4)
        self.train_loader = MultiStreamDataLoader(datasets)
    
    def __len__(self):
        return self.size 

    def __iter__(self):
        for batch in iter(self.train_loader):
            yield batch