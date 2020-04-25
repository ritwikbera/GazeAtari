import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import random
import time
from itertools import islice, chain, cycle

class MyIterableDataset(IterableDataset):
	def __init__(self, data_list, batch_size):        
		self.data_list = data_list        
		self.batch_size = batch_size    

	@property
	def shuffled_data_list(self):
		return random.sample(self.data_list, len(self.data_list))

	def process_data(self, data):
		for x in data:            
			worker = torch.utils.data.get_worker_info()            
			worker_id =id(self) if worker is not None else-1            
			start = time.time()            
			time.sleep(0.1)            
			end = time.time()            
			yield x, worker_id, start, end

	def get_stream(self, data_list):        
		return chain.from_iterable(map(self.process_data, cycle(data_list)))

	def get_streams(self):
		return zip(*[self.get_stream(self.shuffled_data_list)for _ inrange(self.batch_size)])

	def __iter__(self):
		return self.get_streams()    

	@classmethod
	def split_datasets(cls, data_list, batch_size, max_workers):        
		for n in range(max_workers, 0, -1):
			if batch_size % n ==0:                
				num_workers = nbreak        
				split_size = batch_size // num_workers
				return [cls(data_list, batch_size=split_size) for _ in range(num_workers)]

class MultiStreamDataLoader:
	def __init__(self, datasets):        
	self.datasets = datasets
	
	def get_stream_loaders(self):
		return zip(*[DataLoader(dataset, num_workers=1, batch_size=None) for dataset in datasets])

	def __iter__(self):
		for batch_parts in self.get_stream_loaders():
			yield list(chain(*batch_parts))