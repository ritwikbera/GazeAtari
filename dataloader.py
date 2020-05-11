import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import random
import time
from itertools import islice, chain, cycle
from glob import glob 

random.seed(10)

class MyIterableDataset(IterableDataset):
	def __init__(self, data_list, batch_size):        
		self.data_list = data_list        
		self.batch_size = batch_size    

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
		# return chain.from_iterable(map(self.process_data, cycle(data_list))) # if recurrence order has to be maintained
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
		return zip(*[DataLoader(dataset, num_workers=1, batch_size=None) for dataset in self.datasets])

	def __iter__(self):
		for batch_parts in self.get_stream_loaders():
			yield list(chain(*batch_parts))

if __name__=='__main__':
	data_list = [[12, 13, 14, 15, 16, 17], \
				 [27, 28, 29], \
				 [31, 32, 33, 34, 35, 36, 37, 38, 39], \
				 [40, 41, 42, 43]]

	a = map(lambda x: x**2, cycle(chain(*random.sample(data_list, len(data_list)))))
	print(next(a))
	print(next(a))
	print(next(a))

	game = 'alien'
	path = 'dataset/'+game+'/data/*'
	gaze_data = [glob(trial+'/*.pth') for trial in glob(path)[:]]

	data_list = gaze_data

	datasets = MyIterableDataset.split_datasets(data_list, batch_size=4, max_workers=1)
	loader = iter(MultiStreamDataLoader(datasets))
	for i in range(1):
		batch = next(loader)
		frames = torch.cat(list(map(lambda x: torch.load(x)['frame_stack'], batch)),0)
		outputs = torch.cat(list(map(lambda x: torch.load(x)['gaze_point'], batch)),0)
		print(frames.size())
		print(outputs.size())
		# print(batch)