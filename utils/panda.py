from csv import reader, DictReader
import os, glob, fnmatch, shutil
from collections import defaultdict, OrderedDict
from pprint import pprint 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image 
import math
from scipy.ndimage import gaussian_filter
import scipy.misc as sm 
import torch
from .data_utils import *


def prep_data(game='alien', num_trials=1, sampling_interval=5, num_actions=18):
	'''
	game = 'alien'
	episode = 0 # trial
	sampling_interval = 5 # every 5th frame sampled (with stacked history)
	'''
	files = glob.glob('dataset/'+game+'/*.txt')
	for episode in range(num_trials):
		filename = files[episode]

		img_folder = 'dataset/'+game+'/extracted/'+filename.split('/')[-1][:-4] # raw frames
		out_folder = 'dataset/'+game+'/gaze/'+filename.split('/')[-1][:-4] # heatmap only
		viz_folder = 'dataset/'+game+'/viz/'+filename.split('/')[-1][:-4]  # full visualiztion
		data_folder = 'dataset/'+game+'/data/'+filename.split('/')[-1][:-4] # pickled data
		try:
			shutil.rmtree(out_folder)
			shutil.rmtree(viz_folder)
			shutil.rmtree(data_folder)
		except:
			pass
		os.makedirs(out_folder)
		os.makedirs(viz_folder)
		os.makedirs(data_folder)

		assert os.path.isdir(img_folder) and os.path.isdir(out_folder)

		i = 0
		with open(filename, 'r') as read_obj:
			csv_reader = DictReader(read_obj)
			# header = next(csv_reader)
			# if header != None:
			if True:   # replace with commented block if using just reader instead of DictReader
				for i,row in enumerate(csv_reader):
					if i%sampling_interval > 0:
						continue
					try:
						row['gaze_positions'] = row[None] + [row['gaze_positions']]
						del row[None]
					except KeyError:
						continue  # No None means just gaze pixel present, skip this frame

					# print(row)
					
					frame_file = img_folder+'/'+row['frame_id']+'.png'
					out_file = out_folder+'/'+row['frame_id']+'.png'
					viz_file = viz_folder+'/img_{:05}.png'.format(i//sampling_interval)
					data_file = data_folder+'/'+row['frame_id']+'.pth'

					assert os.path.isfile(frame_file)

					frame = np.asarray(Image.open(frame_file))
					# frame = crop_image(frame)
					# TODO: downpooling and processing to get standard 84x84 shape for ALE
					# frame = frame.resize((84,84))
					print(frame.shape)
					# plt.imsave(frame_file, frame)

					gazemap = np.zeros_like(frame)[:,:,0]
					heatmap = gazemap
					assert gazemap.shape == frame.shape[:-1]
					
					h_sig, w_sig = get_sigma(frame)

					gazemap, x, y = create_gazemap(gazemap, row['gaze_positions'])
					heatmap = gazemap_to_heatmap(gazemap)

					data = {'frame_stack': stack_frames(i, img_folder, sampling_interval=sampling_interval)[0], \
					 'action':one_hot(int(row['action']), num_labels=num_actions), \
					 'gaze_point': torch.Tensor([[x,y]])}
					torch.save(data, data_file)

					plt.imsave(viz_file, gazemap)
					plt.imsave(out_file, heatmap)
				
					blend = blend_(frame_file, viz_file, alpha=0.3, w2r=True)
					blend.save(viz_file)
					blend = blend_(viz_file, out_file)
					blend.save(viz_file)

					print('Done with frame {}'.format(i))
					
					# blend.show()
					# fig = plt.figure()
					# ax1 = fig.add_subplot(2,2,1)
					# ax1.imshow(frame)
					# ax2 = fig.add_subplot(2,2,2)
					# ax2.imshow(heatmap)
					# ax3 = fig.add_subplot(2,2,3)
					# ax3.imshow(gazemap)

					# if i == 100:
					# 	break

		# after all frames, goto viz folder and run ffmpeg -i img_%05d.png video.mp4

		# plt.show()

if __name__=='__main__':
	prep_data()
