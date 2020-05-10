import pandas as pd 
from csv import reader, DictReader
import os, glob, fnmatch
from collections import OrderedDict
from pprint import pprint 
import numpy as np 
# from skimage import data, io
import matplotlib.pyplot as plt
from PIL import Image 
import math
from scipy.ndimage import gaussian_filter
import scipy.misc as sm 
import torch

def get_sigma(img):
	Vs = 1*math.pi/180
	D = 78.7
	H, W = 44.6, 28.5
	h, w = img.shape[0], img.shape[1]
	S = D*math.tan(Vs)
	h_sig, w_sig = h*S/H, w*S/W
	return ((h_sig-1)/2-0.5)/4.0, ((w_sig-1)/2 - 0.5)/4.0

def crop_image(img,tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def white_to_red(im):
	data = np.array(im)   # "data" is a height x width x 4 numpy array
	red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

	# Replace white with red... (leaves alpha values alone...)
	white_areas = (red == 255) & (blue == 255) & (green == 255)
	data[..., :-1][white_areas.T] = (255, 0, 0) # Transpose back needed

	im2 = Image.fromarray(data)
	return im2

def blend_(file1, file2, alpha=0.5, w2r=False):
	im1 = Image.open(file1)
	im2 = Image.open(file2)
	im1 = im1.convert('RGBA')
	im2 = im2.convert('RGBA')

	if w2r:
		im2 = white_to_red(im2)

	return Image.blend(im1, im2, alpha=alpha)

def stack_frames(index, img_folder):
	# TODO: keep a running stack to avoid repeated fetching of frames.
	stack_size = 4
	files = glob.glob(img_folder+'/*.png')
	files = [files[0]]*(stack_size-1) + files
	tensors = list(map(lambda x: torch.Tensor(np.asarray(Image.open(x))), files[i:i+4]))
	return torch.cat((tensors), -1).unsqueeze(0)

game = 'alien'
files = glob.glob('dataset/'+game+'/*.txt')
episode = 0

filename = files[episode]

img_folder = 'dataset/'+game+'/extracted/'+filename.split('/')[-1][:-4]

out_folder = 'dataset/'+game+'/gaze/'+filename.split('/')[-1][:-4]
os.makedirs(out_folder, exist_ok=True)

viz_folder = 'dataset/'+game+'/viz/'+filename.split('/')[-1][:-4]
os.makedirs(viz_folder, exist_ok=True)

data_folder = 'dataset/'+game+'/data/'+filename.split('/')[-1][:-4]
os.makedirs(data_folder, exist_ok=True)

cmap = plt.get_cmap('jet')

assert os.path.isdir(img_folder) and os.path.isdir(out_folder)

i = 0
with open(filename, 'r') as read_obj:
	csv_reader = DictReader(read_obj)
	# header = next(csv_reader)
	# if header != None:
	if True:   # replace with commented block if using just reader instead of DictReader
		for row in csv_reader:
			try:
				row['gaze_positions'] = row[None] + [row['gaze_positions']]
				del row[None]
			except KeyError:
				continue  # No None means just gaze pixel present, skip this frame

			# print(row)
			
			frame_file = img_folder+'/'+row['frame_id']+'.png'
			out_file = out_folder+'/'+row['frame_id']+'.png'
			viz_file = viz_folder+'/img_{:05}.png'.format(i)
			data_file = data_folder+'/'+row['frame_id']+'.pth'

			assert os.path.isfile(frame_file)

			frame = np.asarray(Image.open(frame_file))
			# frame = crop_image(frame)
			# TODO: downpooling and processing to get standard 84x84 shape for ALE
			# print(frame.shape)
			# plt.imsave(frame_file, frame)

			gazemap = np.zeros_like(frame)[:,:,0]
			heatmap = gazemap
			assert gazemap.shape == frame.shape[:-1]
			
			h_sig, w_sig = get_sigma(frame)

			x,y =0,0
			for ind in range(0,len(row['gaze_positions']),2):
				x,y = float(row['gaze_positions'][ind]), float(row['gaze_positions'][ind+1])
				x,y = np.clip(int(x),0,gazemap.shape[0]-1), np.clip(int(y),0,gazemap.shape[1]-1)
				gazemap[x,y] = 255.0


			h_sig, w_sig = 3, 2
			heatmap = gaussian_filter(gazemap, sigma = (h_sig,w_sig), mode='constant')
			heatmap = (heatmap*(1.0/np.amax(heatmap))) # rescaling max to 1.0
			heatmap = cmap(heatmap)
			heatmap = np.delete(heatmap, 3, 2) # delete alpha channel

			data = {'frame_stack': stack_frames(i, img_folder), 'gaze_point': torch.Tensor([[x,y]])}
			torch.save(data, data_file)

			plt.imsave(viz_file, gazemap)
			plt.imsave(out_file, heatmap)
		
			blend = blend_(frame_file, viz_file, alpha=0.3, w2r=True)
			blend.save(viz_file)
			blend = blend_(viz_file, out_file)
			blend.save(viz_file)
			i = i + 1

			print('Done with frame {}'.format(i))
			
			# blend.show()
			# fig = plt.figure()
			# ax1 = fig.add_subplot(2,2,1)
			# ax1.imshow(frame)
			# ax2 = fig.add_subplot(2,2,2)
			# ax2.imshow(heatmap)
			# ax3 = fig.add_subplot(2,2,3)
			# ax3.imshow(gazemap)

			if i == 100:
				break

# after all frames, goto viz folder and run ffmpeg -i img_%05d.png video.mp4

# plt.show()

# df = pd.read_csv('dataset/meta_data.csv')
# print(df[df['GameName']=='alien'])
# print(df.columns)
