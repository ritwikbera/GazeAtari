import os, glob, fnmatch
from collections import defaultdict, OrderedDict
from pprint import pprint 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image 
import math
from scipy.ndimage import gaussian_filter
import scipy.misc as sm 
import torch
from functools import wraps
import inspect

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

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

def create_gazemap(gazemap, coords):
	x,y = 0,0
	for ind in range(0,len(coords),2):
		x,y = float(coords[ind]), float(coords[ind+1])
		x,y = np.clip(int(x),0,gazemap.shape[0]-1), np.clip(int(y),0,gazemap.shape[1]-1)
		gazemap[x,y] = 255.0

	return gazemap, x, y

def gazemap_to_heatmap(gazemap, h_sig=3, w_sig=2):
	cmap = plt.get_cmap('jet')
	heatmap = gaussian_filter(gazemap, sigma = (h_sig,w_sig), mode='constant')
	heatmap = (heatmap*(1.0/np.amax(heatmap))) # rescaling max to 1.0
	heatmap = cmap(heatmap)
	heatmap = np.delete(heatmap, 3, 2) # delete alpha channel

	return heatmap

def frame_fetch(func):
	values = {'files':None, 'stack': None}

	@wraps(func)
	def inner(*args, **kwargs):
		kwargs.update(get_default_args(func))
		index = args[0]
		if values['files'] is None:
			values['stack'], values['files'] = func(*args, **kwargs)
		else:
			kwargs['files'] = values['files']

			# stack memoization (only useful when dilation is 1. i.e. every frame is used)
			if kwargs['dilation'] == kwargs['sampling_interval']:
				file = values['files'][index]
				values['stack'] = torch.cat((values['stack'][:,:,:,3:],\
					torch.Tensor(np.asarray(Image.open(file))).unsqueeze(0)), -1)
			else:
				values['stack'], _ = func(*args, **kwargs) # enable if not using stack memoization
		return values['stack'], values['files']

	return inner

@frame_fetch
def stack_frames(index, img_folder, sampling_interval, stack_size=4, dilation=5, files=None):
	'''
	dilation : every dilation^th frame included in stacdilationk
	stack_size : number of frames in stack
	'''
	# TODO: keep a running stack to avoid repeated fetching of frames.

	if not files:
		files = glob.glob(img_folder+'/*.png')
		files = [files[0]]*(stack_size*dilation-1) + files
	
	tensors = list(map(lambda x: torch.Tensor(np.asarray(Image.open(x))), \
		files[i:i+stack_size*dilation:dilation]))
	return torch.cat((tensors), -1).unsqueeze(0), files