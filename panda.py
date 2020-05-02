import pandas as pd 
from csv import reader, DictReader
import os, glob, fnmatch
from collections import OrderedDict
from pprint import pprint 
import numpy as np 
from skimage import data, io
import matplotlib.pyplot as plt
from PIL import Image 

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

files = glob.glob('dataset/alien/*.txt')
episode = 0
filename = files[episode]
img_folder = 'dataset/alien/extracted/'+filename.split('/')[-1][:-4]
out_folder = 'dataset/alien/gaze/'+filename.split('/')[-1][:-4]
os.makedirs(out_folder, exist_ok=True)
print(filename)
print(img_folder)
print(out_folder)

assert os.path.isdir(img_folder) and os.path.isdir(out_folder)

with open(filename, 'r') as read_obj:
	csv_reader = DictReader(read_obj)
	header = next(csv_reader)
	if header != None:
		for row in csv_reader:
			row['gaze_positions'] = row[None] + [row['gaze_positions']]
			del row[None]
			
			# print(row)
			
			frame_file = img_folder+'/'+row['frame_id']+'.png'
			assert os.path.isfile(frame_file)
			frame = io.imread(frame_file)
			frame = crop_image(frame)
			gazemap = np.zeros_like(frame)[:,:,0]
			for coord in row['gaze_positions']:
				x,y = coord.strip().split('.')
				x,y = int(x), int(y)
				gazemap[x,y] = 255

			io.imsave(out_folder+'/'+row['frame_id']+'.png', gazemap)
			
			# print(gazemap.shape)
			# plt.imshow(gazemap)
			# plt.imshow(frame)
			
			# break

# plt.show()

# df = pd.read_csv('dataset/meta_data.csv')
# print(df[df['GameName']=='alien'])
# print(df.columns)
