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

def white_to_red(im):
	data = np.array(im)   # "data" is a height x width x 4 numpy array
	red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

	# Replace white with red... (leaves alpha values alone...)
	white_areas = (red == 255) & (blue == 255) & (green == 255)
	data[..., :-1][white_areas.T] = (255, 0, 0) # Transpose back needed

	im2 = Image.fromarray(data)
	return im2

files = glob.glob('dataset/alien/*.txt')
episode = 0
filename = files[episode]
img_folder = 'dataset/alien/extracted/'+filename.split('/')[-1][:-4]
out_folder = 'dataset/alien/gaze/'+filename.split('/')[-1][:-4]
os.makedirs(out_folder, exist_ok=True)
viz_folder = 'dataset/alien/viz/'+filename.split('/')[-1][:-4]
os.makedirs(viz_folder, exist_ok=True)
print(filename)
print(img_folder)
print(out_folder)

assert os.path.isdir(img_folder) and os.path.isdir(out_folder)

i = 0
with open(filename, 'r') as read_obj:
	csv_reader = DictReader(read_obj)
	header = next(csv_reader)
	if header != None:
		for row in csv_reader:
			try:
				row['gaze_positions'] = row[None] + [row['gaze_positions']]
				del row[None]
			except KeyError:
				continue  # No None means just gaze pixel present, skip this frame

			# print(row)
			
			frame_file = img_folder+'/'+row['frame_id']+'.png'
			assert os.path.isfile(frame_file)
			frame = io.imread(frame_file)

			# frame = crop_image(frame)

			gazemap = np.zeros_like(frame)[:,:,0]
			assert gazemap.shape == frame.shape[:-1]

			for coord in row['gaze_positions']:
				x,y = coord.strip().split('.')
				x,y = np.clip(int(x),0,gazemap.shape[0]-1), np.clip(int(y),0,gazemap.shape[1]-1)
				gazemap[x,y] = 255

			out_file = out_folder+'/'+row['frame_id']+'.png'
			io.imsave(out_file, gazemap)
			
			im1 = Image.open(frame_file)
			im2 = Image.open(out_file)
			im1 = im1.convert('RGBA')
			im2 = im2.convert('RGBA')
			im2 = white_to_red(im2)
			blend = Image.blend(im1, im2, alpha=0.5)
			blend.save(viz_folder+'/img_{:05}.png'.format(i))
			i = i + 1
			
			# blend.show()
			# print(gazemap.shape)
			# plt.imshow(gazemap)
			# plt.imshow(frame)
			
			# break

# plt.show()

# df = pd.read_csv('dataset/meta_data.csv')
# print(df[df['GameName']=='alien'])
# print(df.columns)
