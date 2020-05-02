import pandas as pd 
from csv import reader, DictReader
import glob
from collections import OrderedDict
from pprint import pprint 

files = glob.glob('dataset/alien/*.txt')

with open(files[0], 'r') as read_obj:
	csv_reader = DictReader(read_obj)
	header = next(csv_reader)
	if header != None:
		for row in csv_reader:
			row['gaze_positions'] = row[None] + [row['gaze_positions']]
			del row[None]
			print(row)
			break

# df = pd.read_csv('dataset/meta_data.csv')

# print(df[df['GameName']=='alien'])

# alien_df = pd.read_csv('dataset/alien/314_RZ_9847886_Jun-06-14-05-27.txt')

# print(df.columns)
# print(alien_df.columns)

# print(alien_df['gaze_positions'].iloc[0:1])