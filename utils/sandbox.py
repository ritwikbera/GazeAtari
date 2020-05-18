from glob import glob
import os
from random import sample 
from shutil import copyfile
import shutil

def gen_sandbox(num_samples_per_trial = 80, game = 'alien'):
	path = 'dataset/'+game+'/data/*'
	gaze_data = [(trial, sample(glob(trial+'/*.pth'), num_samples_per_trial)) for trial in glob(path)]

	sandbox_path = 'dataset/'+game+'/overfit/'

	try:
	    shutil.rmtree(sandbox_path)
	except OSError as e:
	    print("Error: %s : %s" % (sandbox_path, e.strerror))

	os.makedirs(sandbox_path)

	for (trial, files) in gaze_data:
		trial_folder = sandbox_path+os.path.basename(trial)
		os.makedirs(trial_folder)
		for file in files:
			# print(os.path.basename(file))
			copyfile(file, trial_folder+'/'+os.path.basename(file))

if __name__=='__main__':
	gen_sandbox()