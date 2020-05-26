# Gaze Guided Imitation
Using eye gaze as a supervision modality for imitation learning using graph neural networks.

## Repository Structure

* _preprocess_atari.py_ : Data preprocessing script. Generates visualizations of the eye-gaze heatmap from raw frames and csv file data. Serializes stacked frames and outputs into a _pickle_ file.

* _train.py_ : Master script for both training and evaluation. Parameters set in the config dictionary loaded from the config.json file in the main directory.

* _generate\_sandbox.py_ : Creates an overfit folder that contains a small subset of data for code testing purposes.

* _unzip.py_ : Extracts all frames from the zipped _trial_ folders into a new folder

### models

* _graphLayer.py_ : Graph Convolution Unit (uses _Named Tensors_). Multiple of these can instantiated and stacked either sequentially or in parallel to create more sophisticated graph NN based architectures.

* _gauss.py_ : Performs gaussion filtering over a _PyTorch_ tensor.

* _policy.py_ : Contains the policy networks used for gaze-augmented behavior cloning.

* _convfc.py_ : A fully connected layer implemented by two consecutive 1x1 convolutions as described [here](https://tech.hbc.com/2016-05-18-fully-connected-to-convolutional-conversion.html).

* _gaze\_model.py_ : Stores the model. Includes the GCU units, a 3x3 convolutional layer on top to aggregate information across neigboring nodes in the scene graph and 2 1x1 convolutions that together replace a fully connected layer.

* _gcn.py_ : Familiarizing myself with the concept of graph neural networks on simple MNIST data.

### utils

* _dataloader.py_ : A streaming dataloader that fetches the pickled files scattered across different trials (in different folders) via multiple workers concurrently (to form a single batch). Provides for train/test splits and maintaining sequential order of data within individual folders/trials.

* _train\_utils.py_ : Contains helper functions that are required for training such as creation of Tensorboard writer, normalization and denormalization, conversion of model to a data parallel version etc.

* _data\_utils.py_ : Contains helper functions used in data pre-processing for data found in the _Atari-HEAD_ dataset.

* _viz\_utils.py_ : Contains helper functions used in visualization of gaze heatmaps and weight histograms (during training) for any specific layer in the model.

* _debug\_utils.py_ : Contains a helper function to visualize the gradient flow in the model across layers.

* _build\_commands.py_ : Creates a JSON dictionary mapping action names to integer values.

* _instance.py_ : Helper function that creates instances of items required for the training loop such as th eoptimizer, model, loss etc.

### dataset

* getdata.sh : Fetches and unpacks the dataset for any specified game (which is available in the Atari-HEAD dataset). 

### experiments

* checkpoints : Stores checkpoint files for the models.

* logs : Stores log files containing Tensorboard data.

### config_templates
Contains templates for config.json files.

## Instructions

* If dataset is not downloaded, head to the dataset folder and run ```./getdata.sh```.
    * Set the _game_ field in the file to download files relating to the appropriate _Atari_ game.
* Run ```python3 preprocess_atari.py```.
    * Set the fields in the function call before the script is run.
* Set up the experiment by setting the fields in the ```config.json``` file.
* Run ```python3 train.py```.
* Run ```tensorboard --logdir=experiments/logs```. 
    * Ensure only the latest log file is present in the logs folder before the command is run.

## Dependencies
* torch (PyTorch)
* pytorch-ignite
* matplotlib
* numpy
* scipy
* pillow (PIL)

## References
* [AGIL: Learning Attention from Human for Visuomotor Tasks](https://arxiv.org/abs/1806.03960)
* [Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset](https://arxiv.org/pdf/1903.06754.pdf)
* [Beyond Grids: Learning Graph Representations for Visual Recognition](https://papers.nips.cc/paper/8135-beyond-grids-learning-graph-representations-for-visual-recognition)