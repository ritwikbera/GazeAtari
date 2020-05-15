# Gaze Guided Imitation
Using eye gaze as a supervision modality for imitation learning using graph neural networks.

### Repository Structure
* _panda.py_ : Data preprocessing script. Generates visualizations of the eye-gaze heatmap from raw frames and csv file data. Serializes stacked frames and outputs into a _pickle_ file.

* _dataloader.py_ : A streaming dataloader that fetches the pickled files scattered across different trials (in different folders) via multiple workers concurrently (to form a single batch).

* _train.py_ : Training script. Parameters set in the config dictionary.

* _eval.py_ : Evaluation script to test the code on a small but diverse, sampled batch. Includes support for visualization of NN weights (as histograms) in Tensorboard.

* _graphLayer.py_ : Graph Convolution Unit (uses _Named Tensors_). Multiple of these can instantiated and stacked either sequentially or in parallel to create more sophisticated graph NN based architectures.

* _model.py_ : Stores the model. Includes the GCU units, a 3x3 convolutional layer on top to aggregate information across neigboring nodes in the scene graph and 2 1x1 convolutions that together replace a fully connected layer.

* _utils.py_ : Contains utility and helper functions that are required for training such as creation of Tensorboard writer, conversion of model to a data parallel version etc.

* _unzip.py_ : Extracts all frames from the zipped _trial_ folders into a new folder

* _build_commands.py_ : Creates a JSON dictionary mapping action names to integer values.

* _gcn.py_ : Familiarising myself with the concept of graph neural networks on simple MNIST data.

### Dependencies
* torch
* ignite
* matplotlib
* numpy
* scipy
* PIL

### References
* [AGIL: Learning Attention from Human for Visuomotor Tasks](https://arxiv.org/abs/1806.03960)
* [Beyond Grids: Learning Graph Representations for Visual Recognition](https://papers.nips.cc/paper/8135-beyond-grids-learning-graph-representations-for-visual-recognition)