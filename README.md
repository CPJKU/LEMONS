# LEMONS: Listenable Explanations for Music recOmmeNder Systems
**LEMONS** addresses the issue of explaining of why a track has been recommended to a user by providing listenable explanations based on the track itself.

## Overview

**LEMONS** consists of the following 2 parts: 

1. Music Recommender System. The RS takes in input the audio tracks and outputs the relevance for the user. 
2. Listenable Explanations. Explanations are computed post-hoc using [audioLIME](https://github.com/CPJKU/audioLIME), an extension of [LIME](https://arxiv.org/abs/1602.04938) for audio data.

The functionality is demonstrated using a [streamlit](https://www.streamlit.io/) app. A screenshot of the **LEMONS** app can be seen below.

![](imgs/landing_page.png)

You can check out the [video of our demo](https://www.youtube.com/watch?v=giSPrPnZ7mc) (~9 minutes).

In the following you can find the details about the recommender system, how to setup and conduct the same experiments and how to run the `streamlit` app to play around with the explanations.

## Audio-based Recommender System - Model and Training details
### Input
For training on the Million Song Dataset, we use snippets from 7digital. Snippet durations range from 30s to 60s.
Audios are downsampled to 16kHz and transformed in decibel mel-spectograms. We use 256 mel bins with a hop size of 512. Only for training, we train on 1s randomly selected part of the snippet, leading to the input shape of 256x63.

### Model
The structure of the audio-based recommender system is depicted below.
| Layers                                        |
|-----------------------------------------------|
| BatchNorm2d                                   |
| Conv2d(1,64), BatchNorm2d, ReLU, MaxPool2d    |
| Conv2d(64,128), BatchNorm2d, ReLU, MaxPool2d  |
| Conv2d(128,128), BatchNorm2d, ReLU, MaxPool2d |
| Conv2d(128,128), BatchNorm2d, ReLU, MaxPool2d |
| Conv2d(128,64), BatchNorm2d, ReLU, MaxPool2d  |
| Cat(AdaptiveAvgPool2d + AdaptiveMaxPool2d)    |
| Dropout(0.5)                                  |
| Linear(128,1)                                 |

Convolutions have a kernel of 3x3 while MaxPooling halves in both dimensions each time.
In the last layers we concatenate global average pooling and global max pooling, apply dropout, and feed it to a linear layer.

### Training
We use a batch size of 20 and train for 1000 epochs with a learning rate of 1e-3, weight decay of 1e-4, and Adam optimizer.
We train a total of 7 models, one for each user. 

### Validation and Testing
For evaluation, we use as input the whole track.

## Setup

### Create an environment with all dependencies

```shell script
conda env create -f lemons.yml
conda activate lemons
```

### Install `lemons`
In the root directory, run the following:
```shell script
python3 setup.py develop
```
or, if it doesn't work
```shell script
pip install -e .
```

### Config

Some paths need to be set, e.g. to the location of your data.
Copy `config_template.py` to `config.py` and set your paths there. `config.py` is in `.gitignore` 
such that each user has their own config without overwriting the others.

## Training
Before training, it could be necessary to tune the following parameters.

We use [`sacred`](https://github.com/IDSIA/sacred) to log all experiments. In `local_conf` in `recsys/experiment.py`, change the following:
- mongodb_db_name 
```python
local_conf = {
    'mongodb_url': '',
    'mongodb_db_name': '',
    'experiment_name': '',
}
```

In `experiment_config()` in `recsys/experiment.py`, you can change the following parameters:
```python
# --Logging Parameters-- #

use_tensorboard = 0  # if also tensorboard (together with sacred) should be used
log_step = 100 # how many batches have to pass before logging the batch loss (NB. this is not for avg_loss)

# --Training Parameters-- #

training_seed = 1930289 # seed used for training (independent of the data seed)
model_load_path = '' # if load pre-trained model
freeze = 1  # if freeze the weights of a pre-trained model
batch_size = 20 # batch size
n_epochs = 1000 # epochs for training
lr = 1e-3 # learning rate
wd = 1e-4 # weight decay
num_workers = 10 # number of workers
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # which device to use

# --Data Parameters-- #

data_path = '' # path to the npys
meta_path = '../data/' # path to the meta data
user_name = 'marko' # users (check utils, get_user_id)
```
Then training can be run with:
```shell script
cd recsys
python3 experiment.py
```
The best model will be saved by default in the directory /experiments/<date>.

## Testing

In local_conf in experiment.py, change the following:
- mongodb_db_name  (similar to above):
In experiment_config() in eval.py, you can change the following parameters (commented):
```python
# --Logging Parameters-- #

use_tensorboard = 1  # if also tensorboard (together with sacred) should be used

# --Evaluation Parameters-- #

model_load_path = 'best_model.pth'  # path to the trained model
batch_size = 20  # batch size
num_workers = 10  # number of workers
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   # which device to use

# --Data Parameters-- #

data_path = ''  # path to the npys
meta_path = '../data/'  # path to the meta data
user_name = 'marko'  # users (check utils, get_user_id)
```
Then the evaluation can be run with:
```shell script
cd training/
python3 eval.py
```
The results will be saved in the same directory of "model_load_path".

## Demo 

You can look at a demonstration using the `streamlit` app. 
It has to be run from the `lemons` root directory.

```
streamlit run explanations/lemons.py
```

## Experiments & Results

We split the tracks into train, validation,and test set in an 80-10-10 fashion and select the model that achieves the bestresults in terms of AUC and MAP on the validation set. The results on the testset averaged across the users are 0.734±0.130 MAP and 0.758±0.113 AUC.

### Stability of explanations
We select the number of samples in the neighborhood N_s to get stable explanations by following the procedure described in [[Mishra 2020] Reliable Local Explanations for Machine Listening](https://arxiv.org/abs/2005.07788). In this experiment, the computation of the explanations is repeated 5 times, and each time the top k=3 interpretable components are recorded. With increasing number of samples N_s the number of unique components U_n should approach k (in our case: 3). We found that a number of N_s=2^11=2048 suffices to compute stable explanations in a reasonable amount of time.

![](imgs/stability.png)

Each violin represents the results for one user model for a subset of the test set (50 examples). Each data point in a violin shows how many unique components U_n (shown on the x-axis) were selected when repeating computation of the explanation for a test sample 5 times. The y-axis shows the number of neighborhood examples N_s that was used for training the explainer in each case. The figure shows that increasing N_s decreases U_n, on average. This means that for example for Sandra (purple), using `N_s=2048` and repeatedly computing an explanation consisting of 3 components for the same track will result in the same 3 components being picked (for a majority of the test songs).
