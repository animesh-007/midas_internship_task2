# Solution+Report for **MIDAS Internship task 2**
This repository contains solution+report for the MIDAS internship task 2.

## Installation
The model is built in PyTorch 1.8.1 and tested on Ubuntu 18.04 environment (Python3.7.10, CUDA10.1, cuDNN7.6.3).

For installing, follow these intructions
```
conda create -n pytorch1.8 python=3.7.10
conda activate pytorch1.8
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.1 -c pytorch
pip install requirements.txt
```
## Network Architecture
The network architecture below is used for all the 3 subtasks by changing the last layer of the architecture.

<img src="https://github.com/animesh-007/midas_internship_task2/blob/master/images/architecture.png">

## Task2_1
Change directory to Task2_1 using `cd task2_1`
### Prepare data
- Download data using `python download.py`
- It will download data for the Task2_1 in the `./downloadeddata` directory and will also rename the folders according to the labels in the MNIST.
- Run `python split.py` for splitting the dataset in 80:20 train-val ratio for training and validating the trained model on the given dataset and save the data in `./data`.

### Training
- For Training the model from scratch on the Task2_1 dataset. Run `python train.py`

### Results
|                Method                	| Epochs  | Accuracy 	|
|:------------------------------------:	|:------: |:--------:	|
| CNN without Scheduler                	|   30  	|  67.94  	|
| CNN with CosineAnnealingLR Scheduler 	|   30  	|  68.75  	|

## Task2_2
Change directory to Task2_2 using `cd task2_2`

### Prepare data
- Run `python process.py` for creating a subset from Task2_1 containing only images with digits labels in the `./data` directory.

### Training
- For Training the model from scratch on the Task2_3 dataset. Run `python train.py`

### Results
|                                      Method                                      	| Epochs| Accuracy 	|
|--------------------------------------------------------------------------------	|:------: |:--------:	|
| CNN on MIDAS dataset containing only digits, with a CosineAnnealingLR scheduler	|   30  	|   66.36  	|
| CNN on MNIST dataset with random weights, with a CosineAnnealingLR scheduler.	 	|   30  	|   99.39  	|
| CNN on MNIST dataset with pretrained weights, with a CosineAnnealingLR scheduler|   30  	|   99.34 	|

## Task2_3
Change directory to Task2_3 using `cd task2_3`

### Prepare data
- Run `python download.py` for for downloading the data in the `./data` directory.

### Training
- For Training the model from scratch on the Task2_3 dataset. Run `python train.py`


### Results
| Method                                                                                	| Accuracy 	|
|---------------------------------------------------------------------------------------	|:--------:	|
| CNN on MIDAS Dataset with random weights.                                             	|    1.74      	|
| CNN on MIDAS Dataset with pretrained weights of<br> MIDAS dataset containing only digits. 	|       10.32   	|

### Model Checkpoints

