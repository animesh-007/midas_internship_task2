# Solution of **MIDAS Internship task 2**
This repository contains solution for the MIDAS internship task 2.

## Installation
The model is built in PyTorch 1.8.1 and tested on Ubuntu 18.04 environment (Python3.7.10, CUDA10.1, cuDNN7.6.3).

For installing, follow these intructions
```
conda create -n pytorch1.8 python=3.7.10
conda activate pytorch1.8
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.1 -c pytorch
pip install requirements.txt
```
## SubTask1

### Prepare data
- Download data using task1.py.
```
python subtask1_download.py
```
- It will download data for the SubTask 1 in the `./subtask1` directory and will also rename the folders according to the labels in the MNIST.
- Run `split_dataset.py` for splitting the dataset in 80:20 train-val ratio for training and validating the trained model on the given dataset.
```
python subtask1_trainsplit.py
```

### Training
- For Training the model from scratch on the SubTask1 dataset.
- Run `subtask1_train.py`
```
python subtask1_train.py 
```
### Results
|                Method                	| Epochs 	|  LR  	| Accuracy 	|
|:------------------------------------:	|:------:	|------	|:--------:	|
| CNN without Scheduler                	|   30   	|   1  	|  67.94  	|
| CNN with CosineAnnealingLR Scheduler 	|   30   	|   1  	|  68.75  	|

## SubTask 2

### Prepare data
- Run `subtask2_process.py` for creating a subset from SubTask 1 containing only images with digits labels in the `./subtask2` directory.
```
cd scripts
python task2.py
```

### Training
- For Training the model from scratch on the SubTask2 dataset.
- Run `subtask2_train.py`
```
python subtask2_train.py 
```

### Results
|                                      Method                                      	| Epochs 	|  LR  	| Accuracy 	|
|--------------------------------------------------------------------------------	|:------:	|:------:	|:--------:	|
| CNN on MIDAS dataset containing only digits, without a scheduler.                	|   30   	|   1  	|   66.36  	|
| CNN on MIDAS dataset containing only digits, with a CosineAnnealingLR scheduler. 	|   30   	|   1  	|   66.56  	|
| CNN on MNIST dataset with random weights, without a scheduler.                   	|   30   	|   1  	|   99.31  	|
| CNN on MNIST dataset with random weights, with a CosineAnnealingLR scheduler.                   	|   30   	|   1  	|   99.39   	|
| CNN on MNIST dataset with pretrained weights, without a scheduler.               	|   30   	|   1  	|   99.33  	|

## SubTask 3

### Prepare data
- Run `subtask3_download.py` for for downloading the data in the `./subtask3_data` directory.

### Training
- For Training the model from scratch on the SubTask3 dataset.
- Run `subtask3_train.py`


### Results
| Method                                                                                	| Accuracy 	|
|---------------------------------------------------------------------------------------	|:--------:	|
| CNN on MIDAS Dataset with random weights.                                             	|    1.74      	|
| CNN on MIDAS Dataset with pretrained weights of<br> MIDAS dataset containing only digits. 	|       10.32   	|
