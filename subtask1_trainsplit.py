#author: animesh-007
import os
from glob import glob
from tqdm import tqdm
import shutil
import argparse
import random
random.seed(230)

## arguments used while running this file
parser = argparse.ArgumentParser(description='Split Midas task 1 dataset')
parser.add_argument(
        "--path",
        nargs="?",
        type=str,
        default="./subtask1_downloadeddata",
        help="Path to the dataset and splitting in same folder",
    )
parser.add_argument(
        "--path2",
        nargs="?",
        type=str,
        default="./subtask1_data",
        help="Path to the dataset and splitting in same folder",
    )

args = parser.parse_args()
path = args.path + "/train/*"

## read data inside the folder
for folder in tqdm(glob(path)):
    files = glob(folder+"/*")
    files.sort() # make sure that the filenames have a fixed order before shuffling
    random.shuffle(files) # shuffles the ordering of filenames (deterministic given the chosen seed)
    split_1 = int(0.8 * len(files))
    train_filenames = files[:split_1]
    val_filenames = files[split_1:]
    folder_name = os.path.split(folder)[-1]
    train_destination = f"{args.path2}/train/{folder_name}"
    val_destination = f"{args.path2}/val/{folder_name}"
    
    os.makedirs(train_destination,exist_ok=True)
    os.makedirs(val_destination,exist_ok=True)
    
    for train_file in train_filenames:
        shutil.copy(train_file,train_destination)
        
    for val_file in val_filenames:
        shutil.copy(val_file,val_destination)
    
    
    
    
    
