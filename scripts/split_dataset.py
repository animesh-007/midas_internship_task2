import os
from glob import glob
from tqdm import tqdm
import shutil
import argparse
import random
random.seed(230)

parser = argparse.ArgumentParser(description='Split Midas task 1 dataset')
parser.add_argument(
        "--path",
        nargs="?",
        type=str,
        default="../subtask1",
        help="Path to the dataset and splitting in same folder",
    )
args = parser.parse_args()
path = args.path + "/train/*"


for folder in tqdm(glob(path)):
    files = glob(folder+"/*")
    files.sort() # make sure that the filenames have a fixed order before shuffling
    random.shuffle(files) # shuffles the ordering of filenames (deterministic given the chosen seed)

    
    split_1 = int(0.8 * len(files))
    train_filenames = files[:split_1]
    val_filenames = files[split_1:]
    folder_name = os.path.split(folder)[-1]
    train_destination = f"{args.path}/processed/train/{folder_name}"
    val_destination = f"{args.path}/processed/val/{folder_name}"
    
    os.makedirs(train_destination,exist_ok=True)
    os.makedirs(val_destination,exist_ok=True)
    
    for train_file in train_filenames:
        shutil.copy(train_file,train_destination)
        
    for val_file in val_filenames:
        shutil.copy(val_file,val_destination)
    
    
    
    
    