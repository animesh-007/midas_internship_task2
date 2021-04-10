import os
from glob import glob
from natsort import natsorted
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Split Midas task 2 dataset')
parser.add_argument(
        "--path",
        nargs="?",
        type=str,
        default="../task2_1/downloadeddata/train",
        help="Path to store the dataset",
    )
parser.add_argument(
        "--path2",
        nargs="?",
        type=str,
        default="./data/train",
        help="Path to the subtask1 dataset",
    )
args = parser.parse_args()

folders = (natsorted(glob(args.path+"/*")))

folders = folders[0:10]

os.makedirs(args.path2,exist_ok=True)

for folder in folders:
    print(folder)
    train_destination = f"{args.path2}/{os.path.split(folder)[-1]}"
    shutil.copytree(folder,train_destination)


