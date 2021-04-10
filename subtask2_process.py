import os
from glob import glob
from natsort import natsorted
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Split Midas task 2 dataset')
parser.add_argument(
        "--subtask2",
        nargs="?",
        type=str,
        default="../subtask2/train",
        help="Path to store the dataset",
    )
parser.add_argument(
        "--subtask1",
        nargs="?",
        type=str,
        default="../subtask1/train",
        help="Path to the subtask1 dataset",
    )
args = parser.parse_args()

folders = (natsorted(glob(args.subtask1+"/*")))

folders = folders[0:10]

os.makedirs(args.subtask2,exist_ok=True)

for folder in folders:
    print(folder)
    train_destination = f"{args.subtask2}/{os.path.split(folder)[-1]}"
    shutil.copytree(folder,train_destination)


