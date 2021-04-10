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
        default="./subtask2",
        help="Path to store the dataset",
    )
parser.add_argument(
        "--subtask1",
        nargs="?",
        type=str,
        default="./subtask1",
        help="Path to the subtask1 dataset",
    )
args = parser.parse_args()

folders = (natsorted(glob(args.subtask1+"train/*")))

folders = folders[0:10]

os.makedirs("../subtask2/train",exist_ok=True)

for folder in folders:
    print(folder)
    train_destination = f"../subtask2/train/{os.path.split(folder)[-1]}"
    shutil.copytree(folder,train_destination)


