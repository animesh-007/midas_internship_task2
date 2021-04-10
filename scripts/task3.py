import os
from glob import glob
from natsort import natsorted
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Split Midas task 3 dataset')
parser.add_argument(
        "--path",
        nargs="?",
        type=str,
        default="./subtask3",
        help="Path to store the dataset",
    )
args = parser.parse_args()


os.system("wget -c https://www.dropbox.com/s/otc12z2w7f7xm8z/mnistTask3.zip?dl=0")
os.system(f"unzip mnistTask3.zip?dl=0 -d {args.path}")
os.system("rm trainPart1.zip")
