import os
from glob import glob
from natsort import natsorted
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Split Midas task 1 dataset')
parser.add_argument(
        "--path",
        nargs="?",
        type=str,
        default="./subtask1",
        help="Path to store the dataset",
    )
args = parser.parse_args()

classes = [
'0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', 
'7 - seven', '8 - eight', '9 - nine', 'A - Captial', 'B - Captial', 'C - Captial',
'D - Captial', 'E - Captial', 'F - Captial', 'G - Captial', 'H - Captial', 'I - Captial',
'J - Captial', 'K - Captial', 'L - Captial', 'M - Captial', 'N - Captial',
'O - Captial', 'P - Captial', 'Q - Captial',
'R - Captial', 'S - Captial', 'T - Captial', 'U - Captial', 'V - Captial', 'W - Captial',
'X - Captial', 'Y - Captial', 'Z - Captial', 'a - Small', 'b - Small', 'c - Small', 'd - Small',
'e - Small', 'f - Small', 'g - Small', 'h - Small', 'i - Small', 'j - Small',
'k - Small', 'l - Small', 'm - Small', 'n - Small', 'o - Small', 'p - Small', 'q - Small', 'r - Small',
's - Small', 't - Small', 'u - Small',
'v - Small', 'w - Small', 'x - Small', 'y - Small', 'z - Small']

os.system("wget -c https://www.dropbox.com/s/pan6mutc5xj5kj0/trainPart1.zip")
os.system(f"unzip trainPart1.zip -d {args.path}")
os.system("rm trainPart1.zip")

# Renaming folder to MNISt classes
foldernames = natsorted(glob(args.path+"/train/*"))

for folder, classname in tqdm(zip(foldernames,classes)):
    os.renames(folder,(args.path + "/train/" + classname))