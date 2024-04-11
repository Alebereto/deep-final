from glob import glob
from shutil import copyfile
from PIL import Image
import os
from tqdm import tqdm
import numpy as np


SEED = 123
ORIG_DIR = 'data\\food-101\\images'
NEW_DIR = 'data\\food'

if not os.path.isdir(NEW_DIR): os.mkdir(NEW_DIR)

np.random.seed(SEED)

dirs = os.listdir(ORIG_DIR)

# make directories in other folder
for dir in dirs:
    if not os.path.isdir(f'{NEW_DIR}\{dir}'): os.mkdir(f'{NEW_DIR}\{dir}')

# copy 500 images with dimensions 512 x 512
for i, dir in tqdm(enumerate(dirs), total=len(dirs)):
    good_paths = list()
    paths = glob(f'{ORIG_DIR}\{dir}\*.jpg')

    for path in paths:
        img = Image.open(path).convert("RGB")
        height, width = img.size
        if height == 512 and width == 512: good_paths.append(path)

    dest_path = f'{NEW_DIR}\{dir}'

    indeces = np.random.choice(len(good_paths), 500, replace=False)
    for n, idx in enumerate(indeces):
        copyfile(good_paths[idx], f'{dest_path}\{n+1}.jpg')
