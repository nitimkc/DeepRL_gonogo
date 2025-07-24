
# import 
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse
import cv2
import numpy as np
from skimage.util import random_noise

print('creating parser')
parser = argparse.ArgumentParser(description="Twitter ILI infection detection with LLMs")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
args = parser.parse_args()

ROOT = Path(args.root)
DATA = ROOT.joinpath(f"data")
DATA.mkdir(parents=True, exist_ok=True)

SAVEPATH = DATA.joinpath(f"concatdata")
SAVEPATH.mkdir(parents=True, exist_ok=True)

# concat two images
img_fnames = [i for i in DATA.glob('*.png')]
# print(img_fnames[:2])

for img in img_fnames:
  img_fnames_copy = img_fnames.copy()
  for dup in img_fnames_copy:
    fimg1 = img.stem[-3:].replace("_","") # first image name
    fimg2 = dup.stem[-3:].replace("_","") # second image name
    print(fimg1, fimg2)
    slct_img_fnames = [img, dup]          # both image names

    slct_img = []
    for s_img in slct_img_fnames:
      read_img = cv2.imread(s_img)        # read image
      slct_img.append(read_img)           # append image to list
    concat_img = cv2.hconcat(slct_img)    # horizontally concat images in list

    to_path = SAVEPATH.joinpath(f"img_{fimg1}_{fimg2}.png")
    print(to_path)
    cv2.imwrite(to_path, concat_img)
    # plt.imshow(concat_img, cmap='gray')
    # plt.axis('off')
    # plt.savefig(to_path)                  # save concatnated images
    # # plt.show()