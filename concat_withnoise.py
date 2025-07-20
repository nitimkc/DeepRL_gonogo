
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
parser.add_argument("--run_no", type=str, help="The number of run for adding noise to same pair of images.")
args = parser.parse_args()

ROOT = Path(args.root)
DATA = ROOT.joinpath(f"data")
DATA.mkdir(parents=True, exist_ok=True)

SAVEPATH = DATA.joinpath(f"concatdata")
SAVEPATH.mkdir(parents=True, exist_ok=True)

# concat two images
img_fnames = [i for i in DATA.glob('*.png')]
# print(img_fnames[:2])

# concat two images with noise in the second one
N_RUN = Path(args.run_no)
for img in img_fnames:
  img_fnames_copy = img_fnames.copy()
  for dup in img_fnames_copy:
    fimg1 = img.stem[-3:].replace("_","") # first image name
    fimg2 = dup.stem[-3:].replace("_","") # second image name
    # print(fimg1, fimg2)
    slct_img_fnames = [img, dup]          # both image names

    slct_img = []
    for idx, s_img in enumerate(slct_img_fnames):
      read_img = cv2.imread(s_img)        # read image
      if idx > 0:
        # print(idx)
        noise_level = random.uniform(0.1,0.31)
        noise = random_noise(read_img, mode='s&p', amount=noise_level) # add salt-and-pepper noise to the second image
        read_img = np.array(255 * noise, dtype=np.uint8)               # change back to image size
      slct_img.append(read_img)           # append image to list
    concat_img = cv2.hconcat(slct_img)    # horizontally concat images in list

    to_path = SAVEPATH.joinpath(f"img_{fimg1}_{fimg2}_noise{N_RUN}.png")
    print(to_path)
    # plt.imshow(concat_img, cmap='gray')
    plt.axis('off')
    plt.savefig(to_path)                  # save concatnated images