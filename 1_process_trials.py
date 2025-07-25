
# import 
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse

import numpy as numpy

print('creating parser')
parser = argparse.ArgumentParser(description="Twitter ILI infection detection with LLMs")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
parser.add_argument("--filename", type=str, help="The name of session file to load.")
args = parser.parse_args()

ROOT = Path(args.root)
DATA = ROOT.joinpath(f"data/session")
DATA.mkdir(parents=True, exist_ok=True)

SAVEPATH = DATA.parent

    # images and hyperparameters for observations
    print('\ncreating image trial data')
    # ===========================================
    session_images_nums = ['075', '106', '073', '045', '035', '031', '000', '054'] 
    img_fnames = [i for i in DATAPATH.glob("*.png")]
    img_fnames = [i for i in img_fnames if i.name.split('.')[0][2:] in session_images_nums]
    session_data = pd.read_csv(DATAPATH.joinpath(f"session/{train_config['TRIALS']}"))
    
    N_TRIALS = len(session_data)
    N_TIMESTEPS = session_data['num_time_bins'].to_numpy().astype(int) # # N_TIMESTEPS varies each trial, make a numpy array of the num_time_bins column in session_data
    MAX_TIMESTEPS = session_data['num_time_bins'].max()
    # MAX_TIMESTEPS = int(session_data['change_time_bin'].max() + 5)
    print(f"N_TIMESTEPS shape: {N_TIMESTEPS.shape}")
    print(f"First few values of N_TIMESTEPS: {N_TIMESTEPS[:10]}")
    print(f"Maximum number of steps in a trials: {MAX_TIMESTEPS}")

    # CHANGE_TIME = session_data['change_time_bin'].to_numpy().astype(int)
    # TARGET_TIME = session_data['response_time_bin'].to_numpy()
    CHANGE_TIME = pd.Series(session_data['change_time_bin'], dtype="Int64")
    TARGET_TIME = pd.Series(session_data['response_time_bin'], dtype="Int64")
    print(f"CHANGE_TIME shape: {CHANGE_TIME.shape}")
    print(f"TARGET_TIME shape: {TARGET_TIME.shape}")

    unique_image_paths = [IMGPATH.joinpath(f"{i}.png") for i in session_data['initial_image_name'].unique()]
    # print(unique_image_paths)
    # unique_images = {img.stem: transform(Image.open(img)) for img in unique_image_paths}
    one_image =  np.array(Image.open(unique_image_paths[0]))
    IMG_SHAPE = one_image.shape
    print(f"Shape of image: {IMG_SHAPE}")

    # load img index instead of image as tensor will be too large to pass
    img_path_index = {img_path.stem: idx for idx, img_path in enumerate(unique_image_paths)}
    print(img_path_index)
    N_IMAGES = len(img_path_index)
    print(f"No. of unique images: {N_IMAGES}")
    
    # initialize tensor to save images
    # observations = torch.zeros((N_TRIALS, MAX_TIMESTEPS) + IMG_SHAPE, dtype=torch.uint8)
    observations = torch.zeros((N_TRIALS, MAX_TIMESTEPS) + (1,), dtype=torch.uint8) # placeholder for IMG_SHAPE is IMG_PATH
    print(f"Shape of tensor to store image: {observations.shape}")

    for t_idx, t in session_data.iterrows():
      n_steps = N_TIMESTEPS[t_idx]
      initial_imgidx, change_imgidx = img_path_index[t.loc['initial_image_name']], img_path_index[t.loc['change_image_name']]
      change_step = int(t['change_time_bin']) if t['is_change'] else None

      for s in range(n_steps):
        if t['is_change'] and change_step is not None and s >= change_step:
          observations[t_idx, s] = change_imgidx
        else:
          observations[t_idx, s] =  initial_imgidx  

    targets = session_data['is_change']
    # obs_shape = observations.shape[:-1] + IMG_SHAPE
    obs_shape = observations.shape
    print(f"No. of input observations: {len(observations)}  and targets: {len(targets)}")
    print(f"shape of input observation: {obs_shape}")