import time
from pathlib import Path
import yaml 
import logging
import argparse

import gymnasium as gym 
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import PPO

import random
import cv2
import numpy as np
import pandas as pd

import torch
from PIL import Image
from torchvision import transforms
from cnn_model import VGG_finetuned
# from trial_env2 import ConcatImageEnv, PrintCallback
from trial_env3 import ConcatImageEnv, PrintCallback

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

print('creating parser')
parser = argparse.ArgumentParser(description="DESCRIPTION HERE")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
parser.add_argument("--config_file", type=str, help="File containing run configurations.")
args = parser.parse_args()

ROOT = Path(args.root)
DATAPATH = ROOT.joinpath("data")
IMGPATH = DATAPATH.joinpath("images_lowres")
CONFIG_FILE = ROOT.joinpath(args.config_file)
with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)
train_config = CONFIG["TRAINING_RL"]
print(train_config)

CNN_PATH = ROOT.joinpath("cnn_models").joinpath(train_config["CNN_NAME"])

RL_PATH = ROOT.joinpath("rl_models")
RL_PATH.mkdir(parents=True, exist_ok=True)

log_dir = RL_PATH.joinpath("logs")
log_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    
    # images and hyperparameters for observations
    print('\ncreating image trial data')
    # ===========================================
    session_images_nums = ['075', '106', '073', '045', '035', '031', '000', '054'] 
    img_fnames = [i for i in DATAPATH.glob("*.png")]
    img_fnames = [i for i in img_fnames if i.name.split('.')[0][2:] in session_images_nums]
    session_data = pd.read_csv(DATAPATH.joinpath(f"session/{train_config['TRIALS']}"))
    session_data = session_data.iloc[:train_config['N_TRIALS'],:]

    N_TRIALS = len(session_data)
    # N_TIMESTEPS varies each trial, make a numpy array of the num_time_bins column in session_data
    N_TIMESTEPS = session_data['num_time_bins'].to_numpy().astype(int)
    MAX_TIMESTEPS = session_data['num_time_bins'].max()
    # Check the shape of N_TIMESTEPS
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
    observations = torch.full((N_TRIALS, MAX_TIMESTEPS, 1), -1, dtype=torch.int8)
    print(f"Shape of tensor to store image: {observations.shape}")

    for t_idx, t in session_data.iterrows():
      n_steps = N_TIMESTEPS[t_idx]
      initial_imgidx, change_imgidx = img_path_index[t.loc['initial_image_name']], img_path_index[t.loc['change_image_name']]
      change_step = int(t['change_time_bin'])  if t['is_change'] else None
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

    # save for later
    t_np = observations.squeeze(-1).cpu().numpy().astype(int)
    np.savetxt(DATAPATH.joinpath('t_np.csv'), t_np, delimiter=',', fmt='%d')

    # set up custom environment
    print('\n Setting custom environment')
    # ====================================
    # CNN
    vgg = VGG_finetuned(num_classes=2)
    vgg.load_state_dict(torch.load(CNN_PATH, weights_only=True)) # inplace=True

    # wrap env and cnn
    env = DummyVecEnv([lambda: ConcatImageEnv(observations=observations, 
                                              input_dim=obs_shape, 
                                              image_paths=unique_image_paths,
                                              cnn_model=vgg,
                                              targets=targets
                                              )]) # for non-image / image_idx
    print("Environment loaded:", env)
    # # to parallelize env process
    # from stable_baselines3.common.vec_env import SubprocVecEnv
    # env = SubprocVecEnv([make_env_fn for _ in range(num_envs)])
    env = VecMonitor(env)

    # train with PPO
    print('\nTraining with PPO')
    # ====================================
    start = time.time()
    model = PPO(
      "MlpPolicy", 
      env, 
      batch_size=64,
      n_steps=2048,
      learning_rate=3e-4,
      verbose=1,
      device='cuda'
      )

    callback = PrintCallback()
    model.learn(total_timesteps=10_000, callback=callback)

    end = time.time()
    print(f"Time taken for 500 timesteps: {end - start:.2f} seconds")
    print("Training complete.")

    print('\nsave and evaluate')
    # ==========================
    fname = train_config['MODEL_NAME']
    model.save(RL_PATH.joinpath(fname))

    # evaluate
    n_trials = train_config['N_TRIALS']
    obs = env.reset()
    actions = []
    for _ in range(n_trials):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        if terminated:
            obs = env.reset()
    # Convert to numpy array and save
    actions = np.array(actions)
    np.save(RL_PATH.joinpath(f"{fname}_actions.npy", actions))
