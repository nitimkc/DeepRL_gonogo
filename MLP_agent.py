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

import torch
from cnn_model import VGG_finetuned
from trial_env2 import ConcatImageEnv

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

print('creating parser')
parser = argparse.ArgumentParser(description="DESCCRIPTION HERE")
parser.add_argument("--root", type=str, help="root directory where data and eval folders exists.")
parser.add_argument("--config_file", type=str, help="File containing run configurations.")
args = parser.parse_args()

ROOT = Path(args.root)
DATAPATH = ROOT.joinpath("data")
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
    N_CLASSES = len(img_fnames)
    N_SAME = train_config["N_TRIALS"] // 2
    N_DIFF = train_config["N_TRIALS"] // 2
    N_TIMESTEPS = train_config['N_TIMESTEPS']

    observations = []
    targets = []

    # Same image trials i.e. concat same images <<<changed to horizontal stack>>>
    for _ in range(N_SAME):
        cls = random.choice(range(N_CLASSES))
        # print(cls)
        img = cv2.imread(img_fnames[cls])
        obs = np.concatenate([img, img], axis=1)  # NOT vertical stack
        tiled_obs = np.tile(obs[np.newaxis, ...], (N_TIMESTEPS, 1, 1, 1))  # Tile the image for each timestep
        # print(tiled_obs.shape)

        observations.append(tiled_obs)
        targets.append([0] * N_TIMESTEPS)  # Target is 0 for all timesteps
        # plt.imshow(obs, cmap="gray")
    print(f"shape of input image: {img.shape}")
    print(f"shape of input observation: {tiled_obs.shape}")
    print(f"No. of input observations: {len(observations)}")
    
    # Change image trials
    for _ in range(N_DIFF):
        c1, c2 = random.sample(range(N_CLASSES), 2)
        # print(c1, c2)
        img1 = cv2.imread(img_fnames[c1])
        img2 = cv2.imread(img_fnames[c2])
        obs = np.concatenate([img1, img1], axis=1)  # Initially concatenate c1 and c1 HORIZONTALLY
        tiled_obs = np.tile(obs[np.newaxis, ...], (N_TIMESTEPS, 1, 1, 1))  # Tile the image for each timestep
        # print(tiled_obs.shape)
        change_timestep = random.randint(5, N_TIMESTEPS)  # Randomly choose a timestep between 5 and 10
        tiled_obs[change_timestep - 1] = np.concatenate([img1, img2], axis=1)  # Overwrite with c1 and c2 at the chosen timestep
        tiled_obs[change_timestep:] = np.concatenate([img2, img2], axis=1)  # Overwrite with c2 and c2 after the chosen timestep

        target = [0] * N_TIMESTEPS  # Target is 0 for all timesteps
        target[change_timestep - 1] = 1  # Set target to 1 at the change timestep

        observations.append(tiled_obs)
        targets.append(target)
    obs_shape = tiled_obs[1].shape
    print(f"No. of input observations: {len(observations)}")
    print(f"shape of input observation: {obs_shape}")
    
    # set up custom environment
    print('\n Setting custom environment')
    # ====================================
    # CNN
    vgg = VGG_finetuned(num_classes=2)
    vgg.load_state_dict(torch.load(CNN_PATH, weights_only=True)) # inplace=True

    # wrap env and cnn
    env = DummyVecEnv([lambda: ConcatImageEnv(observations=observations, 
                                              input_dim=obs_shape, 
                                              cnn_model=vgg,
                                              targets=targets
                                              )]) # for non-image
    print("Environment loaded:", env)
    # # to parallelize env process
    # from stable_baselines3.common.vec_env import SubprocVecEnv
    # env = SubprocVecEnv([make_env_fn for _ in range(num_envs)])

    env = VecMonitor(env)

    # train with PPO
    print('\n training with PPO')
    # ====================================
    # model = PPO("MlpPolicy", env, verbose=1)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256)
    # model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64)

    # model.learn(total_timesteps=100_000, log_interval=1)
    model.learn(total_timesteps=10_000, log_interval=1)
    print("Training complete.")

    print('\nsave and evaluate')
    # ==========================
    model.save(RL_PATH.joinpath("ppo_vgg_policy"))

    # evaluate
    n_trials = train_config['N_TRIALS']
    obs = env.reset()
    for _ in range(n_trials):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        if done:
            obs = env.reset()
