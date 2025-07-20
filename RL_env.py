import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class concatImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        self.observation_shape = (224*2, 224, 1) # Image  shape is 224x224 and we concat two images. Third dimension is 1 because image is in black and white.
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float32) # Values in the image matrix are floats from 0 to 1.
        self.action_space = spaces.Discrete(2) # Two possible actions, 0 (no lick) and 1 (lick)
        self.render_mode = render_mode
        self.current_step = 0
        self.max_steps = 10 # the maximum number of discrete time steps per trial (episode) - this can be changed to better match the timings in the experiment

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = np.zeros(self.observation_shape, dtype=np.float32)  # At the beginning of a trial, just have an empty (entirely black) image
        info = {}
        return obs, info

    def step(self, action):
        # Get the current observation and target
        obs = observations[self.current_step] # TO DO: We need some code to define the observations and targets from the ground truth on each trial
        target = targets[self.current_step]

        # Increment the step counter
        self.current_step += 1

        # Calculate reward
        if action == target == 1:
            reward = 1.0  # Reward when action == target == 1, i.e., when the image has changed and the action is lick
        else:
            reward = 0.0  # No reward otherwise

        # Check termination and truncation conditions
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Return the observation, reward, and other information
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
