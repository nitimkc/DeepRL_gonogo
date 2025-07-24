import numpy as np 
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

# import torch.nn.functional as F
# from stable_baselines3 import DQN

class ConcatImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, observations, input_dim, targets, render_mode=None):
        
        self.observations = observations
        self.targets = targets

        self.observation_shape = input_dim
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(2) # Two possible actions, 0 (no lick) and 1 (lick)
        
        self.render_mode = render_mode
        self.current_step = 0
        self.max_steps = 10 # the maximum number of discrete time steps per trial (episode) - this can be changed to better match the timings in the experiment

        self.num_trials = len(observations)
        self.trial_order = np.arange(self.num_trials)
        self.current_trial_idx = 0 # Training starts with the first trial

        self.last_action = None
        self.last_reward = None

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.current_step = 0
            
            # Move to the next trial
            self.current_trial_idx += 1
            if self.current_trial_idx >= self.num_trials:
                self.current_trial_idx = 0
            self.current_trial = self.trial_order[self.current_trial_idx]

            # obs = observations[self.current_trial, self.current_step] 
            obs = self.observations[self.current_trial][self.current_step]
            obs = self._get_cnn_prediction(obs) 
            info = {}
            return obs, info
        except Exception as e:
            print("Exception in step:", e)
            raise

    def step(self, action):
        try:
            target_seq = self.targets[self.current_trial]  # shape (T,)
            self.current_step += 1

            terminated = False
            truncated = self.current_step >= self.max_steps

            reward = 0
            # Determine time of first target==1 (animal lick) in the trial
            target_onset_indices = np.where(target_seq == 1)[0]
            target_onset = target_onset_indices[0] if len(target_onset_indices) > 0 else None

            if target_onset is not None:
                time_since_target = self.current_step - target_onset # Timesteps since animal has licked

                if action == 1:
                    if time_since_target >= 0:
                        reward = 1 * 0.5 ** max(0, time_since_target - 1)
                    else:
                        reward = -1
                    terminated = True  # trial ends on agent lick
            elif action == 1:
                reward = -1
                terminated = True

            if not terminated and not truncated:
                # obs = self.observations[self.current_trial, self.current_step]
                obs = self.observations[self.current_trial][self.current_step]
            else:
                obs = np.zeros(self.observation_shape, dtype=np.uint8)  # placeholder

            # For callback logs
            self.last_action = action
            self.last_reward = reward

            return obs, reward, terminated, truncated, {}
        except Exception as e:
            print("Exception in step:", e)
            raise

    def render(self):
        pass

    def close(self):
        pass


class FeatureExtractor(BaseFeaturesExtractor):
   
    def __init__(self, observation_space, model):
        
        super(FeatureExtractor, self).__init__(observation_space, features_dim=4096)  # Final feature dim

        self.model = model
        # self.transform = T.Compose([T.ToPILImage(),
        #                             T.Resize((224, 224)),
        #                             T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #                             ])
        # Freeze if necessary
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, observations):
        # observations [ batch_size, 918, 2348, 3]
        # x = self.transform(x)
        # with torch.no_grad():
        #     x = self.vgg(x)
        # return x
        return self.model(observations)


class CustomCNNPolicy(ActorCriticPolicy):
    
    def __init__(self, observation_space, action_space, lr_schedule, model, **kwargs):
        
        super(CustomCNNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )


