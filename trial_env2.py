import numpy as np 
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T 

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

def preprocess_tensor_batch(batch_tensor):
    """
    Input: batch_tensor of shape (B*N, H, W, C), dtype uint8 or float
    Output: preprocessed tensor of shape (B*N, 3, 224, 224), on GPU
    """
    # Convert to float32 and permute to (B*N, C, H, W)
    batch_tensor = batch_tensor.float().permute(0, 3, 1, 2) / 255.0  # (B*N, 3, H, W)

    # Resize using interpolate
    batch_tensor = F.interpolate(batch_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    # Normalize manually
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch_tensor.device).view(1, 3, 1, 1)
    batch_tensor = (batch_tensor - mean) / std

    return batch_tensor

class ConcatImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, observations, input_dim, cnn_model, targets, render_mode=None):
        
        self.observations = observations
        self.targets = targets
        
        self.observation_shape = input_dim
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(2) # Two possible actions, 0 (no lick) and 1 (lick)
        
        self.cnn_model = cnn_model # finetuned vgg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model.to(self.device)
        self.preprocess = T.Compose([T.ToPILImage(),
                                     T.Resize((224, 224)),
                                     T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     ])

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
            predictions = self._get_batchsequence_predictions() # get all predictions
            
            super().reset(seed=seed)
            self.current_step = 0
            
            # Move to the next trial
            self.current_trial_idx += 1
            if self.current_trial_idx >= self.num_trials:
                self.current_trial_idx = 0
            self.current_trial = self.trial_order[self.current_trial_idx]

            # for single step of a trial
            # obs = observations[self.current_trial, self.current_step] 
            # obs = self.observations[self.current_trial][self.current_step]
            # obs = self._get_cnn_prediction(obs) 

            # for all observations i.e. multiple trials with its own N steps
            obs = predictions[self.current_trial, self.current_step]

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

    def _get_cnn_prediction(self, img):
        """
        for single step of a trial 
        """
        transformed_img = self.preprocess(img).to(self.device).unsqueeze(0).to(self.device) # torch.Size([1, 3, 224, 224])
        with torch.no_grad():
            output = self.cnn_model(transformed_img)  # torch.Size([1, 2])
        pred = torch.argmax(output, dim=1).cpu().numpy().astype(np.float32)  # (1,) of either 0 or 1
        return pred
    
    def _get_sequence_predictions(self):
        """
        for all steps of a trial 
        """
        # Preprocess all 10 images of all trials at once
        self.batch_images = self.observations.view(-1, 3, 224, 224)  # Shape: (10, 3, 224, 224)

        # Model to eval mode and inference only
        self.cnn_model.eval()
        with torch.no_grad():
            logits = self.cnn_model(batch_images)  # (500, 2)
            preds = torch.argmax(logits, dim=1) 
        preds = preds.view(50, 10).cpu().numpy()
        return preds  # Still in temporal order

    def _get_batchsequence_predictions(self):
        """
        for all steps of all trials 
        """
        processed_batch = self.batch_preprocess() # (B*N, 3, 224, 224)
        self.cnn_model.eval()
        with torch.no_grad():
            logits = self.cnn_model(processed_batch)  # shape: (B*N, num_classes)
            preds = torch.argmax(logits, dim=1)       # (B*N, )
        batch_size = preds.size(0) // 10              # calculate batch size dynamically
        preds = preds.view(-1, 10).cpu().numpy()      # (B, 10)
        return preds 

    def batch_preprocess(self):
        """
        transform all images of all trials onn GPU
        """
        batch_size, n_steps, H, W, C = (self.num_trials, self.max_steps) + self.observation_shape
        obs_array = np.array(self.observations).reshape(-1, H, W, C) # Reshape to (B*N, H, W, C)
        obs_tensor = torch.from_numpy(obs_array).to(self.device)     # Convert to tensor and move to GPU
        processed_batch = preprocess_tensor_batch(obs_tensor)        # Preprocess entire batch on GPU

        # processed = []
        # for i in range(batch_size):
        #     for j in range(n_steps):
        #         img = self.observations[i][j]
        #         transformed_img = self.preprocess(img)  # tensor shape (3, 224, 224)
        #         processed.append(transformed_img)
        # processed_batch = torch.stack(processed)       # (batch_size*seq_len, 3, 224, 224)
        return processed_batch


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