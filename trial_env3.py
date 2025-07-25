import numpy as np 
import gymnasium as gym
from gymnasium import spaces

from torchvision.io import read_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T 

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.callbacks import BaseCallback

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

    def __init__(self, observations, input_dim, image_paths, cnn_model, targets,  decay_type="exponential", decay_rate=0.9, render_mode=None):
        
        self.observations = observations
        self.targets = targets

        # load all images once
        self.image_paths = image_paths
        self.concat_image_path = self.image_paths[0].parents[1].joinpath('concatimg_lowres')
        self.cached_images = {}
        session_images_nums = ['075', '106', '073', '045', '035', '031', '000', '054'] 
        for name1 in session_images_nums:
            for name2 in session_images_nums:
                img_name = f"img_{name1}_{name2}.png"
                img_path = self.concat_image_path.joinpath(img_name)
                if img_path.exists():
                    self.cached_images[f"{name1}_{name2}"] = read_image(img_path)
                else:
                    # Handle missing files gracefully, e.g.:
                    print(f"Warning: Missing image {img_path}")

        self.observation_shape = input_dim
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(2) # Two possible actions, 0 (no lick) and 1 (lick)
        
        self.cnn_model = cnn_model # finetuned vgg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model.to(self.device)
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.render_mode = render_mode
        self.current_step = 0
        self.max_steps = 10 # the maximum number of discrete time steps per trial (episode) - this can be changed to better match the timings in the experiment

        self.num_trials = len(observations)
        self.trial_order = np.arange(self.num_trials)
        self.current_trial_idx = -1 # Training starts with the first trial

        self.last_action = None
        self.last_reward = None

        # Decay parameters
        self.decay_type = decay_type
        self.decay_rate = decay_rate

        # To log lick times
        self.lick_times = []

        # if not hasattr(self, "predictions"):
        #     print("Precomputing CNN predictions")
        #     self.predictions = self._get_batchsequence_predictions()
        
    def image_change_index(self):
        """
        change idx for whole observation 
        """
        obs_reshaped = self.observations.squeeze(-1)
        change_indices = []
        for i in range(obs_reshape.shaped[0]):
            x = obs_reshaped[i]
            diff = x[:-1] != x[1:]
            idx = torch.nonzero(diff).flatten()[0].item() + 1
            change_indices.append(idx)
        return (change_indices)

    def trial_index_to_observation(self, trial_img_idx):
        """
        get concatenated images for each step/observation of a trial 
        """
        if self.current_trial_idx % 50 == 0:
            print(f"{self.current_trial_idx + 1} trial")
        
        # print(trial_img_idx)
        x = trial_img_idx.flatten()
        diff = torch.nonzero(x[:-1] != x[1:]).flatten()
        change_idx = diff[0].item() + 1 if diff.numel() > 0 else len(x)
        # print(change_idx)
        
        unique_img_idx = trial_img_idx.unique() # 1 if else 2, 
        unique_img_idx = unique_img_idx[unique_img_idx != -1] # do not take -1
        # print(unique_img_idx)
        
        unique_img_names = [self.image_paths[idx].stem[2:] for idx in unique_img_idx]
        n_unique = len(unique_img_names)
        n_idx = len(trial_img_idx)

        # no change
        if n_unique == 1:
            name1 = unique_img_names[0]
            # print(name1)
            concatimg = self.cached_images[f"{name1}_{name1}"]
            trial_obs = concatimg.unsqueeze(0).repeat(n_idx, 1, 1, 1)
            # print(f"no image change shape: {trial_obs.shape}")

        # change
        elif n_unique == 2:
            name1, name2 = unique_img_names[0], unique_img_names[1]
            # print(name1, name2)
            concatimg1 = self.cached_images[f"{name1}_{name1}"]
            concatimg2 = self.cached_images[f"{name1}_{name2}"]
            concatimg3 = self.cached_images[f"{name2}_{name2}"]
            assert concatimg1.shape == concatimg2.shape == concatimg3.shape
            img_shape = concatimg1.shape # 3x369x944

            # obs wrt to image change
            before = concatimg1.unsqueeze(0).repeat(change_idx, 1, 1, 1)
            at = concatimg2.unsqueeze(0)
            after = concatimg3.unsqueeze(0).repeat(n_idx-change_idx - 1, 1, 1, 1)
            trial_obs = torch.cat([before, at, after], dim=0)

        else:
            print(f"More than 2 images in same trial: {n_unique} at step {self.current_step} of {self.current_trial_idx} trial")

        return trial_obs

    def reset(self, seed=None, options=None):
        try:                        
            super().reset(seed=seed)
            self.current_step = 0
            
            # Move to the next trial
            self.current_trial_idx += 1
            if self.current_trial_idx >= self.num_trials:
                self.current_trial_idx = 0
            self.current_trial = self.trial_order[self.current_trial_idx]

            # for single step of a trial
            # obs = self.observations[self.current_trial][self.current_step]
            # obs = self._get_cnn_prediction(obs) 

            # for all steps of a trial
            trial_img_idx = self.observations[self.current_trial]      # take all observations idx for one trial # 50x1
            trial_obs = self.trial_index_to_observation(trial_img_idx) # with concated images for each observation/step
            # print(trial_obs.shape)
            trial_obs = self._get_sequence_predictions(trial_obs)
            # print(trial_obs.shape)

            # for all observations i.e. multiple trials with its own N steps
            # obs = self.predictions[self.current_trial, self.current_step]

            info = {}
            return trial_obs, info
        except Exception as e:
            print("Exception in step:", e)
            raise

    def step(self, action):
        try:
            # target_seq = self.targets[self.current_trial]  # shape (T,)
            # self.current_step += 1

            # terminated = False
            # truncated = self.current_step >= self.max_steps

            # reward = 0
            # # Determine time of first target==1 (animal lick) in the trial
            # target_onset_indices = np.where(target_seq == 1)[0]
            # target_onset = target_onset_indices[0] if len(target_onset_indices) > 0 else None

            # if target_onset is not None:
            #     time_since_target = self.current_step - target_onset # Timesteps since animal has licked

            #     if action == 1:
            #         if time_since_target >= 0:
            #             reward = 1 * 0.5 ** max(0, time_since_target - 1)
            #         else:
            #             reward = -0.5
            #         terminated = True  # trial ends on agent lick
            # elif action == 1:
            #     reward = -0.5
            #     terminated = True

            # if not terminated and not truncated:
            #     obs = self.observations[self.current_trial, self.current_step]
            #     # obs = self.observations[self.current_trial][self.current_step]
            # else:
            #     obs = np.zeros(self.observation_shape, dtype=np.uint8)  # placeholder

            # # For callback logs
            # self.last_action = action
            # self.last_reward = reward

            # return obs, reward, terminated, truncated, {}

            # Determine time of first target==1 (animal lick) in the trial
            target_seq = self.targets[self.current_trial]  # shape (T,)
            self.current_step += 1

            terminated = False
            truncated = self.current_step >= self.max_steps

            reward = 0
            # Determine time of first target==1 (animal lick) in the trial
            target_seq = self.targets[self.current_trial]  # shape (T,)
            target_onset_indices = np.where(target_seq == 1)[0]
            target_onset = target_onset_indices[0] if len(target_onset_indices) > 0 else None
            time_since_target = None if target_onset is None else self.current_step - target_onset
            
            if target_onset is not None:
                if action == 1:  # agent licks
                    self.lick_times.append(self.current_step)
                    if time_since_target >= 0:
                        if self.decay_type == "exponential":
                            reward = 1 * (self.decay_rate ** max(0, time_since_target - 1))  # slower decay
                        elif self.decay_type == "linear":
                            # Example linear decay: reward decreases by decay_rate per timestep after target
                            reward = max(0, 1 - self.decay_rate * max(0, time_since_target - 1))
                        else:
                            reward = 1  # no decay fallback
                    else:
                        reward = -0.5  # reduced penalty for premature lick
                    terminated = True  # trial ends on lick
            elif action == 1:
                self.lick_times.append(self.current_step)
                reward = -0.5  # reduced penalty for lick when no target
                terminated = True

            if not terminated and not truncated:
                obs = self.observations[self.current_trial, self.current_step]
                # obs = self.observations[self.current_trial][self.current_step]
            else:
                obs = np.zeros(self.observation_shape, dtype=np.uint8)  # placeholder

            # For callback logs
            self.last_action = action
            self.last_reward = reward

            info = {}
            info['lick_times'] = self.lick_times 
            
            return obs, reward, terminated, truncated, info
        except Exception as e:
            print("Exception in step:", e)
            raise

    def render(self):
        pass

    def close(self):
        pass

    def _get_cnn_prediction(self, img_idx):
        """
        for single step of a trial 
        """
        transformed_img = self.preprocess(img).to(self.device).unsqueeze(0).to(self.device) # torch.Size([1, 3, 224, 224])
        with torch.no_grad():
            output = self.cnn_model(transformed_img)  # torch.Size([1, 2])
        pred = torch.argmax(output, dim=1).cpu().numpy().astype(np.float32)  # (1,) of either 0 or 1
        return pred
    
    def _get_sequence_predictions(self, trial_obs):
        """
        for all steps of a trial 
        """
        # Preprocess all images of a trials at once
        batch_transformed = torch.stack([self.preprocess(img) for img in trial_obs]) # 50*3*224*224
        self.cnn_model.eval()
        with torch.no_grad():
            logits = self.cnn_model(batch_transformed)  # (50, 2)
            preds = torch.argmax(logits, dim=1)         # (50,  )
        preds = preds.unsqueeze(1).detach().cpu().numpy()
        return preds  

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
        print(obs_array.shape)
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

class PrintCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:  # print every 100 steps
            print(f"Step: {self.n_calls}, Reward: {self.locals.get('rewards', 'N/A')}")
        return True