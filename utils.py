import random
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset


def augment_state(state, command, command_scale):
	""" Appends scaled command values (horizon, reward) to the 
	original state vector.
	"""
	tgt_horizon, tgt_return = command
	horizon_scale, return_scale = command_scale
	horizon_scale = 0; return_scale = 0
	tgt_horizon *= horizon_scale
	tgt_return *= return_scale
	state_ = np.append(
		state, [tgt_horizon, tgt_return]
	)
	return state_

class BehaviorDataset(TorchDataset):
	""" Samples behavior segments for supervised learning 
	from given input episodes.
	"""
    def __init__(self, episodes, size, horizon_scale, return_scale):
        super(BehaviorDataset, self).__init__()
        self.episodes = episodes
        self.horizon_scale = horizon_scale
        self.return_scale = return_scale
        self.size = size

    def __len__(self):
        # just returning a placeholder number for now
        return self.size

    def __getitem__(self, idx):
        # get episode
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        # randomly sample an episode
        episode = random.choice(self.episodes)
        S, A, R, S_ = episode

        # extract behavior segment
        episode_len = S.shape[0]
        start_index = np.random.choice(episode_len - 1) # ensures cmd_steps >= 1
        command_horizon = (episode_len - start_index - 1)
        command_return = np.sum(R[start_index:])
        command = command_horizon, command_return
        command_scale = self.horizon_scale, self.return_scale

        # construct sample
        features = augment_state(
            S[start_index,:], command, command_scale
        )
        label = A[start_index]               # action taken
        sample = {
            'features': torch.tensor(features, dtype=torch.float), 
            'label': torch.tensor(label, dtype=torch.long) # categorical val
        }        
        return sample

        
