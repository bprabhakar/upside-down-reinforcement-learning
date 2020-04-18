import heapq
import random
import numpy as np


class ReplayBuffer(object):
    """Implemented as a priority queue, where the priority value is
    set to be episode's total reward. Note that unlike usual RL buffers,
    we store entire 'trajectories' together, instead of just transitions.
    """
    def __init__(self, size, seed=4076):
        self.size = size
        self.buffer = [] # initialized as a regular list; use heapq functions
        self.rg = np.random.RandomState(seed)

    def __getitem__(self, key):
        return self.buffer[key]
    
    def __len__(self):
        return len(self.buffer)

    def add_episode(self, S, A, R, S_):
        """ all inputs are numpy arrays; num_rows = timesteps
        S  : states
        A  : actions
        R  : rewards
        S_ : next states
        """
        episode = (S, A, R, S_)
        episode_reward = np.sum(R)
        if S.shape[0] > 1: # ignore episodes that only last 1 step
            item = (episode_reward, episode) # -1 for desc ordering
            if len(self.buffer) < self.size:
                heapq.heappush(self.buffer, item) 
            else:
                _ = heapq.heappushpop(self.buffer, item) # ignore the popped obj
    
    def top_episodes(self, K):
        """ Returns K episodes with highest total ep rewards.
        Output: [(state_arr, action_arr, reward_arr, next_state_arr), ... ]
        """
        episodes = [x[1] for x in self.buffer[-K:]] # buffer has (-reward, episode)
        return episodes

    def sample_episodes(self, K):
        """ Returns random K episodes.
        Output: [(state_arr, action_arr, reward_arr, next_state_arr), ... ]
        """
        sampled_items = random.choices(self.buffer, k=K)
        episodes = [x[1] for x in sampled_items] # buffer has (-reward, episode)
        return episodes
