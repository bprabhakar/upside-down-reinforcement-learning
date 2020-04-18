from torch import nn


class BehaviorNet(nn.Module):
	""" Policy network takes state and target commands as input
	and returns probability distribution over all actions.
	"""
    def __init__(self, state_dim, action_dim):
        super(BehaviorNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + 2, 32),  # 2 = command_dim
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, features):
        """
        Input: features = [state command_steps command_reward] x batch_size
        Output: action_probs = [action_dim] x batch_size
        """
        logprobs = self.model(features)
        return logprobs


