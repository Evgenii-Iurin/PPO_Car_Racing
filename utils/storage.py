import torch

class SampleStorage():
    def __init__(self, num_steps, state_dim):
        """
        num_steps : number of steps per rollout
        state_dim : state dimension
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs = torch.zeros((num_steps, *state_dim)).to(self.device)
        self.actions = torch.zeros((num_steps, 1)).to(self.device)
        self.logprobs = torch.zeros((num_steps, 1)).to(self.device)
        self.rewards = torch.zeros((num_steps, 1)).to(self.device)
        self.done = torch.zeros((num_steps, 1)).to(self.device)
        self.values = torch.zeros((num_steps, 1)).to(self.device)

        self.index = 0
        self.max_size = num_steps

    def update(self, obs, actions, logprobs, rewards, done, values):
        """
        Update memory. It doesn't reset, it just overwrites the instances
        """
        new_obs = torch.from_numpy(obs)
        self.obs[self.index] = new_obs
        self.actions[self.index] = actions
        self.logprobs[self.index] = logprobs
        self.rewards[self.index] = rewards
        self.done[self.index] = done
        self.values[self.index] = values

        self.index = (self.index + 1) % self.max_size