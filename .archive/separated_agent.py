import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, output_dim=1, activation=F.relu):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ActorCritic(nn.Module):
  def __init__(self,
               state_dim,
               action_dim,
               actor_lr,
               critic_lr,
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               ):
    super(ActorCritic, self).__init__()
    self.device = device
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr
    # Actor - policy | Critic - value function
    self.actor = Actor(state_dim[0], action_dim).to(self.device)
    self.critic = Critic(state_dim[0]).to(self.device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

  def get_value(self, x):
    """
    Return "return" which is the total amount of reward an agent
    can expect to accumulate over the future,
    starting from that state and following the policy.
    """
    x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
    return self.critic(x)

  def get_action_and_value(self, x, action=None, training=False):
    if not training:
        # When sampling (not training), the state takes the numpy format and needs to be reformatted to the torch format
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
    logits = self.actor(x)
    probs = Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(x)

  def update_actor(self, actor_loss):
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
      self.actor_optimizer.step()


  def update_critic(self, critic_loss):
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
      self.critic_optimizer.step()

  def load_weights(self, actor_w_path: str, critic_w_path: str):
      try:
          self.actor.load_state_dict(torch.load(actor_w_path, map_location=self.device))
          self.critic.load_state_dict(torch.load(critic_w_path, map_location=self.device))
          print(f"Weights loaded successfully!")
      except:
          print(f"Weights were not successfully loaded!")