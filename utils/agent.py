import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)

        # Separate output layers for actor and critic
        self.actor_fc = nn.Linear(256, action_dim)
        self.critic_fc = nn.Linear(256, 1)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.activation(self.fc1(x))

        # Actor output (policy logits)
        policy_logits = self.actor_fc(x)

        # Critic output (value estimate)
        value = self.critic_fc(x)

        return policy_logits, value


class Agent(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        super(Agent, self).__init__()
        self.device = device
        self.agent = ActorCritic(state_dim[0], action_dim).to(device)

    def get_value(self, x):
        """
        Return "return" which is the total amount of reward an agent
        can expect to accumulate over the future,
        starting from that state and following the policy.
        """
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        _, value = self.agent(x)
        return value

    def get_action_and_value(self, x, action=None, training=False):
        """
        :return:
            action : action is taken by the agent
            log_prob : Logarithmic probability of the action for efficient computation.
            entropy : action entropy
            value : value provided the critic
        """
        if not training:
            # When sampling (not training), the state takes the numpy format and needs to be reformatted to the torch format
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        logits, value = self.agent(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


    def load_weights(self, agent_weights_path: str):
        try:
            self.agent.load_state_dict(torch.load(agent_weights_path, map_location=self.device))
            print(f"Weights loaded successfully!")
        except:
            print(f"Weights were not successfully loaded!")