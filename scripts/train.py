import torch
import gymnasium as gym
import tqdm
import numpy as np

import torch.nn as nn
import torch.optim as optim

from utils.environment import CarEnv
from utils.agent import Agent
from utils.storage import SampleStorage
from utils.tools import save_returns, save_loss
from pathlib import Path



# Parameters
log = True
state_dim = (4, 84, 84)
total_steps = 0
action_dim = 5         # Discrete action space (Don't change it)
num_steps = 1024       # the number of steps to be performed in the environment with each policy rollout. Divide by 4
max_steps = 1_024_000  # the max number of steps per complete training
gamma = 0.99           # Gamma, parameter in advantage computation
num_epochs = 4         # Number of epochs to update the actor model
gae_lambda = 0.95      # GAE parameter
clip_coef = 0.2        # Clipping coefficient
norm_adv = True        # Advantage normalization

# Training Parameters
lr = 0.00025
actor_lr = 0.00025      # If separated actor and critic architecture is used
critic_lr = 0.00025     # If separated actor and critic architecture is used
c2 = 0.01               # Coefficient of the entropy
c1 = 1.0                # Coefficient of the value function
max_grad_norm = 0.5     # the maximum norm for the gradient clipping
num_minibatches = 32
batch_size = num_steps
mini_batch_size = batch_size // num_minibatches
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

# Evaluation Parameters
n_evals=5
eval_interval = 10 * num_steps    # Evaluate the police each 10 rollouts
best_avg_return = -np.inf


def evaluate(n_evals=5):
  """
  Take an actual trained agent and evaluate the policy.
  :param n_evals: number of evaluations (games)
  :return: average reward
  """
  eval_env = gym.make("CarRacing-v2", continuous=False)
  eval_env = CarEnv(eval_env)

  scores = 0
  for _ in tqdm.tqdm(range(n_evals)):
    (s, _), done, ret = eval_env.reset(), False, 0
    while not done:
      a, _, _, _ = agent.get_action_and_value(s)
      s_prime, r, terminated, truncated, info = eval_env.step(a.cpu().numpy()[0])
      s = s_prime
      ret += r
      done = terminated or truncated
    scores += ret

  return np.round(scores / n_evals, 4)


if __name__ == "__main__":
  env = gym.make("CarRacing-v2", continuous=False)
  env = CarEnv(env)

  # Agent initialization
  agent = Agent(state_dim, action_dim)
  optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

  # Storage initialization
  storage = SampleStorage(num_steps, state_dim)

  # Main loop
  while total_steps < max_steps:
    (next_obs, _), done, _ = env.reset(), False, None

    # Rollouts sampling
    episode = total_steps // num_steps + 1
    for step in tqdm.tqdm(range(num_steps)):
      total_steps += 1
      with torch.no_grad():
        action, log_prob, _, value = agent.get_action_and_value(next_obs)
      next_obs, reward, next_done, _, _ = env.step(action.cpu().numpy()[0])
      storage.update(next_obs, action, log_prob, reward, done, value)
      done = next_done

      if done:
        (next_obs, _), done, _ = env.reset(), False, None

    # Advantage method: GAE
    # A(t) = delta(t) + (gamma * lambda)*delta(t+1) + (gamma * lambda)^2*delta(t+2)
    #      where delta(t) = R + V(St+1) - V(St)
    with torch.no_grad():
      advantages = torch.zeros_like(storage.rewards).to(device)
      lastgaelam = 0
      for t in reversed(range(num_steps)):
        if t == num_steps - 1:
          nextnonterminal = 1.0 - next_done
          nextvalues = agent.get_value(next_obs)
        else:
          nextnonterminal = 1.0 - storage.done[t + 1]
          nextvalues = storage.values[t + 1]
        delta = storage.rewards[t] + gamma * nextvalues * nextnonterminal - storage.values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
      returns = advantages + storage.values

    # Flatteining
    b_actions = storage.actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = storage.values.reshape(-1)
    b_logprobs = storage.logprobs.reshape(-1)

    # Training model
    b_inds = np.arange(batch_size)
    for epoch in range(num_epochs):
      np.random.shuffle(b_inds)
      for start in range(0, batch_size, mini_batch_size):
        end = start + mini_batch_size
        mb_inds = b_inds[start:end]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(x=storage.obs[mb_inds],
                                                                      action=b_actions.long()[mb_inds],
                                                                      training=True
                                                                      )
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        if norm_adv:
          b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        mb_advantages = b_advantages[mb_inds]

        # Policy loss
        pg_loss1 = mb_advantages * ratio
        pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

        # Value loss from the PPO Paper : V(st) - R
        newvalue = newvalue.view(-1)
        mse_loss = nn.MSELoss()
        v_loss = mse_loss(newvalue, b_returns[mb_inds])

        # Entropy bonus
        entropy_loss = entropy.mean()

        # L(PG) - Entropy Bonus
        actor_loss = pg_loss - c2 * entropy_loss
        # L(VF)
        critic_loss = c1 * v_loss

        # Loss = pg_loss - c2 * entropy_loss + c1 * v_loss
        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

      if log:
        print(f"{episode} : Actor Loss {round(float(actor_loss), 4)}, Critic Loss : {round(float(critic_loss), 4)}, Entropy : {round(float(entropy.mean()), 4)}")

    save_loss(
      episode=episode,
      actor_loss=round(float(actor_loss), 4),
      critic_loss=round(float(critic_loss), 4),
      entropy=round(float(entropy.mean()), 4)
    )

    if total_steps % eval_interval == 0:
      ret = evaluate(n_evals)
      save_returns(episode=episode, ret=ret)
      print(f"VALIDATION : Return : {ret} - episode {episode}")

      if ret > best_avg_return:
        best_avg_return = ret
        model_save_path = Path("../models/")
        if not model_save_path.exists():
          model_save_path.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), str(model_save_path / f"agent_best_reward_{best_avg_return}.pt"))
        print(f"Best model updated at episode {episode} with Avg Return: {ret}")

    if total_steps > max_steps:
      model_save_path = Path("../models/")
      if not model_save_path.exists():
        model_save_path.mkdir(parents=True, exist_ok=True)
      torch.save(agent.state_dict(), str(model_save_path / f"last_agent_{best_avg_return}.pt"))
      print(f"The training has been completed!")
      break
  env.close()