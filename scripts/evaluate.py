import cv2
import torch
import gymnasium as gym
import tqdm
import numpy as np

import torch.nn as nn

from utils.agent import ActorCritic
from utils.agent_vr2 import Agent
    


def _preprocess(img):
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img



class CareEnvEvaluate(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(CareEnvEvaluate, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    def _preprocess(self, img):
        img = img[:84, 6:90]  # CarRacing-v2-specific cropping
        # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
        return img

    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)

        # Convert a frame to 84 X 84 gray scale one
        s = self._preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info

    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                print(terminated, truncated)
                break

        frame = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
        from pathlib import Path
        path_to_save = Path("../evaluation_replays/exp_7/rollout_3/")
        if not path_to_save.exists():
            path_to_save.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(f"../evaluation_replays/exp_5/rollout_1/step_{step}.jpg", frame)
        cv2.imwrite(str(path_to_save / f"step_{step}.jpg"), frame)
        # Convert a frame to 84 X 84 gray scale one
        s = self._preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info

if __name__ == "__main__":
    eval_env = gym.make("CarRacing-v2", continuous=False)
    eval_env = CareEnvEvaluate(eval_env)
    agent = Agent((4, 84, 84), 5)
    agent.load_state_dict(torch.load("result_exp/exp_7/models/agent_best_reward_4.06.pt"))

    (s, _), done, ret = eval_env.reset(), False, 0
    step = 0
    scores = 0
    while not done:
        step += 1
        a, _, _, _ = agent.get_action_and_value(s)
        s_prime, r, terminated, truncated, _ = eval_env.step(a.cpu().numpy()[0])
        s = s_prime
        ret += r
        done = terminated or truncated
    scores += ret


