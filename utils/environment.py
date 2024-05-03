import cv2
import numpy as np
import gymnasium as gym
import datetime
step = 0


class CarEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(CarEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    def _preprocess(self, img):
        """
        :param img: frame
        :return: cropped mad normalized frame
        """
        img = img[:84, 6:90]  # CarRacing-v2-specific cropping
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
        return img

    def reset(self):
        """
        Modified reset function. Skip the first 50 frames, since it's just a zoom-up of the environment.
        :return: Four identical images in the stack.
        """
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
        """
        Take the step but skip the self.skip_frames steps to improve performance
        :param action: discrete action, e.g. 2.
        :return: Stacked state, reward after taking action, done, truncated (not used), info (not used)
        """
        reward = 0
        # We take an action for self.skip_frames steps
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)

            # Penalty for being on the grass when the green channel value exceeds 180.
            if np.mean(s[:, :, 1]) > 180.0:
                reward -= 0.05

            reward += r

            done = terminated or truncated
            if done:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = self._preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, done, truncated, info