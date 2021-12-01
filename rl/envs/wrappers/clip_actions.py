import gym
import numpy as np

class ClipActions(gym.Wrapper):
    def step(self, action):
        action = np.nan_to_num(action)
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )
        return self.env.step(action)