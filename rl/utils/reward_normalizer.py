import numpy as np
from rl.utils import RunningMeanStd

class RewardNormalizer:
    def __init__(
        self,
        clip_value: float = 10.,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        self.rms = RunningMeanStd(shape=(1,))
        self.clip_value = clip_value
        self.gamma = gamma
        self.epsilon = epsilon
        self.norm_rewards = np.zeros(())

    def __call__(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        is_eval: bool = False
    ) -> np.ndarray:
        '''
		Update running mean std. and return normalized reward.
		Note that the first channel of reward indicates the batch size.
		'''
        self.norm_rewards = self.norm_rewards * self.gamma + rewards
        if not is_eval:
            self.rms.update(np.copy(self.norm_rewards))
        norm_rewards = np.clip(
            rewards / np.sqrt(self.rms.var + self.epsilon),
            -self.clip_value,
            self.clip_value
        )
        self.norm_rewards = self.norm_rewards * (1 - dones)
        return norm_rewards