import gym

class TimeLimit(gym.Wrapper):
	def __init__(self, env, max_episode_steps=None):
		super(TimeLimit, self).__init__(env)
		self._max_episode_steps = max_episode_steps
		self._elapsed_steps = 0

	def step(self, ac):
		observation, reward, done, info = self.env.step(ac)
		self._elapsed_steps += 1
		if self._max_episode_steps and self._elapsed_steps >= self._max_episode_steps:
			done = True
			info['TimeLimit.truncated'] = True
		return observation, reward, done, info

	def set_max_episode_steps(self, max_episode_steps):
		self._max_episode_steps = max_episode_steps