import gym
from gym.spaces import Discrete
from gym.wrappers import RecordEpisodeStatistics

from rl import envs
from rl.envs.wrappers import ClipActions, FrameStack, NoopResetWrapper
from rl.envs.wrappers.atari_wrappers import make_atari, wrap_deepmind

def make_atari_env(
	gym_id: str,
	frame_size: int = 1,
	seed: int = 0,
	data_format: str = "channels_last"
) -> gym.Env:
	env = make_atari(gym_id)
	env = wrap_deepmind(
		env,
		frame_size=frame_size,
		scale=False,
		data_format=data_format
	)
	env.seed(seed)
	env.action_space.seed(seed)
	env.observation_space.seed(seed)
	return env

def make_env(
	gym_id: str,
	frame_size: int = None,
	is_lstm: bool = False,
	n_noops: int = None,
	seed: int = 0
) -> gym.Env:
	env = envs.make(gym_id)
	if not isinstance(env.action_space, Discrete):
		env = ClipActions(env)
	env = RecordEpisodeStatistics(env)
	if n_noops:
		env = NoopResetWrapper(env, n_noops)
	if frame_size:
		data_format = "channels_first" if is_lstm else "channels_last"
		env = FrameStack(env, frame_size=frame_size, data_format=data_format)
	env.seed(seed)
	env.action_space.seed(seed)
	env.observation_space.seed(seed)
	return env