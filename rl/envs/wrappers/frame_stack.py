from collections import deque
from gym import Env, Wrapper
from gym.spaces import Box
import numpy as np

class LazyFrames(object):
	def __init__(self, frames, data_format):
		"""This object ensures that common frames between the observations are only stored once.
		It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
		buffers.
		This object should only be converted to numpy array before being passed to the model.
		You'd not believe how complex the previous solution was."""
		self._frames = frames
		self._data_format = data_format
		self._out = None

	def _force(self):
		if self._out is None:
			if self._data_format == "channels_last":
				self._out = np.concatenate(self._frames, axis=-1)
			else:
				self._out = np.stack(self._frames)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		frames = self._force()
		return frames.shape[frames.ndim - 1]

	def frame(self, i):
		return self._force()[..., i]

class FrameStack(Wrapper):
	def __init__(
		self,
		env: Env,
		frame_size: int,
		data_format: str = "channels_last",
		starting_frames: np.ndarray = None
	) -> None:
		"""Stack k last frames.
		Returns lazy array, which is much more memory efficient.
		See Also
		--------
		baselines.common.atari_wrappers.LazyFrames
		"""
		assert data_format in ["channels_first", "channels_last"]
		assert (starting_frames is None) or (len(starting_frames) == frame_size)
		super().__init__(env)
		self.frame_size = frame_size
		self.data_format = data_format
		self.starting_frames = starting_frames

		self.frames = deque([], maxlen=frame_size)
		low = env.observation_space.low
		high = env.observation_space.high
		shape = env.observation_space.shape
		dtype = env.observation_space.dtype
		if data_format == "channels_last":
			shape = (shape[:-1] + (shape[-1] * frame_size,))
			low = np.repeat(low, frame_size, axis=-1)
			high = np.repeat(high, frame_size, axis=-1)
		else:
			shape = ((frame_size,) + shape)
			low = np.repeat(low[np.newaxis,:], frame_size, axis=0)
			high = np.repeat(high[np.newaxis,:], frame_size, axis=0)
		self.observation_space = Box(low=low, high=high, shape=shape, dtype=dtype)

	def reset(self):
		ob = self.env.reset()
		if self.starting_frames is not None:
			for starting_frame in self.starting_frames:
				self.frames.append(starting_frame)
		else:
			for _ in range(self.frame_size):
				self.frames.append(ob)
		return self._get_ob()

	def step(self, action):
		ob, reward, done, info = self.env.step(action)
		self.frames.append(ob)
		return self._get_ob(), reward, done, info

	def _get_ob(self):
		assert len(self.frames) == self.frame_size
		return LazyFrames(list(self.frames), self.data_format)
