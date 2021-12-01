from enum import Enum
import numpy as np
from rl.utils import RunningMeanStd

class DataFormat(Enum):
	NO_STACK = 0
	CHANNELS_FIRST = 1
	CHANNELS_LAST = 2

class ObsNormalizer:
	def __init__(
		self,
		shape,
		clip_value: float = 10.,
		frame_size: int = None,
		data_format: str = None,
		epsilon: float = 1e-8
	) -> None:
		self.clip_value = clip_value
		self.frame_size = frame_size
		self.epsilon = epsilon
		if data_format is None:
			self.data_format = DataFormat.NO_STACK
		elif data_format == "channels_first":
			self.data_format = DataFormat.CHANNELS_FIRST
		elif data_format == "channels_last":
			self.data_format = DataFormat.CHANNELS_LAST
			self.data_size = shape[-1] // frame_size
		else:
			raise ValueError("Unresolved value data_format.")
		shape = self.__get_shape(shape)
		self.rms = RunningMeanStd(shape=shape)

	def __call__(
		self,
		obs: np.ndarray,
		is_eval: bool = False
	) -> np.ndarray:
		'''
		Update running mean std. and return normalized obs.
		Note that the first channel of obs indicates the batch size.
		'''
		if not is_eval:
			self.__update(obs)
		norm_obs = self.__get_norm_obs(obs)
		return norm_obs
	
	def __get_shape(self, shape):
		if self.data_format == DataFormat.NO_STACK:
			return shape
		elif self.data_format == DataFormat.CHANNELS_FIRST:
			return shape[1:]
		elif self.data_format == DataFormat.CHANNELS_LAST:
			return shape[:-1] + (self.data_size,)

	def __update(self, obs:np.ndarray) -> None:
		if self.data_format == DataFormat.NO_STACK:
			update_obs = obs
		elif self.data_format == DataFormat.CHANNELS_FIRST:
			update_obs = obs[:,-1]
		elif self.data_format == DataFormat.CHANNELS_LAST:
			update_obs = obs[..., -self.data_size:]
		self.rms.update(update_obs)

	def __get_norm_obs(self, obs:np.ndarray) -> np.ndarray:
		mean, var = self.rms.mean, self.rms.var
		if self.data_format == DataFormat.CHANNELS_FIRST:
			mean = np.tile(mean, (self.frame_size, 1))
			var = np.tile(var, (self.frame_size, 1))
		elif self.data_format == DataFormat.CHANNELS_LAST:
			mean = np.tile(mean, self.frame_size)
			var = np.tile(var, self.frame_size)
		norm_obs = np.clip(
			(obs - mean) / np.sqrt(var + self.epsilon),
			-self.clip_value,
			self.clip_value
		)
		return norm_obs