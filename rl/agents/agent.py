import abc
from contextlib import redirect_stdout
import imageio
import os
import pickle
import time
from typing import Tuple, Union

import gym
import numpy as np
import tensorflow as tf
from tensorflow.train import Checkpoint, CheckpointManager

from rl.networks import Network
from rl.utils import ObsNormalizer, RewardNormalizer

class Agent(metaclass=abc.ABCMeta):
	def __init__(
		self,
		train_env: gym.Env,
		eval_env: gym.Env,
		gamma: float = 0.99,
		is_norm_obs: bool = True,
		is_norm_rewards: bool = True,
		folder_name: str = None
	) -> None:
		self.train_env = train_env
		self.eval_env = eval_env
		self.folder_dir = folder_name
		self.metrics_dict = {}

		if hasattr(train_env, "env_fns"):
			dummy_env = train_env.env_fns[0]()
		else:
			dummy_env = train_env
		data_format = getattr(dummy_env, "data_format", None)
		frame_size = getattr(dummy_env, "frame_size", None)
		if is_norm_obs:
			self.obs_norm = ObsNormalizer(
				shape=train_env.observation_space.shape[1:],
				frame_size=frame_size,
				data_format=data_format
			)
		else:
			self.obs_norm = None
		if is_norm_rewards:
			self.reward_norm = RewardNormalizer(gamma=gamma)
		else:
			self.reward_norm = None

		self.state = self.reset_env(is_eval=False)
		self.n_step = 0
		
		os.makedirs(self.folder_dir)
		os.makedirs(os.path.join(self.folder_dir, "metrics"), exist_ok=True)
		self.__save_summary()
		self.__set_checkpoint_manager()

	@abc.abstractmethod
	def train(self) -> None:
		raise NotImplementedError

	@abc.abstractmethod
	def _get_eval_action(self, state: np.ndarray) -> tf.Tensor:
		raise NotImplementedError

	def reset_env(self, is_eval: bool=False) -> np.ndarray:
		if is_eval:
			states = np.array(self.eval_env.reset())
		else:
			states = np.array(self.train_env.reset())
		states = self.__norm_obs(states, is_eval)
		return states

	def step(
		self,
		actions: np.ndarray,
		is_eval: bool = False
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
		if is_eval:
			env = self.eval_env
		else:
			env = self.train_env
		next_states, rewards, dones, infos = env.step(actions.numpy())
		next_states = self.__norm_obs(next_states, is_eval)
		rewards = self.__norm_rewards(rewards, dones, is_eval)
		return next_states, rewards, dones, infos

	def eval(
		self,
		fps: float = 60,
		max_video_length: float = 180,
		filename: str = None
	) -> None:
		start_time = time.time()
		print(" - Start evaluate...")
		max_step = max_video_length * fps
		self.eval_env.set_max_episode_steps(max_step)
		self.eval_env.override_num_noops = 0
		if filename is None:
			filename = f"evaluation_{self.n_step}"
		state = self.reset_env(is_eval=True)
		info = {}

		video = None
		while not "episode" in info:
			rgb_array = self.eval_env.render("rgb_array")
			if video is None and rgb_array is not None:
				filename += ".mp4"
				path = os.path.join(self.folder_dir, filename)
				video = imageio.get_writer(path, fps=fps)
			if video:
				video.append_data(rgb_array)
			action = self._get_eval_action(state)
			next_state, _, done, info = self.step(action, is_eval=True)
			if done and not "episode" in info:
				next_state = self.reset_env(is_eval=True)
			state = next_state

		reward = info["episode"]["r"]
		n_step = info["episode"]["l"]
		runtime = time.time() - start_time
		print("Reward: {:.2f} with frame count {}, runtime: {:.2f}".format(
			reward,
			n_step,
			runtime
		))

		if video:
			video.close()
			print("Video saved as", filename)
		elif hasattr(self.eval_env, "render_all"):
			filename += ".png"
			path = os.path.join(self.folder_dir, filename)
			self.eval_env.render_all(path)
			print("Plot saved as", filename)
		self.reset_env(is_eval=True)

	def __norm_obs(self, states: np.ndarray, is_eval: bool=False) -> np.ndarray:
		if self.obs_norm:
			return self.obs_norm(states, is_eval)
		else:
			return states

	def __norm_rewards(
		self,
		rewards: np.ndarray,
		dones: np.ndarray,
		is_eval: bool = False
	) -> np.ndarray:
		if self.reward_norm:
			return self.reward_norm(
				rewards=rewards,
				dones=dones,
				is_eval=is_eval
			)
		else:
			return rewards

	def save(self) -> None:
		start_time = time.time()
		self.__save_metrics()
		ckpt_manager = self.ckpt_manager
		del self.ckpt_manager
		path = os.path.join(self.folder_dir, "agent.pkl")
		with open(path, "wb") as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		self.ckpt_manager = ckpt_manager
		self.ckpt_manager.save()
		print(" - Saved agent at", self.folder_dir)
		runtime = time.time() - start_time
		print("Save runtime:", runtime)

	@classmethod
	def load(
		cls,
		folder_name: str
	) -> "Agent":
		print(f" - Restoring agent from {folder_name}...")
		path = os.path.join("trained", folder_name, "agent.pkl")
		with open(path, "rb") as f:
			agent = pickle.load(f)
		agent.__set_checkpoint_manager()
		print(" - Restored agent!")
		return agent

	def _append_metrics(
		self,
		key: str,
		value : Union[int, float, np.ndarray]
	) -> None:
		if key in self.metrics_dict:
			self.metrics_dict[key].append(value)
		else:
			self.metrics_dict[key] = [value]

	def __save_metrics(self) -> None:
		base_path = os.path.join(self.folder_dir, "metrics")
		for key, value in self.metrics_dict.items():
			path = os.path.join(base_path, key + ".pkl")
			with open(path, "ab") as f:
				for v in value:
					pickle.dump(v, f, pickle.HIGHEST_PROTOCOL)
		self.metrics_dict = {}

	def __set_checkpoint_manager(self) -> None:
		model_dict = {}
		self.model_config_dict = {}
		for key, value in vars(self).items():
			if isinstance(value, Network):
				model_dict[key] = value
				self.model_config_dict[key] = value.get_config()
		ckpt = Checkpoint(**model_dict)
		self.ckpt_manager = CheckpointManager(
			checkpoint=ckpt,
			directory=os.path.join(self.folder_dir, "models"),
			max_to_keep=2
		)
		ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
		if self.ckpt_manager.latest_checkpoint:
			print(" - Checkpoint available:", self.ckpt_manager.latest_checkpoint)
			print(" - Successfully restored to step", self.n_step)
		else:
			print(" - No checkpoint available at", os.path.join(self.folder_dir, "models"))

	def __save_summary(self) -> None:
		path = os.path.join(self.folder_dir, "summary.txt")
		networks = []
		with redirect_stdout(open(path, "w")):
			for key, value in self.__dict__.items():
				if isinstance(value, Network):
					networks.append(value)
				elif not (key in ["state", "done"]):
					print(f"{key}: {value}")
			for network in networks:
				print()
				network.summary()

	@property
	def folder_dir(self) -> str:
		return self._folder_dir
	
	@folder_dir.setter
	def folder_dir(self, folder_name: str) -> None:
		if folder_name is None:
			folder_name = self.__get_defualt_filename()
		self._folder_dir = os.path.join("trained", folder_name)

	@property
	def name(self) -> str:
		raise NotImplementedError	

	def __get_defualt_filename(self) -> dict:
		return "{}_{}".format(
			self.eval_env.unwrapped.spec._env_name,
			self.name,
		)
