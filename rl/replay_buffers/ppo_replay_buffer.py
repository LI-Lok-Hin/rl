from typing import Tuple
import gym
from gym import spaces
import numpy as np
import tensorflow as tf

class PPOReplayBuffer:
	def __init__(
		self,
		capacity: int,
		n_env: int,
		state_space: gym.Space,
		action_space: gym.Space,
		gamma: float = 0.99,
		lam: float = 0.99
	) -> None:
		if isinstance(action_space, spaces.Tuple):
			state_shape = state_space.shape[1:]
			action_shape = action_space[0].shape
			action_dtype = action_space[0].dtype
		else:
			state_shape = state_space.shape
			action_shape = action_space.shape
			action_dtype = action_space.dtype
		self.state_buffer = np.zeros(
			(capacity, n_env) + state_shape,
			dtype = state_space.dtype
		)
		self.action_buffer = np.zeros(
			(capacity, n_env) + action_shape,
			dtype = action_dtype
		)
		self.reward_buffer = np.zeros(
			(capacity, n_env),
			dtype = np.float32
		)
		self.done_buffer = np.zeros(
			(capacity+1, n_env),
			dtype = np.bool
		)
		self.advantage_buffer = np.zeros(
			(capacity, n_env),
			dtype = np.float32
		)
		self.return_buffer = np.zeros(
			(capacity, n_env),
			dtype = np.float32
		)
		self.value_buffer = np.zeros(
			(capacity+1, n_env),
			dtype = np.float32
		)
		self.log_prob_buffer = np.zeros(
			(capacity, n_env),
			dtype = np.float32
		)
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.capacity = capacity
		self.n_env = n_env
		self.gamma = gamma
		self.lam = lam
		self.cursor = 0

	def add(
		self,
		state: np.ndarray,
		action: np.ndarray,
		reward: np.ndarray,
		done: np.ndarray,
		value: np.ndarray,
		log_prob: np.ndarray
	) -> None:
		self.state_buffer[self.cursor] = state
		self.action_buffer[self.cursor] = action
		self.reward_buffer[self.cursor] = reward
		self.done_buffer[self.cursor] = done
		self.value_buffer[self.cursor] = value
		self.log_prob_buffer[self.cursor] = log_prob
		self.cursor += 1

	def get(
		self,
		n_batch: int
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		indices = np.arange(self.capacity * self.n_env)
		np.random.shuffle(indices)
		shuffled_indices = indices.reshape((n_batch, -1))
		batch_states = self.state_buffer.reshape((-1,) + self.state_shape)[shuffled_indices]
		batch_actions = self.action_buffer.reshape((-1,) + self.action_shape)[shuffled_indices]
		batch_advantages = self.advantage_buffer.reshape((-1))[shuffled_indices]
		batch_returns = self.return_buffer.reshape((-1))[shuffled_indices]
		batch_log_probs = self.log_prob_buffer.reshape((-1))[shuffled_indices]
		batch_values = self.value_buffer.reshape((-1))[shuffled_indices]
		
		advs_mean = np.mean(batch_advantages, axis=1)[:,np.newaxis]
		advs_std = np.std(batch_advantages, axis=1)[:,np.newaxis]
		batch_norm_advs = (batch_advantages - advs_mean) / (advs_std + 1e-8)
		
		return (
			batch_states,
			batch_actions,
			batch_norm_advs,
			batch_returns,
			batch_log_probs,
			batch_values
		)
	
	def finish_trajectory(
		self,
		last_value: tf.Tensor,
		last_done: np.ndarray
	) -> None:
		self.cursor = 0
		self.value_buffer[-1] = last_value
		self.done_buffer[-1] = last_done

		self.done_buffer = 1.0 - self.done_buffer
		deltas = self.reward_buffer + self.gamma * self.value_buffer[1:] * self.done_buffer[1:] - self.value_buffer[:-1]
		last_gae_lam = 0
		for i in reversed(range(self.capacity)):
			last_gae_lam = deltas[i] + self.gamma * self.lam * self.done_buffer[i+1] * last_gae_lam
			self.advantage_buffer[i] = last_gae_lam
		self.return_buffer = self.advantage_buffer + self.value_buffer[:-1]