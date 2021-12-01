import time
from typing import Callable

import numpy as np
import tensorflow as tf

from rl.agents import Agent
from rl.networks import Network
from rl.replay_buffers import ReplayBuffer
from rl.schedules import PolynomialDecay

class DDQN(Agent):

	def __init__(
		self,
		env_fn: Callable,
		network: Network,
		is_norm_obs: bool = True,
		is_norm_rewards: bool = True,
		gamma: float = 0.99,
		initial_epsilon: float = 0.1,
		final_epsilon: float = 1.0,
		batch_size: int = 32,
		n_collection_step: int = 50000,
		n_epsilon_greedy_step: int = 1000000,
		memory_capacity: int = 10000,
		main_update_freq: int = 1,
		target_update_freq: int = 10000,
		log_freq: int = 1000,
		save_freq: int = 100000,
		folder_name: str = None
	):
		train_env = env_fn()
		eval_env = env_fn()
		
		self.main_model = network.clone_config()
		self.target_model = network.clone_config()

		self.replay_buffer = ReplayBuffer(
			capacity=memory_capacity
		)

		self.step_counter = tf.Variable(0)
		self.__epsilon_scheduler = PolynomialDecay(
			initial_learning_rate=initial_epsilon,
			decay_steps=n_epsilon_greedy_step,
			end_learning_rate=final_epsilon,
			step_counter=self.step_counter
		)
		
		self.gamma = gamma
		self.batch_size = batch_size
		self.n_collection_step = n_collection_step
		self.main_update_freq = main_update_freq
		self.target_update_freq = target_update_freq
		self.log_freq = log_freq
		self.save_freq = save_freq

		self.n_action = train_env.action_space.n
		self.n_step = 0
		self.n_episode = 0
		self.episode_rewards = []
		self.state = np.array(train_env.reset())
		super(DDQN, self).__init__(
			train_env=train_env,
			eval_env=eval_env,
			gamma=gamma,
			is_norm_obs=is_norm_obs,
			is_norm_rewards=is_norm_rewards,
			folder_name=folder_name
		)

	def __update_target(self):
		self.target_model.clone_weight(self.main_model)

	@property
	def epsilon(self):
		self.step_counter.assign_add(-self.n_collection_step)
		epsilon = self.__epsilon_scheduler(None)
		self.step_counter.assign_add(self.n_collection_step)
		return epsilon

	def train(self):
		start_time = time.time()
		while True:
			self.n_step += 1
			self.step_counter.assign_add(1)
			action = self._get_train_action(self.state)
			next_state, reward, done, info = self.train_env.step(action)
			if done:
				next_state = self.train_env.reset()
			next_state = np.array(next_state)
			self.replay_buffer.append(self.state, action, reward, next_state, done)
			self.state = next_state

			if "episode" in info:
				self.episode_rewards.append(info["episode"]["r"])
				if len(self.episode_rewards) >= 100:
					del self.episode_rewards[:1]
				self.n_episode += 1

			if len(self.replay_buffer) > self.batch_size and self.n_step % self.main_update_freq == 0:
				self.__update__main()

			if self.n_step % self.target_update_freq == 0:
				self.__update_target()

			if self.n_step % self.log_freq == 0:
				runtime = time.time() - start_time
				start_time = time.time()
				template = "running reward: {:.2f} at episode {}, frame count {}, runtime: {}"
				print(template.format(
					np.mean(self.episode_rewards),
					self.n_episode,
					self.n_step,
					runtime
				))
				self._append_metrics(
					"Time Steps", self.n_step
				)
				self._append_metrics(
					"Average Episode Reward", np.mean(self.episode_rewards)
				)

			if self.n_step % self.save_freq == 0:
				self.save()

	def _get_train_action(self, state: np.ndarray):
		if self.n_step < self.n_collection_step or self.epsilon > np.random.rand():
			action = np.random.choice(self.n_action)
		else:
			action_probs = self.main_model(state[np.newaxis,:], training=False)
			action = np.argmax(action_probs[0])
		return action

	def _get_eval_action(self, state: np.ndarray) -> tf.Tensor:
		return tf.argmax(self.main_model(state[np.newaxis,:])[0])

	def __update__main(self):
		states, actions, rewards, next_states, dones = self.replay_buffer.sample(
			batch_size=self.batch_size
		)
		future_rewards = self.target_model(next_states, training=False)
		updated_q_values = rewards + self.gamma * tf.reduce_max(
			future_rewards, axis=1
		)
		updated_q_values = updated_q_values * (1 - dones) - dones
		masks = tf.one_hot(actions, self.n_action)
		self.main_model.train(
			x=states,
			target_y=updated_q_values,
			masks=masks
		)

	@property
	def name(self):
		return "DDQN"