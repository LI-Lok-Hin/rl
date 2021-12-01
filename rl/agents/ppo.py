import functools
import time
from typing import Callable, Tuple

from gym import spaces
from gym.spaces import Discrete
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical, Distribution, Normal

from rl.agents import Agent
from rl.envs.vector import PicklableAsyncVectorEnv
from rl.networks import Network
from rl.optimizers import Adam
from rl.replay_buffers import PPOReplayBuffer
from rl.schedules import PolynomialDecay

class PPO(Agent):
	def __init__(
		self,
		train_env_fn: Callable,
		eval_env_fn: Callable,
		actor: Network,
		critic: Network,
		is_norm_obs: bool = True,
		is_norm_rewards: bool = True,
		n_env: int = 1,
		n_epoch: int = 10000,
		n_epoch_step: int = 128,
		n_update_epoch: int = 4,
		n_batch: int = 4,
		gamma: float = 0.99,
		lam: float = 0.95,
		initial_lr: float = 1e-4,
		target_kl: float = 0.03,
		clip_ratio: float = 0.1,
		max_grad_norm: float = 0.5,
		entropy_coef: float = 0.01,
		value_coef: float = 0.5,
		save_freq: int = 100,
		folder_name: str = None,
	) -> None:
		seed = 0
		train_env = PicklableAsyncVectorEnv([functools.partial(train_env_fn, seed=seed+i) for i in range(n_env)])
		train_env.seed(seed)

		if eval_env_fn is not None:
			eval_env = eval_env_fn(seed=seed)
		else:
			eval_env = train_env_fn(seed=seed)
		self.replay_buffer = PPOReplayBuffer(
			capacity=n_epoch_step,
			n_env=n_env,
			state_space=train_env.observation_space,
			action_space=train_env.action_space,
			gamma=gamma,
			lam=lam
		)
		self.actor = actor
		self.critic = critic
		self.epoch_counter = tf.Variable(0)
		self.lr = PolynomialDecay(
			initial_learning_rate=initial_lr,
			decay_steps=n_epoch,
			end_learning_rate=0.0,
			step_counter=self.epoch_counter
		)
		self.optimizer = Adam(
			learning_rate=self.lr,
			epsilon=1e-5
		)

		if isinstance(train_env.action_space, spaces.Tuple):
			action_space = train_env.action_space[0]
		else:
			action_space = train_env.action_space
		self.is_discrete = isinstance(action_space, Discrete)
		if self.is_discrete:
			self.distribution = Categorical
		else:
			action_shape = eval_env.action_space.shape
			self.distribution = Normal
			self.log_stds = tf.Variable(tf.zeros((1, np.prod(action_shape))))
		
		self.n_env = n_env
		self.n_epoch = n_epoch
		self.n_epoch_step = n_epoch_step
		self.n_update_epoch = n_update_epoch
		self.n_batch = n_batch
		self.target_kl = target_kl
		self.clip_ratio = clip_ratio
		self.max_grad_norm = max_grad_norm
		self.entropy_coef = entropy_coef
		self.value_coef = value_coef
		self.save_freq = save_freq

		self.done = np.zeros(n_env)
		self.episode_rewards = []
		self.n_step = 0
		self.n_episode = 0

		super(PPO, self).__init__(
			train_env=train_env,
			eval_env=eval_env,
			gamma=gamma,
			is_norm_obs=is_norm_obs,
			is_norm_rewards=is_norm_rewards,
			folder_name=folder_name
		)

	def train(self) -> None:
		while self.epoch_counter.numpy() < self.n_epoch:
			start_time = time.time()
			for _ in range(self.n_epoch_step):
				action, log_prob = self._get_train_action(self.state)
				next_state, reward, next_done, infos = self.step(action)
				value = self.critic(self.state)
				self.replay_buffer.add(
					state=self.state,
					action=action,
					reward=reward,
					done=self.done,
					value=value,
					log_prob=log_prob
				)
				self.state = next_state
				self.done = next_done
				self.n_step += self.n_env
				for i, info in enumerate(infos):
					if "episode" in info:
						self.episode_rewards.append(info["episode"]["r"])
						if len(self.episode_rewards) >= 100:
							del self.episode_rewards[:1]
						self.n_episode += 1

			last_value = self.critic(self.state)
			self.replay_buffer.finish_trajectory(last_value, next_done)
			for _ in range(self.n_update_epoch):
				(
					batch_states,
					batch_actions,
					batch_advs,
					batch_returns,
					batch_log_probs,
					batch_values
				) = self.replay_buffer.get(self.n_batch)
				for i in range(self.n_batch):
					kl = self.__update(
						batch_states[i],
						batch_actions[i],
						batch_advs[i],
						batch_returns[i],
						batch_log_probs[i],
						batch_values[i]
					)
				if self.target_kl and kl > self.target_kl:
					break

			runtime = time.time() - start_time
			template = "[Epoch {}] running reward: {:.2f} at episode {}, frame count {}, runtime: {:.2f}"
			print(template.format(
				self.epoch_counter.numpy() + 1,
				np.mean(self.episode_rewards),
				self.n_episode,
				self.n_step,
				runtime
			))
			self.epoch_counter.assign_add(1)

			self._append_metrics(
				"Time Steps", self.n_step
			)
			self._append_metrics(
				"Average Episode Reward", np.mean(self.episode_rewards)
			)
			if self.epoch_counter.numpy() % self.save_freq == 0:
				self.save()

	def _get_distribution(self, states: np.ndarray) -> Distribution:
		outputs = self.actor(states)
		if self.is_discrete:
			return self.distribution(logits=outputs)
		else:
			log_stds = tf.broadcast_to(self.log_stds, outputs.shape)
			stds = tf.exp(log_stds)
			return self.distribution(loc=outputs, scale=stds)

	@tf.function
	def _get_train_action(self, states: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
		distribution = self._get_distribution(states)
		actions = distribution.sample()
		log_probs = get_log_probs(distribution, actions)
		return actions, log_probs

	@tf.function
	def _get_eval_action(self, state: np.ndarray) -> tf.Tensor:
		state = state[np.newaxis,:]
		output = tf.squeeze(self.actor(state))
		if self.is_discrete:
			return tf.argmax(output)
		else:
			return output
		
	@tf.function
	def __update(
		self,
		states: np.ndarray,
		actions: np.ndarray,
		advs: np.ndarray,
		returns: np.ndarray,
		log_probs: np.ndarray,
		values: np.ndarray
	) -> tf.Tensor:
		with tf.GradientTape() as tape:
			distribution = self._get_distribution(states)
			new_log_probs = get_log_probs(distribution, actions)
			log_ratio = new_log_probs - log_probs
			ratio = tf.exp(log_ratio)
			policy_loss1 = -advs * ratio
			policy_loss2 = -advs * tf.clip_by_value(
				ratio,
				1 - self.clip_ratio,
				1 + self.clip_ratio
			)
			policy_loss = tf.reduce_mean(
				tf.maximum(policy_loss1, policy_loss2)
			)

			new_values = self.critic(states)
			value_loss_unclip = (new_values - returns) ** 2
			value_clip = values + tf.clip_by_value(
				new_values - values,
				-self.clip_ratio,
				self.clip_ratio
			)
			value_loss_clip = (value_clip - returns) ** 2
			value_loss_max = tf.maximum(value_loss_unclip, value_loss_clip)
			value_loss = 0.5 * tf.reduce_mean(value_loss_max)

			entropy_loss = tf.reduce_mean(distribution.entropy())
			
			loss = policy_loss - self.entropy_coef * entropy_loss + value_loss * self.value_coef

		kl = tf.reduce_mean(log_probs - new_log_probs)
		trainable_variables = tape.watched_variables()
		grads = tape.gradient(loss, trainable_variables)
		grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
		self.optimizer.apply_gradients(zip(grads, trainable_variables))
		return kl

	@property
	def name(self) -> str:
		return "PPO"

def get_log_probs(
	distribution: Distribution,
	values: tf.Tensor
) -> tf.Tensor:
	log_probs = distribution.log_prob(values)
	if len(log_probs.shape) > 1:
		log_probs = tf.reduce_sum(log_probs, axis=-1)
	return log_probs