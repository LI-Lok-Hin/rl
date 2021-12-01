'''
Train PPO agent for given environment
'''
from utils import import_path

import argparse
import functools
import multiprocessing as mp

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal, Zeros

from rl.agents import PPO
from rl.layers import CNNBlock
from rl.networks import Actor, Critic
from utils.make_env import make_env

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--clip-ratio", type=float, default=0.2,
		help="Value for clipping loss of PPO")
	parser.add_argument("--entropy-coef", type=float, default=0.0,
		help="Entropy coefficient of loss vale")
	parser.add_argument("--folder-name", type=str, default=None,
		help="Folder name of the agent")
	parser.add_argument("--frame-size", type=int, default=None,
		help="Number of frames for observation")
	parser.add_argument("--gamma", type=float, default=0.99,
		help="Discount factor of PPO")
	parser.add_argument("--gym-id", type=str, default="BipedalWalker-v3",
		help="ID of OpenAI gym environment")
	parser.add_argument("--initial-lr", type=float, default=3e-4,
		help="Initial learning rate of the agent")
	parser.add_argument("--is-lstm", type=bool, default=False,
		help="Indicate whether using LSTM for for actor-critic networks")
	parser.add_argument("--is-norm-obs", type=bool, default=True,
		help="Indicate whether perform observation normalization")
	parser.add_argument("--is-norm-rewards", type=bool, default=True,
		help="Indicate whether perform reward normalization")
	parser.add_argument("--lam", type=float, default=0.95,
		help="Lambda for general advantage estimation")
	parser.add_argument("--max-grad-norm", type=float, default=0.5,
		help="Maximum norm of gradient")
	parser.add_argument("--n-batch", type=int, default=32,
		help="Number of batches in each updating")
	parser.add_argument("--n-env", type=int, default=8,
		help="Number of parallel environments")
	parser.add_argument("--n-epoch", type=int, default=1000,
		help="Number of epoch of learning process")
	parser.add_argument("--n-epoch-step", type=int, default=256,
		help="Number of step in each epoch")
	parser.add_argument("--n-noops", type=int, default=30,
		help="Number of no-ops before each cycle")
	parser.add_argument("--n-update-epoch", type=int, default=10,
		help="Number of times of updating in each epoch")
	parser.add_argument("--save-freq", type=int, default=10,
		help="Frequenct of saving the agent")
	parser.add_argument("--target-kl", type=float, default=0.03,
		help="Target KL divergence threshold of PPO")
	parser.add_argument("--value-coef", type=float, default=0.5,
		help="Value network loss coefficient")
	args = parser.parse_args()
	
	# Build environment
	train_env_fn = functools.partial(
		make_env,
		gym_id=args.gym_id,
		frame_size=args.frame_size,
		is_lstm=args.is_lstm,
		n_noops=args.n_noops
	)
	eval_env_fn = None
	dummy_env = train_env_fn()
	input_shape = dummy_env.observation_space.shape
	output_space = dummy_env.action_space
	# Build actor netowrk
	actor = build_actor(
		input_shape=input_shape,
		output_space=output_space,
		is_lstm=args.is_lstm
	)
	# Build critic network
	critic = build_critic(
		input_shape=input_shape,
		is_lstm=args.is_lstm
	)
	# Build PPO agent
	agent = PPO(
		train_env_fn = train_env_fn,
		eval_env_fn = eval_env_fn,
		actor = actor,
		critic = critic,
		is_norm_obs = args.is_norm_obs,
		is_norm_rewards = args.is_norm_rewards,
		n_env = args.n_env,
		n_epoch = args.n_epoch,
		n_epoch_step = args.n_epoch_step,
		n_update_epoch = args.n_update_epoch,
		n_batch = args.n_batch,
		gamma = args.gamma,
		lam = args.lam,
		initial_lr = args.initial_lr,
		target_kl = args.target_kl,
		clip_ratio = args.clip_ratio,
		max_grad_norm = args.max_grad_norm,
		entropy_coef = args.entropy_coef,
		value_coef = args.value_coef,
		save_freq = args.save_freq,
		folder_name = args.folder_name
	)
	agent.train()

def build_actor(
	input_shape: tuple,
	output_space: gym.Space,
	is_lstm: bool,
) -> Actor:
	cnn_kernel_initializer = Orthogonal(np.sqrt(2))
	cnn_bias_initializer = Zeros()
	actor_kernel_initializer = Orthogonal(0.01)
	actor_bias_initializer = Zeros()
	cnn = CNNBlock(
		conv_shapes=[],
		fc_sizes=[64, 64],
		is_lstm=is_lstm,
		activation="tanh",
		kernel_initializer=cnn_kernel_initializer,
		bias_initializer=cnn_bias_initializer
	)
	layer = [cnn]
	actor = Actor(
		input_shape=input_shape,
		output_space=output_space,
		layers=layer,
		kernel_initializer=actor_kernel_initializer,
		bias_initializer=actor_bias_initializer
	)
	return actor

def build_critic(input_shape: tuple, is_lstm: bool) -> Critic:
	cnn_kernel_initializer = Orthogonal(np.sqrt(2))
	cnn_bias_initializer = Zeros()
	critic_kernel_initializer = Orthogonal(1)
	critic_bias_initializer = Zeros()
	cnn = CNNBlock(
		conv_shapes=[],
		fc_sizes=[64, 64],
		is_lstm=is_lstm,
		activation="tanh",
		kernel_initializer=cnn_kernel_initializer,
		bias_initializer=cnn_bias_initializer
	)
	layer = [cnn]
	critic = Critic(
		input_shape=input_shape,
		layers=layer,
		kernel_initializer=critic_kernel_initializer,
		bias_initializer=critic_bias_initializer
	)
	return critic

if __name__ == "__main__":
	mp.freeze_support()
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	main()