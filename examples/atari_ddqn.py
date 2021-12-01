'''
Train DDQN agent for atari environment
'''
from utils import import_path

import argparse
import functools

from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from rl.agents import DDQN
from rl.layers import DivisionLayer
from rl.networks import CNN
from utils.make_env import make_atari_env

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32,
	help="Batch size for training step")
parser.add_argument("--clipnorm", type=float, default=1.0,
	help="Maximum norm of gradient")
parser.add_argument("--final_epsilon", type=float, default=0.1,
	help="Final exploration rate")
parser.add_argument("--folder-name", type=str, default=None,
	help="Folder name of the agent")
parser.add_argument("--gamma", type=float, default=0.99,
	help="Discount factor of DDQN")
parser.add_argument("--gym-id", type=str, default="BreakoutNoFrameskip-v4",
	help="ID of OpenAI gym environment")
parser.add_argument("--initial_epsilon", type=float, default=1.0,
	help="Initial exploration rate")
parser.add_argument("--is-deuling", type=bool, default=True,
	help="Whether applying dealing layer to the network or not")
parser.add_argument("--learning-rate", type=float, default=2.5e-4,
	help="Learning rate of the optimizer")
parser.add_argument("--log-freq", type=int, default=1000,
	help="Frequency of logging")
parser.add_argument("--main-update-freq", type=int, default=4,
	help="Frequency for updating main network")
parser.add_argument("--memory-capacity", type=int, default=100000,
	help="Capacity of replay buffer")
parser.add_argument("--n-collection-step", type=int, default=50000,
	help="Number of steps before learning for collecting experience")
parser.add_argument("--n-epsilon-greedy-step", type=int, default=1000000,
	help="Number of steps to decay exploration rate from initial rate to final rate")
parser.add_argument("--save-freq", type=int, default=100000,
	help="Frequenct of saving the agent")
parser.add_argument("--target-update-freq", type=int, default=10000,
	help="Frequency for updating target network")
args = parser.parse_args()

# Build environment
env_fn = functools.partial(
	make_atari_env,
	gym_id=args.gym_id,
	frame_size=4
)
# Build network
dummy_env = env_fn()
input_shape = dummy_env.observation_space.shape
output_shape = (dummy_env.action_space.n,)
network = CNN(
	input_shape=input_shape,
	output_shape=output_shape,
	preprocessing_layers=[DivisionLayer(255)],
	conv_shapes=[(32, 8, 4),
				 (64, 4, 2),
				 (64, 3, 1)],
	fc_sizes=[512],
	is_deuling=args.is_deuling,
	optimizer=Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm),
	loss=Huber()
)
# Build DDQN agent
agent = DDQN(
	env_fn=env_fn,
	network=network,
	is_norm_obs=False,
	is_norm_rewards=False,
	gamma=args.gamma,
	initial_epsilon=args.initial_epsilon,
	final_epsilon=args.final_epsilon,
	batch_size=args.batch_size,
	n_collection_step=args.n_collection_step,
	n_epsilon_greedy_step=args.n_epsilon_greedy_step,
	memory_capacity=args.memory_capacity,
	main_update_freq=args.main_update_freq,
	target_update_freq=args.target_update_freq,
	log_freq=args.log_freq,
	save_freq=args.save_freq,
	folder_name=args.folder_name
)
agent.train()