from utils import import_path

import argparse
import multiprocessing as mp

import tensorflow as tf

from rl.agents import Agent

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--folder-name",
		type=str,
		help="Folder name of the agent"
	)
	args = parser.parse_args()

	agent = Agent.load(args.folder_name)
	agent.train()

if __name__ == "__main__":
	mp.freeze_support()
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	main()