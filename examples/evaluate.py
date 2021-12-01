from utils import import_path

import argparse
import multiprocessing as mp

import tensorflow as tf

from rl.agents import Agent
from utils import load_metrics, plot_line_graph

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder-name", type=str,
		help="Folder name of the agent")
	parser.add_argument("--evaluate-filename", type=str,
		help="Name of the evaluation result file")
	parser.add_argument("--fps", type=float, default=60,
		help="FPS of the evaluation video")
	parser.add_argument("--max_video_length", type=float, default=180,
		help="Maximum length of the evaluation video")
	args = parser.parse_args()

	metrics_dict = load_metrics(args.folder_name)
	plot_line_graph(
		xs=metrics_dict["Time Steps"],
		ys=metrics_dict["Average Episode Reward"],
		xlabel="Time Steps",
		ylabel="Average Episode Reward"
	)
	agent = Agent.load(args.folder_name)
	agent.eval(
		fps=args.fps,
		max_video_length=args.max_video_length,
		filename=args.evaluate_filename
	)

if __name__ == "__main__":
	mp.freeze_support()
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	main()