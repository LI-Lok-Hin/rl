from utils import import_path
import argparse
from utils import load_metrics, plot_line_graph

parser = argparse.ArgumentParser()
parser.add_argument("--folder-names", nargs="+", type=str,
    help="Folder names of the agents")
args = parser.parse_args()

steps = []
rewards = []
labels = []
for folder_name in args.folder_names:
	metrics_dict = load_metrics(folder_name)
	steps.append(metrics_dict["Time Steps"])
	rewards.append(metrics_dict["Average Episode Reward"])
	labels.append(folder_name)
plot_line_graph(
	xs=steps,
	ys=rewards,
	labels=labels,
	xlabel="Time Steps",
	ylabel="Average Episode Reward"
)
