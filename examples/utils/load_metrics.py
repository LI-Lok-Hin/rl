import os
import pickle

def load_metrics(folder_name: str):
	metrics_dict = {}
	path = os.path.join("trained", folder_name, "metrics")
	for metrics_file in os.listdir(path):
		l = []
		metrics_path = os.path.join(path, metrics_file)
		with open(metrics_path, "rb") as f:
			metrics_name = os.path.splitext(metrics_file)[0]
			while True:
				try:
					l.append(pickle.load(f))
				except:
					break
		metrics_dict[metrics_name] = l
	return metrics_dict