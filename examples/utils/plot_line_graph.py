from typing import List
from matplotlib import pyplot as plt

def plot_line_graph(
	xs: List = None,
	ys: List = None,
	labels: List[str] = None,
	xlabel: str = None,
	ylabel: str = None
):
	"""
	Plot a line graph with params

	ys: Data, or list of data to plot
	xs: x-value of line
	labels: List of label of respected data
	xlabel: Lable of x-axis
	ylabel: Lable of y-axis
	"""
	if not isinstance(ys[0], list):
		xs = [xs]
		ys = [ys]
		labels = [labels]
	fig, ax = plt.subplots()
	for i in range(len(ys)):
		x = xs[i]
		y = ys[i]
		label = labels[i]
		ax.plot(x, y, label=label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if label:
		ax.legend()
	plt.show()