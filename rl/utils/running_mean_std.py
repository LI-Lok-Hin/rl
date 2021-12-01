from typing import Tuple
import numpy as np

class RunningMeanStd:
	def __init__(
		self,
		epsilon: float = 1e-4,
		shape = ()
	) -> None:
		self.mean = np.zeros(shape, "float64")
		self.var = np.ones(shape, "float64")
		self.count = epsilon

	def update(self, x: np.ndarray) -> None:
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)
	
	def update_from_moments(
		self,
		batch_mean: np.ndarray,
		batch_var: np.ndarray,
		batch_count: int
	) -> None:
		self.mean, self.var, self.count = update_mean_var_count_from_moments(
			self.mean,
			self.var,
			self.count,
			batch_mean,
			batch_var,
			batch_count
		)

def update_mean_var_count_from_moments(
	mean: np.ndarray,
	var: np.ndarray,
	count: int,
	batch_mean: np.ndarray,
	batch_var: np.ndarray,
	batch_count: float
) -> Tuple[np.ndarray, np.ndarray, float]:
	delta = batch_mean - mean
	tot_count = count + batch_count

	new_mean = mean + delta * batch_count / tot_count
	m_a = var * count
	m_b = batch_var * batch_count
	m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
	new_var = m2 / tot_count
	new_count = tot_count

	return new_mean, new_var, new_count