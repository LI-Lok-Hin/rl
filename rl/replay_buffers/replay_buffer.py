import numpy as np

class ReplayBuffer():
	def __init__(
		self,
		capacity: int = 1000,
	) -> None:
		self.buffer = np.empty(capacity).astype(object)
		self.size = 0
		self.cursor = 0

	def append(self, state, action, reward, next_state, done) -> None:
		self.buffer[self.cursor] = [state, action, reward, next_state, done]
		self.size = min(self.size + 1, self.capacity)
		self.cursor = (self.cursor + 1) % self.capacity

	def sample(self, batch_size:int=32):
		samples = np.random.choice(self.buffer[:self.size], size=batch_size, replace=False)
		states, actions, rewards, next_states, dones = map(np.asarray, zip(*samples))
		return states, actions, rewards, next_states, dones

	@property
	def capacity(self) -> int:
		return len(self.buffer)

	def __len__(self) -> int:
		return self.size