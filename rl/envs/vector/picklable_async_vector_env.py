from enum import Enum
import multiprocessing as mp
import sys

from gym.vector.async_vector_env import AsyncVectorEnv, AsyncState
from gym.vector.utils import write_to_shared_memory

class PicklableAsyncState(Enum):
	WAITING_GET_STATE = "get_state"
	WAITING_SET_STATE = "set_state"

class PicklableAsyncVectorEnv(AsyncVectorEnv):
	def __init__(
		self,
		env_fns,
		observation_space=None,
		action_space=None,
		copy=True,
		context=None,
		daemon=True
	):
		self.context = context
		self.daemon = daemon
		super(PicklableAsyncVectorEnv, self).__init__(
			env_fns,
			observation_space=observation_space,
			action_space=action_space,
			shared_memory=True,
			copy=copy,
			context=context,
			daemon=daemon,
			worker=_worker_savable_shared_memory
		)

	def get_state_async(self):
		self._assert_is_running()
		if self._state != AsyncState.DEFAULT:
			raise AlreadyPendingCallError(
				'Calling `__getstate__` while waiting for a pending call to `{0}` to complete.'.format(
					self._state.value),
				self._state.value
			)
		for pipe in self.parent_pipes:
			pipe.send(("get_state", None))
		self._state = PicklableAsyncState.WAITING_GET_STATE

	def get_state_wait(self, timeout=None):
		self._assert_is_running()
		if self._state != PicklableAsyncState.WAITING_GET_STATE:
			raise NoAsyncCallError(
				"Calling `get_state_wait` wothout any prior call to `get_state_async`.",
				PicklableAsyncState.WAITING_GET_STATE.value
			)
		if not self._poll(timeout):
			self._state = AsyncState.DEFAULT
			raise mp.TimeoutError(
				"The call to `get_state_wait` has timed out after {0} second{1}.".format(
				timeout,
				"s" if timeout > 1 else ""
			))
		results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
		self._raise_if_errors(successes)
		self._state = AsyncState.DEFAULT
		return results

	def __getstate__(self):
		settings = {
			"env_fns" : self.env_fns,
			"copy" : self.copy,
			"context" : self.context,
			"daemon" : self.daemon
		}
		self.get_state_async()
		states = self.get_state_wait()
		return {
			"settings" : settings,
			"states" : states
		}

	def set_state_async(self, states):
		self._assert_is_running()
		if self._state != AsyncState.DEFAULT:
			raise AlreadyPendingCallError(
				'Calling `__setstate__` while waiting for a pending call to `{0}` to complete.'.format(
					self._state.value),
				self._state.value
			)
		for pipe, state in zip(self.parent_pipes, states):
			pipe.send(("set_state", state))
		self._state = PicklableAsyncState.WAITING_SET_STATE

	def set_state_wait(self, timeout=None):
		self._assert_is_running()
		if self._state != PicklableAsyncState.WAITING_SET_STATE:
			raise NoAsyncCallError(
				"Calling `set_state_wait` wothout any prior call to `set_state_async`.",
				PicklableAsyncState.WAITING_SET_STATE.value
			)
		if not self._poll(timeout):
			self._state = AsyncState.DEFAULT
			raise mp.TimeoutError(
				"The call to `set_state_wait` has timed out after {0} second{1}.".format(
				timeout,
				"s" if timeout > 1 else ""
			))
		_, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
		self._raise_if_errors(successes)
		self._state = AsyncState.DEFAULT

	def __setstate__(self, state):
		settings = state["settings"]
		states = state["states"]
		self.__init__(**settings)
		self.set_state_async(states)
		self.set_state_wait()
		
def _worker_savable_shared_memory(
	index,
	env_fn,
	pipe,
	parent_pipe,
	shared_memory,
	error_queue
):
	assert shared_memory is not None
	env = env_fn()
	observation_space = env.observation_space
	parent_pipe.close()
	try:
		while True:
			command, data = pipe.recv()
			if command == "reset":
				observation = env.reset()
				write_to_shared_memory(
					index,
					observation,
					shared_memory,
					observation_space
				)
				pipe.send((None, True))
			elif command == "step":
				observation, reward, done, info = env.step(data)
				if done:
					observation = env.reset()
				write_to_shared_memory(
					index,
					observation,
					shared_memory,
					observation_space
				)
				pipe.send(((None, reward, done, info), True))
			elif command == "get_state":
				pipe.send((env, True))
			elif command == "set_state":
				env = data
				pipe.send((None, True))
			elif command == "seed":
				env.seed(data)
				pipe.send((None, True))
			elif command == "close":
				pipe.send((None, True))
				break
			elif command == "_check_observation_space":
				pipe.send((data == observation_space, True))
			else:
				raise RuntimeError(('Received unknown command `{0}`. Must '
					+ 'be one of {`reset`, `step`, `get_state`, `set_state`, `seed`, `close`, '
					+ '`_check_observation_space`}.').format(command))
	except (KeyboardInterrupt, Exception):
		error_queue.put((index,) + sys.exc_info()[:2])
		pipe.send((None, False))
	finally:
		env.close()