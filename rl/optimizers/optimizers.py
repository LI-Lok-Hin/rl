import inspect
import sys
from typing import Any, Dict
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import *

def picklable(cls):
	def __getstate__(self) -> Dict[str, Any]:
		return self.get_config()
	def __setstate__(self, state:Dict[str, Any]) -> None:
		optimizer = self.__class__.from_config(state)
		self.__dict__.update(optimizer.__dict__)
	cls.__getstate__ = __getstate__
	cls.__setstate__ = __setstate__
	return cls

'''
Decorate every class in tensorflow.keras.optimizers
with decorator `picklable` to make the optimizers become picklable.
All API should be same as original optimizer, but having two more
method __getstate__() & __setstate__() for pickle
'''
this_module = sys.modules[__name__]
clsmembers = inspect.getmembers(optimizers, inspect.isclass)
for (name, cls) in clsmembers:
	setattr(this_module, name, type(name, (picklable(cls),), {}))