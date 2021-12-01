from rl.envs.make import make
from rl.envs import vector

from copy import deepcopy
from gym_anytrading import datasets
from gym.envs import register

register(
    id="stocks-v1",
    entry_point="envs.trading:StocksEnv",
    kwargs={
        "df": deepcopy(datasets.STOCKS_GOOGL),
        "window_size": 30,
        "period": ("2016-12-31", "2017-12-31")
    }
)