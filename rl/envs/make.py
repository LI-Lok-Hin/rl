import gym
from rl.envs.wrappers import TimeLimit

def make(id:str, **kwargs):
    env = gym.make(id, **kwargs)
    env = TimeLimit(env)
    return env