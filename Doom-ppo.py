import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

#from common.multiprocessing_env import SubprocVecEnv

num_envs = 16
env_name = "BipedalWalker-v2"

def make_env():
    def _thunk():
        env = gym.make(env_name)
        #env = FetchObservationWrapper(env)
        #print("created new env")
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)
#env = FetchObservationWrapper(env)
print("done")