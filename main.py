from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import numpy as np
from env import MazeEnv

world = np.array([[1, 0, 0, 2, 0],
                  [0, 3, 0, 3, 0],
                  [0, 3, 0, 3, 3],
                  [0, 4, 0, 0, 0],
                  [0, 3, 0, 0, 3]])

env = DummyVecEnv([lambda: MazeEnv(world)])
model = PPO2(MlpPolicy, env, learning_rate=0.001)
model.learn(500000)
