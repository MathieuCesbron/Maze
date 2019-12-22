from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import numpy as np
from env import MazeEnv

world = np.array([[1, 0, 0, 2, 0],
                  [0, 0, 0, 3, 0],
                  [0, 3, 0, 3, 0],
                  [0, 0, 4, 0, 0],
                  [0, 3, 0, 0, 3]])

env = DummyVecEnv([lambda: MazeEnv(world)])

# Training
print('TRAINING')
model = PPO2(MlpPolicy, env, learning_rate=0.01, gamma=0.1)
model.learn(200000)

# Testing
print('TESTING')
obs = env.reset()
for i in range(300):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
