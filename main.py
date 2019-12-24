from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import numpy as np
from env import MazeEnv

world = np.array([[1, 0, 4, 2, 0],
                  [4, 0, 0, 4, 0],
                  [3, 3, 0, 0, 0],
                  [0, 3, 3, 0, 0],
                  [0, 0, 5, 0, 3]])

env = DummyVecEnv([lambda: MazeEnv(world)])

# Training
print('TRAINING')
model = PPO2(MlpPolicy, env, learning_rate=0.001, gamma=0.000001, lam=0)
model.learn(100000)

# Testing
print('TESTING')
result_test = []
obs = env.reset()
for i in range(300):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        result_test.append(info[0]['state'])

# Print result
result_stat = result_test.count('W') / len(result_test)
print(f'Success rate: {result_stat * 100} %')
