from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import numpy as np
from env import MazeEnv
from world import new_world


# Training
for i in range(100):
    world = new_world(5)
    env = DummyVecEnv([lambda: MazeEnv(world)])
    model = PPO2(MlpPolicy, env, learning_rate=0.001, gamma=0.000001, lam=0)
    model.learn(300)

# Testing
world = new_world(5)
env = DummyVecEnv([lambda: MazeEnv(world)])
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
