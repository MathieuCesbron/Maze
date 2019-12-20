from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env import MazeEnv


env = DummyVecEnv([lambda: MazeEnv()])
model = PPO2(MlpPolicy, env, learning_rate=0.001)
model.learn(500000)
