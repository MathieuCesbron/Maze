from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import numpy as np
from env import MazeEnv
from world import new_world


def model(nb_ep_train):

    to_test = [2, 10, 50, 80, 100, 200, 300, 500, 1000, 2000, 5000, 10000]

    # Training
    for i in range(nb_ep_train):
        world = new_world(3)
        env = DummyVecEnv([lambda: MazeEnv(world)])
        model = PPO2(MlpPolicy, env, learning_rate=0.001, gamma=0.00001, lam=0.01)
        model.learn(300)

        # Testing
        if i in to_test:
            world = new_world(3)
            env = DummyVecEnv([lambda: MazeEnv(world)])
            result_test = []
            obs = env.reset()
            for j in range(300):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    result_test.append(info[0]['state'])

            result_stat = result_test.count('W') / len(result_test)
            file = open('render/ep.txt', 'a')
            file.write(f'Result of {300 * i} steps : {round(result_stat * 100)} % of Success\n')
            file.close()


if __name__ == '__main__':
    model(10050)
