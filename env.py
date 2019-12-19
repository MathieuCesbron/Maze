import gym
from gym import spaces
import numpy as np


class MazeEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0,
                                            high=4,
                                            shape=(5, 4),
                                            dtype=np.int16)
        self.reward_range = (-200, 200)
        self.current_player = 1
        # P means the game is playable, W means somenone wins, L someone lose
        self.state = 'P'
        self.world = np.array([[1, 0, 0, 2],
                              [0, 0, 0, 0],
                              [3, 3, 0, 3],
                              [0, 4, 0, 0]])

    def reset(self):
        self.current_player = 1
        self.state = 'P'
        self.current_step = 0
        self.max_step = 30
        self.world = np.array([[1, 0, 0, 2],
                              [0, 4, 0, 0],
                              [3, 3, 0, 3],
                              [0, 4, 0, 0]])

        return self._next_observation()

    def _next_observation(self):
        obs = self.world

        obs = np.append(obs, [[self.current_player, 0, 0, 0]], axis=0)

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        return obs

    def _take_action(self, action):
        current_pos = np.where(self.world == self.current_player)

        if action == 0:
            next_pos = (current_pos[0] - 1, current_pos[1])

            if next_pos[0] >= 0 and int(self.world[next_pos]) == 0:
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0

            elif next_pos[0] >= 0 and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[0] >= 0 and (int(self.world[next_pos]) == 3):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'L'

            elif next_pos[0] >= 0 and (int(self.world[next_pos]) == 4):
                self.state = 'W'

        elif action == 1:
            next_pos = (current_pos[0], current_pos[1] + 1)

            if next_pos[1] < 3 and int(self.world[next_pos]) == 0:
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0

            elif next_pos[1] < 3 and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[1] < 3 and (int(self.world[next_pos]) == 3):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'L'

            elif next_pos[1] < 3 and (int(self.world[next_pos]) == 4):
                self.state = 'W'

        elif action == 2:
            next_pos = (current_pos[0] + 1, current_pos[1])

            if next_pos[0] <= 3 and int(self.world[next_pos]) == 0:
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0

            elif next_pos[0] <= 3 and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[0] <= 3 and (int(self.world[next_pos]) == 3):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'L'

            elif next_pos[0] <= 3 and (int(self.world[next_pos]) == 4):
                self.state = 'W'

        elif action == 3:
            next_pos = (current_pos[0], current_pos[1] - 1)

            if next_pos[1] >= 0 and int(self.world[next_pos]) == 0:
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0

            elif next_pos[1] >= 0 and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 3):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'L'

            elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 4):
                self.state = 'W'

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        print(self.world)

        if self.state == "W":
            print(f'{self.current_player} wins !')
            reward = 100
            done = True
        elif self.state == 'L':
            print(f'{self.current_player} lost')
            reward = -100
            done = True
        elif self.state == 'P':
            print('ca continue !')
            reward = -1
            done = False

        if self.current_step > self.max_step:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}
