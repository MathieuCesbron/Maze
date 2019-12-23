import gym
from gym import spaces
import numpy as np


class MazeEnv(gym.Env):
    def __init__(self, world):
        self.world_start = world
        self.action_space = spaces.Discrete(4)

        shape_0 = np.size(self.world_start, 0)
        shape_1 = np.size(self.world_start, 1)
        self.observation_space = spaces.Box(low=0,
                                            high=4,
                                            shape=(shape_0 + 1, shape_1),
                                            dtype=np.int16)
        self.reward_range = (-200, 200)

        self.current_episode = 0
        self.success_episode = []

    def reset(self):
        self.current_player = 1
        # P means the game is playable, W means somenone wins, L someone lose
        self.state = 'P'
        self.current_step = 0
        self.max_step = 30
        self.world = np.copy(self.world_start)

        self.exploration_prize = np.ones(shape=(np.size(self.world, 0),
                                                np.size(self.world, 1)))
        self.bonus_reward = 0

        return self._next_observation()

    def _next_observation(self):
        obs = self.world

        data_to_add = [0] * np.size(self.world, 1)
        data_to_add[0] = self.current_player

        obs = np.append(obs, [data_to_add], axis=0)

        return obs

    def _take_action(self, action):
        current_pos = np.where(self.world == self.current_player)

        if action == 0:
            next_pos = (current_pos[0] - 1, current_pos[1])

            if next_pos[0] >= 0 and int(self.world[next_pos]) == 0:
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self._exploration_prize(next_pos)

            elif next_pos[0] >= 0 and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[0] >= 0 and int(self.world[next_pos] == 3):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'L'
                self._exploration_prize(next_pos)

            elif next_pos[0] >= 0 and int(self.world[next_pos] == 4):
                pass

            elif next_pos[0] >= 0 and (int(self.world[next_pos]) == 5):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'W'
                self._exploration_prize(next_pos)

        elif action == 1:
            next_pos = (current_pos[0], current_pos[1] + 1)
            limit = np.size(self.world, 1)

            if next_pos[1] < limit and int(self.world[next_pos]) == 0:
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self._exploration_prize(next_pos)

            elif next_pos[1] < limit and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[1] < limit and (int(self.world[next_pos]) == 3):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'L'
                self._exploration_prize(next_pos)

            elif next_pos[1] < limit and (int(self.world[next_pos]) == 5):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'W'
                self._exploration_prize(next_pos)

        elif action == 2:
            next_pos = (current_pos[0] + 1, current_pos[1])
            limit = np.size(self.world, 0)

            if next_pos[0] < limit and int(self.world[next_pos]) == 0:
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self._exploration_prize(next_pos)

            elif next_pos[0] < limit and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[0] < limit and (int(self.world[next_pos]) == 3):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'L'
                self._exploration_prize(next_pos)

            elif next_pos[0] < limit and (int(self.world[next_pos]) == 5):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'W'
                self._exploration_prize(next_pos)

        elif action == 3:
            next_pos = (current_pos[0], current_pos[1] - 1)

            if next_pos[1] >= 0 and int(self.world[next_pos]) == 0:
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self._exploration_prize(next_pos)

            elif next_pos[1] >= 0 and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 3):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'L'
                self._exploration_prize(next_pos)

            elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 5):
                self.world[next_pos] = self.current_player
                self.world[current_pos] = 0
                self.state = 'W'
                self._exploration_prize(next_pos)

    def _exploration_prize(self, next_pos):
        if self.exploration_prize[next_pos] == 1:
            self.exploration_prize[next_pos] = 0
            self.bonus_reward += 1

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        print(self.world)

        if self.state == "W":
            print(f'Player {self.current_player} won')
            reward = 200
            done = True
        elif self.state == 'L':
            print(f'Player {self.current_player} lost')
            reward = -200
            done = True
        elif self.state == 'P':
            reward = -2
            done = False

        if self.current_step >= self.max_step:
            print(f'New episode number {self.current_episode + 1}')
            done = True

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        # Apply the bonus reward for this step then reset him to 0
        reward += self.bonus_reward
        self.bonus_reward = 0

        if done:
            self.render_episode(self.state)
            self.current_episode += 1

        obs = self._next_observation()

        return obs, reward, done, {'state': self.state}

    def render_episode(self, win_or_lose):
        self.success_episode.append(
            'Success' if win_or_lose == 'W' else 'Failure')

        file = open('render/render.txt', 'a')
        file.write('----------------------------\n')
        file.write(f'Episode number {self.current_episode}\n')
        file.write(
            f'{self.success_episode[-1]} in {self.current_step} steps\n')
        file.close()
