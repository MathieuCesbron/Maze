import numpy as np
import random

def world(n):
    '''Generate random maze for the model to train'''
    if not isinstance(n, int):
        raise TypeError('n should be a positive integer')
    elif n <= 2:
        raise ValueError('n should be superior or equal to 2')

    # Create an empty maze
    maze = np.zeros((n, n))

    # Create agent 1 & agent 2
    agent_1_pos = random.randint(0, n - 1)
    agent_2_pos = random.choice([i for i in range(n) if i != agent_1_pos])

    maze[0][agent_1_pos] = 1
    maze[0][agent_2_pos] = 2

    # Add traps and teleporter
    for idx1, row in enumerate(maze):
        for idx2, cell in enumerate(row):
            if cell == 0:
                stat = random.random()
                if stat > 0.8:
                    maze[idx1][idx2] = 3
                elif 0.7 < stat < 0.8:
                    maze[idx1][idx2] = 4

    # Add the exit to the maze
    maze[n - 1][random.randint(0, n - 1)] = 5
