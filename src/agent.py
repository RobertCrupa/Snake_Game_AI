from __future__ import annotations

from collections import deque

import numpy as np
from game import BLOCK_SIZE
from game import Direction
from game import Point
from game import SnakeGame

MAX_MEMORY = 100_000  # Maximum number of experiences we are storing
BATCH_SIZE = 1000  # Number of experiences we use for training per batch
LR = 0.001  # Learning rate


class Agent:

    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None  # TODO: Initialize model
        self.trainer = None  # TODO: Initialize trainer
        pass

    def get_state(self, game):
        """
        Gets the state of the game.

        Parameters:
        game (SnakeGame): The game instance.

        Returns:
        state (np.array): The state of the game.
        """

        head = game.snake[0]

        # Points around the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Create agent state
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y,  # Food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done),
        )

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        pass


def train():
    # Will be used to plot progress
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    record = 0
    agent = Agent()

    game = SnakeGame()

    while True:

        # Get old state
        state_old = agent.get_state(game)

        # Get move
        next_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(next_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(
            state_old, next_move, reward, state_new, done,
        )

        # Remember
        agent.remember(state_old, next_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            record = max(score, record)
            # TODO: save record model

            print(
                'Game', agent.number_of_games,
                'Score', score,
                'Record:', record,
            )

            # TODO: plot progress


if __name__ == '__main__':
    train()
