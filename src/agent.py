from __future__ import annotations

from collections import deque
# from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000  # Maximum number of experiences we are storing
BATCH_SIZE = 1000  # Number of experiences we use for training per batch
LR = 0.001  # Learning rate


class Agent:

    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0  # Discount rate
        self.memore = deque(maxlen=MAX_MEMORY)
        # TODO: Initialize model
        pass

    def get_state(self, game):
        pass

    def remember(Self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state):
        pass


def train():
    pass


if __name__ == '__main__':
    train()
