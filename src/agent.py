from __future__ import annotations

from collections import deque

from game import SnakeGame

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
    # Will be used to plot progress
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    record = 0
    agent = Agent()

    game = SnakeGame()

    while (True):
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

            print(
                'Game', agent.number_of_games,
                'Score', score,
                'Record:', record,
            )


if __name__ == '__main__':
    train()
