from __future__ import annotations

import random

import pygame
from eunm import Enum

# Initialize pygame
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Initialise constants


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Color(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    GREEN = (0, 200, 0)
    BLUE = (0, 0, 200)
    YELLOW = (200, 200, 0)
    ORANGE = (200, 100, 0)
    PURPLE = (200, 0, 200)
    CYAN = (0, 200, 200)
    GREY = (100, 100, 100)
    PINK = (200, 0, 100)


BLOCK_SIZE = 20
SPEED = 40

# Create game class


class SnakeGame:

    def __init__(self, w=640, h=480):
        """
        Initialize the SnakeGame class.

        Parameters:
        w (int): The width of the game window. Defaults to 640.
        h (int): The height of the game window. Defaults to 480.
        """
        self.w = w
        self.h = h

        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        Reset the game to its initial state.
        """
        self.direction = Direction.RIGHT

        self.head = [self.w/2, self.h/2]
        self.snake = [
            self.head, [self.head[0]-BLOCK_SIZE, self.head[1]],
            [self.head[0]-2*BLOCK_SIZE, self.head[1]],
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """
        Place food randomly on the game board.
        """
        x = BLOCK_SIZE * \
            round(random.randrange(0, self.w-BLOCK_SIZE)/BLOCK_SIZE)
        y = BLOCK_SIZE * \
            round(random.randrange(0, self.h-BLOCK_SIZE)/BLOCK_SIZE)
        self.food = [x, y]

        # Make sure food is not placed on the snake
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Play one step of the game.

        Parameters:
        action (int): The action to take. 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT

        Returns:
        reward (int): The reward for the current step.
        game_over (bool): Whether the game is over or not.
        score (int): The current score.
        """
        self.frame_iteration += 1

        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False

        # collisions and endless loops end the game
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Check if the snake has collided with the boundary or itself.

        Parameters:
        pt (list): The point to check. Defaults to None.

        Returns:
        bool: Whether the snake has collided or not.
        """
        pt = pt or self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or \
            pt.x < 0 or \
                pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False
