from __future__ import annotations

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
        self.w = w
        self.h = h

        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Reinitialize game
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
