import random

from pygame import Vector2


def random_vector2(min_x: float, min_y: float, max_x: float, max_y: float):
    return Vector2(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
