from pygame import Vector2

from moon_lander import settings
from moon_lander.objects import GameObject
from moon_lander.utils import random_vector2


class LandingZone(GameObject):
    def __init__(self, pos: Vector2 = random_vector2(*settings.LANDING_STARTING_BOX)):
        super().__init__(pos, *settings.LANDING_SIZE)
