from pygame import Vector2

from moon_lander import settings
from moon_lander.objects import GameObject
from moon_lander.utils import random_vector2


class LandingZone(GameObject):
    """ Game object representing landing zone for lander """

    def __init__(self, pos: Vector2 = None):
        if not pos:
            pos = random_vector2(*settings.LANDING_STARTING_BOX)

        super().__init__(pos, *settings.LANDING_SIZE)
