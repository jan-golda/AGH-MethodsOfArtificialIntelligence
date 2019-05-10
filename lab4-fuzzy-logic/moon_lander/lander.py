import pygame
from pygame import Vector2

from moon_lander import settings
from moon_lander.objects import PhysicalObject
from moon_lander.utils import random_vector2


class Lander(PhysicalObject):

    def __init__(self, pos: Vector2 = random_vector2(*settings.LANDER_STARTING_BOX)):
        super().__init__(pos, *settings.LANDER_SIZE)

        self.left_thruster = 0
        self.right_thruster = 0
        self.bottom_thruster = 0

    def update(self, delta: int):
        self.force = Vector2(
            settings.SIDE_THRUSTERS_STRENGTH * (self.left_thruster - self.right_thruster),
            -settings.BOTTOM_THRUSTER_STRENGTH * self.bottom_thruster
        )

        super().update(delta)


class PlayerLander(Lander):

    def update(self, delta: int):
        keys = pygame.key.get_pressed()

        self.bottom_thruster = keys[pygame.K_UP]
        self.right_thruster = keys[pygame.K_LEFT]
        self.left_thruster = keys[pygame.K_RIGHT]

        super().update(delta)


class AILander(Lander):
    pass
