import pygame
from pygame import Vector2

from moon_lander import settings
from moon_lander.objects import PhysicalObject
from moon_lander.utils import random_vector2


class Lander(PhysicalObject):
    """ Game object representing lander """

    def __init__(self, pos: Vector2 = random_vector2(*settings.LANDER_STARTING_BOX)):
        super().__init__(pos, *settings.LANDER_SIZE)

        self.left_thruster = 0
        self.right_thruster = 0
        self.bottom_thruster = 0

    def update(self, delta: int):
        """ Updates game logic by handling physics and collisions detection """

        # calculate current force generated by thrusters
        self.force = Vector2(
            settings.SIDE_THRUSTERS_STRENGTH * (self.left_thruster - self.right_thruster),
            -settings.BOTTOM_THRUSTER_STRENGTH * self.bottom_thruster
        )

        # update physics
        super().update(delta)

        # detect collisions
        # TODO


class PlayerLander(Lander):
    """ Lander extended with support for player controls """

    def update(self, delta: int):
        keys = pygame.key.get_pressed()

        # set thrusters according to player input
        self.bottom_thruster = keys[settings.CONTROL_THRUSTER_BOTTOM]
        self.right_thruster = keys[settings.CONTROL_THRUSTER_RIGHT]
        self.left_thruster = keys[settings.CONTROL_THRUSTER_LEFT]

        # update physics
        super().update(delta)


class AILander(Lander):
    """ Lander extended with AI driven controls"""
    pass
