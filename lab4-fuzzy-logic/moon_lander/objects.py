""" Module defining different types of game objects """
from typing import List

import pygame
from pygame import Vector2, Surface, Color
from pygame.event import EventType
from pygame.rect import Rect

from moon_lander import settings


class GameObject:
    """ Base class of game object """

    def __init__(self, pos: Vector2, width: float, height: float):
        """ Creates rectangle sized game object at given position """
        self.position = pos
        self.width = width
        self.height = height

    def handle_events(self, events: List[EventType]):
        """ Handles events """
        pass

    def draw(self, surface: Surface):
        """ Draws object to the screen, by default as red rectangle """
        pygame.draw.rect(surface, Color('red'), self.bounds, 1)

    def update(self, delta: int):
        """ Updates game logic """
        pass

    @property
    def bounds(self):
        """ Returns rect representing this object, rect is shifted so that position of game object is in center """
        return Rect(self.position.x - self.width/2, self.position.y - self.height/2, self.width, self.height)


class PhysicalObject(GameObject):
    """ Game object extended with physics calculations """

    def __init__(self, pos: Vector2, width: float, height: float, mass: float = 1.0):
        super().__init__(pos, width, height)

        self.mass = mass
        self.gravitation = Vector2(settings.GRAVITATION)
        self.speed = Vector2()
        self.force = Vector2()

    def update(self, delta: int):
        """ Updates game logic by calculating current speed and position of object using forces applied to it """
        super().update(delta)

        acc = self.force / self.mass
        acc += self.gravitation
        self.speed += acc * (delta / 1000.0 * settings.PHYSICS_SPEED)
        self.position += self.speed * (delta / 1000.0 * settings.PHYSICS_SPEED)
