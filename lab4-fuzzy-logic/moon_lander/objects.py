from typing import List

import pygame
from pygame import Vector2, Surface, Color
from pygame.event import EventType
from pygame.rect import Rect

from moon_lander import settings


class GameObject:

    def __init__(self, pos: Vector2, width: float, height: float):
        self.position = pos
        self.width = width
        self.height = height

    def handle_events(self, events: List[EventType]):
        pass

    def draw(self, surface: Surface):
        pygame.draw.rect(surface, Color('red'), self.bounds, 1)

    def update(self, delta: int):
        pass

    @property
    def bounds(self):
        return Rect(self.position.x - self.width/2, self.position.y - self.height/2, self.width, self.height)


class PhysicalObject(GameObject):

    def __init__(self, pos: Vector2, width: float, height: float, mass: float = 1.0):
        super().__init__(pos, width, height)

        self.mass = mass
        self.gravitation = Vector2(settings.GRAVITATION)
        self.speed = Vector2()
        self.force = Vector2()

    def update(self, delta: int):
        super().update(delta)

        acc = self.force / self.mass
        acc += self.gravitation
        self.speed += acc * (delta / 1000.0 * settings.PHYSICS_SPEED)
        self.position += self.speed * (delta / 1000.0 * settings.PHYSICS_SPEED)
