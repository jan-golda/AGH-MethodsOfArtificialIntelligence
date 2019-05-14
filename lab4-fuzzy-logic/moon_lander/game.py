import sys
from typing import List

import pygame
import pygame.freetype
from pygame.rect import Rect

from moon_lander import settings
from moon_lander.lander import PlayerLander, AILander
from moon_lander.landing_zone import LandingZone
from moon_lander.objects import GameObject


# custom events
RESET_GAME = pygame.USEREVENT + 1


class Game:
    """ Main game class, manages game loop and stores game objects """

    def __init__(self, ai: bool = False):
        """
        Game setup
        :param ai: if lander should be controlled by AI instead of player
        """
        # setup game
        self.ai = ai
        self.score_success = 0
        self.score_fail = 0
        self.game_over = False
        self.game_objects: List[GameObject] = []

        # setup PyGame
        pygame.init()
        self.surface = pygame.display.set_mode(settings.WORLD_SIZE)
        pygame.display.set_caption('Fuzzy Logic: Moon Lander' + (' [AI]' if ai else ''))
        self.clock = pygame.time.Clock()
        self.font = pygame.freetype.Font(settings.FONT, 36)

        # setup game objects
        self._reset_game()

    def _reset_game(self):
        """ Reset game to initial state by creating new lander and landing zone """
        del self.game_objects[:]

        # create lander
        if self.ai:
            self.game_objects.append(AILander(self))
        else:
            self.game_objects.append(PlayerLander(self))

        # create landing zone
        self.game_objects.append(LandingZone(self))

    def handle_events(self):
        """ Handler events from PyGame """
        events = pygame.event.get()

        # handle game events
        for event in events:
            # exit game
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # reset game
            if event.type == RESET_GAME:
                if event.success:
                    self.score_success += 1
                else:
                    self.score_fail += 1
                self._reset_game()

        # allow game objects to also handle events
        for o in self.game_objects:
            o.handle_events(events)

    def update(self, delta: int):
        """ Updates state of the game """
        for o in self.game_objects:
            o.update(delta)

    def draw(self):
        """ Draws game objects on surface """
        # draw game objects
        for o in self.game_objects:
            o.draw(self.surface)

        # draw score
        self.font.render_to(self.surface, (10, 10), f"{self.score_success} - {self.score_fail}", pygame.Color('white'))

    def run(self):
        """ Starts main game loop """
        while not self.game_over:
            self.clock.tick(settings.FRAME_RATE)

            # clear surface
            self.surface.fill(pygame.Color('black'))

            # game loop
            self.handle_events()
            self.update(self.clock.get_time())
            self.draw()

            # send surface to display
            pygame.display.update()

    @property
    def map_border(self):
        """ Returns Rect representing border of the map"""
        return Rect(0, 0, *settings.WORLD_SIZE)

    def get_colliding(self, obj: GameObject, with_self=False):
        """ Returns all colliding game objects """
        return [
            self.game_objects[i]
            for i in obj.bounds.collidelistall([
                obj.bounds
                for obj in self.game_objects
            ])
            if self.game_objects[i] is not obj or with_self
        ]

    def reset(self, success: bool):
        """ Triggers reset game event """
        pygame.event.post(pygame.event.Event(RESET_GAME, success=success))