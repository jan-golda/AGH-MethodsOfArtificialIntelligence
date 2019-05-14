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

    def __init__(self, controller=None):
        """
        Game setup
        :param ai: if lander should be controlled by AI instead of player
        """
        # setup game
        self.score_success = 0
        self.score_fail = 0
        self.game_over = False
        self.controller = controller

        # setup PyGame
        pygame.init()
        self.surface = pygame.display.set_mode(settings.WORLD_SIZE)
        pygame.display.set_caption('Fuzzy Logic: Moon Lander' + (' [AI]' if controller else ''))
        self.clock = pygame.time.Clock()
        self.font = pygame.freetype.Font(settings.FONT, 36)

        # setup game objects
        self._reset_game()

    def _reset_game(self):
        """ Reset game to initial state by creating new lander and landing zone """

        # create landing zone
        self.landing = LandingZone()

        # create lander
        if self.controller:
            self.lander = AILander(self.landing, self.controller)
        else:
            self.lander = PlayerLander(self.landing)

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

        self._check_collisions()

    def draw(self):
        """ Draws game objects on surface """
        # draw game objects
        for o in self.game_objects:
            o.draw(self.surface)

        # draw score
        self.font.render_to(self.surface, (10, 10), f"{self.score_success} / {self.score_fail + self.score_success}", pygame.Color('white'))

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

    @property
    def game_objects(self) -> List[GameObject]:
        """ Returns all game objects in this game"""
        return [self.lander, self.landing]

    def reset(self, success: bool):
        """ Triggers reset game event """
        pygame.event.post(pygame.event.Event(RESET_GAME, success=success))

    def _check_collisions(self):

        # lander inside borders
        if not self.map_border.collidepoint(self.lander.position):
            return self.reset(False)

        # lander under the landing zone
        if self.lander.bounds.bottom > self.landing.bounds.centery:
            return self.reset(False)

        # lander collide with landing
        if self.lander.bounds.colliderect(self.landing.bounds):
            if self.lander.bounds.right > self.landing.bounds.right or self.lander.bounds.left < self.landing.bounds.left:
                return self.reset(False)
            if abs(self.lander.speed.x) > settings.MAX_HORIZONTAL_LANDING_SPEED:
                return self.reset(False)
            if abs(self.lander.speed.y) > settings.MAX_VERTICAL_LANDING_SPEED:
                return self.reset(False)
            return self.reset(True)
