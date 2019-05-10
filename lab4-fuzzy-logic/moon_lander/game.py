import sys
from typing import List

import pygame
import pygame.freetype

from moon_lander import settings
from moon_lander.lander import PlayerLander, AILander
from moon_lander.landing_zone import LandingZone
from moon_lander.objects import GameObject


class Game:

    def __init__(self, ai: bool = False):
        # setup game
        self.ai = ai
        self.score = 0
        self.game_over = False
        self.game_objects: List[GameObject] = []

        # setup PyGame
        pygame.init()
        self.surface = pygame.display.set_mode(settings.WORLD_SIZE)
        pygame.display.set_caption('Fuzzy Logic: Moon Lander' + (' [AI]' if ai else ''))
        self.clock = pygame.time.Clock()
        self.font = pygame.freetype.Font(settings.FONT, 36)

        # setup game objects
        self.reset_game()

    def reset_game(self):
        self.game_objects = []

        # create lander
        if self.ai:
            self.game_objects.append(AILander())
        else:
            self.game_objects.append(PlayerLander())

        # create landing zone
        self.game_objects.append(LandingZone())

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def update(self, delta: int):
        for o in self.game_objects:
            o.update(delta)

    def draw(self):
        for o in self.game_objects:
            o.draw(self.surface)

        self.font.render_to(self.surface, (10, 10), str(self.score), pygame.Color('white'))

    def run(self):
        while not self.game_over:
            self.clock.tick(settings.FRAME_RATE)

            self.surface.fill(pygame.Color('black'))

            self.handle_events()
            self.update(self.clock.get_time())
            self.draw()

            pygame.display.update()
