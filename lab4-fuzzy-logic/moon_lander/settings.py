import os

# render settings
import pygame

FONT = os.path.join(os.path.dirname(__file__), 'fonts/digital.ttf')
FRAME_RATE = 60.

# map settings
WORLD_SIZE = (1000, 800)

# physics settings
GRAVITATION = (0., 10.)
PHYSICS_SPEED = 2.

# lander settings
LANDER_SIZE = (30., 30.)
LANDER_THRUSTER_SIZE = 60
LANDER_STARTING_BOX = (50., 50., 950., 200.)
SIDE_THRUSTERS_STRENGTH = 10.
BOTTOM_THRUSTER_STRENGTH = 20.
MAX_VERTICAL_LANDING_SPEED = 20
MAX_HORIZONTAL_LANDING_SPEED = 30

# landing zone settings
LANDING_SIZE = (120., 20.)
LANDING_STARTING_BOX = (100., 700., 900., 750.)

# controls
CONTROL_THRUSTER_BOTTOM = pygame.K_UP
CONTROL_THRUSTER_LEFT = pygame.K_RIGHT
CONTROL_THRUSTER_RIGHT = pygame.K_LEFT
