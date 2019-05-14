import numpy as np
import skfuzzy.control as ctrl
from pygame import Vector2

from moon_lander import settings


class AIController:

    def __init__(self):

        # inputs

        self._horizontal_speed = ctrl.Antecedent(
            np.arange(-60, 61, 10),
            'horizontal speed'
        )
        self._horizontal_position = ctrl.Antecedent(
            np.arange(-settings.WORLD_SIZE[0], settings.WORLD_SIZE[0]+1, 10),
            'horizontal position'
        )

        self._vertical_speed = ctrl.Antecedent(
            np.arange(-60, 61, 10),
            'vertical speed'
        )
        self._vertical_position = ctrl.Antecedent(
            np.arange(-settings.WORLD_SIZE[1], settings.WORLD_SIZE[1]+1, 10),
            'vertical position'
        )

        # outputs
        self._vertical_thrust = ctrl.Consequent(
            np.arange(0.0, 1.0, 0.05),
            'vertical thrust'
        )
        self._horizontal_thrust = ctrl.Consequent(
            np.arange(-1.0, 1.0, 0.05),
            'horizontal thrust'
        )

        # auto membership
        horizontal_names = ['very-left', 'left', 'center', 'right', 'very-right']
        self._horizontal_speed.automf(names=horizontal_names)
        self._horizontal_position.automf(names=horizontal_names)
        self._horizontal_thrust.automf(names=horizontal_names)

        vertical_names = ['very-low', 'low', 'center', 'high', 'very-high']
        self._vertical_speed.automf(names=vertical_names, invert=True)
        self._vertical_position.automf(names=vertical_names, invert=True)
        self._vertical_thrust.automf(names=vertical_names)

        # rules
        horizontal_rules = [
            ('very-left',   'very-left',    'very-right'),
            ('very-left',   'left',         'very-right'),
            ('very-left',   'center',       'right'),
            ('very-left',   'right',        'center'),
            ('very-left',   'very-right',   'left'),
            ('left',        'very-left',    'very-right'),
            ('left',        'left',         'very-right'),
            ('left',        'center',       'right'),
            ('left',        'right',        'center'),
            ('left',        'very-right',   'left'),
            ('center',      'very-left',    'very-right'),
            ('center',      'left',         'very-right'),
            ('center',      'center',       'center'),
            ('center',      'right',        'very-left'),
            ('center',      'very-right',   'very-left'),
            ('right',       'very-left',    'right'),
            ('right',       'left',         'center'),
            ('right',       'center',       'left'),
            ('right',       'right',        'very-left'),
            ('right',       'very-right',   'very-left'),
            ('very-right',  'very-left',    'right'),
            ('very-right',  'left',         'center'),
            ('very-right',  'center',       'left'),
            ('very-right',  'right',        'very-left'),
            ('very-right',  'very-right',   'very-left'),
        ]
        
        vertical_rules = [
            ('very-low',    'very-low',     'very-high'),
            ('very-low',    'low',          'very-high'),
            ('very-low',    'center',       'high'),
            ('very-low',    'high',         'center'),
            ('very-low',    'very-high',    'low'),
            ('low',         'very-low',     'very-high'),
            ('low',         'low',          'high'),
            ('low',         'center',       'high'),
            ('low',         'high',         'very-low'),
            ('low',         'very-high',    'very-low'),
            ('center',      'very-low',     'very-high'),
            ('center',      'low',          'very-high'),
            ('center',      'center',       'center'),
            ('center',      'high',         'very-low'),
            ('center',      'very-high',    'very-low'),
            ('high',        'very-low',     'very-high'),
            ('high',        'low',          'very-high'),
            ('high',        'center',       'low'),
            ('high',        'high',         'low'),
            ('high',        'very-high',    'very-low'),
            ('very-high',   'very-low',     'high'),
            ('very-high',   'low',          'center'),
            ('very-high',   'center',       'low'),
            ('very-high',   'high',         'very-low'),
            ('very-high',   'very-high',    'very-low'),
        ]

        self._rules = []
        self._rules += [
            ctrl.Rule(self._horizontal_position[p] & self._horizontal_speed[s], self._horizontal_thrust[t])
            for p, s, t in horizontal_rules
        ]
        self._rules += [
            ctrl.Rule(self._vertical_position[p] & self._vertical_speed[s], self._vertical_thrust[t])
            for p, s, t in vertical_rules
        ]

        # control system
        self._ctrl = ctrl.ControlSystem(self._rules)

        # simulation
        self._simulation = ctrl.ControlSystemSimulation(self._ctrl)

    def input(self, pos: Vector2, speed: Vector2, landing: Vector2):
        self._simulation.input['horizontal position'] = pos.x - landing.x
        self._simulation.input['vertical position'] = pos.y - landing.y

        self._simulation.input['horizontal speed'] = speed.x
        self._simulation.input['vertical speed'] = speed.y

        self._simulation.compute()

    def view_memberships(self):
        self._vertical_speed.view()
        self._horizontal_speed.view()
        self._vertical_position.view()
        self._horizontal_position.view()
        self._vertical_thrust.view()
        self._horizontal_thrust.view()

    def view_output(self):
        self._vertical_thrust.view(self._simulation)
        self._horizontal_thrust.view(self._simulation)

    def get_vertical_thrust(self):
        return self._simulation.output['vertical thrust']

    def get_horizontal_thrust(self):
        return self._simulation.output['horizontal thrust']
