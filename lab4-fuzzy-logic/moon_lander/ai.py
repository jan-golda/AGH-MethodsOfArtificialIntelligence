import numpy as np
import skfuzzy.control as ctrl
from pygame import Vector2

from moon_lander import settings


class AIController:

    def __init__(self):

        # inputs

        self._horizontal_speed = ctrl.Antecedent(
            np.arange(-100, 101, 10),
            'horizontal speed'
        )
        self._horizontal_position = ctrl.Antecedent(
            np.arange(-settings.WORLD_SIZE[0], settings.WORLD_SIZE[0]+1, 10),
            'horizontal position'
        )

        self._vertical_speed = ctrl.Antecedent(
            np.arange(-150, 151, 10),
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
        horizontal_names = ['high-left', 'left', 'center', 'right', 'high-right']
        self._horizontal_speed.automf(names=horizontal_names)
        self._horizontal_position.automf(names=horizontal_names)
        self._horizontal_thrust.automf(names=horizontal_names)

        self._vertical_speed.automf(names=['up', 'static', 'down'])
        self._vertical_position.automf(names=['low', 'center', 'high'])
        self._vertical_thrust.automf(names=['none', 'very-low', 'low', 'high', 'very-high'])

        # rules
        horizontal_rules = [
            ('high-left',   'high-left',    'high-right'),
            ('high-left',   'left',         'high-right'),
            ('high-left',   'center',       'right'),
            ('high-left',   'right',        'center'),
            ('high-left',   'high-right',   'left'),
            ('left',        'high-left',    'high-right'),
            ('left',        'left',         'high-right'),
            ('left',        'center',       'right'),
            ('left',        'right',        'center'),
            ('left',        'high-right',   'left'),
            ('center',      'high-left',    'high-right'),
            ('center',      'left',         'high-right'),
            ('center',      'center',       'center'),
            ('center',      'right',        'high-left'),
            ('center',      'high-right',   'high-left'),
            ('right',       'high-left',    'right'),
            ('right',       'left',         'center'),
            ('right',       'center',       'left'),
            ('right',       'right',        'high-left'),
            ('right',       'high-right',   'high-left'),
            ('high-right',  'high-left',    'right'),
            ('high-right',  'left',         'center'),
            ('high-right',  'center',       'left'),
            ('high-right',  'right',        'high-left'),
            ('high-right',  'high-right',   'high-left'),
        ]

        self._rules = [
            ctrl.Rule(self._horizontal_position[p] & self._horizontal_speed[s], self._horizontal_thrust[t])
            for p, s, t in horizontal_rules
        ]

        # rules
        self._rules += [

            # vertical
            ctrl.Rule(
                self._vertical_position['low'],
                self._vertical_thrust['high']
            ),
            ctrl.Rule(
                self._vertical_position['low'] & self._vertical_speed['down'],
                self._vertical_thrust['very-high']
            ),
            ctrl.Rule(
                self._vertical_position['high'],
                self._vertical_thrust['very-low']
            ),
            ctrl.Rule(
                self._vertical_position['high'] & self._vertical_speed['up'],
                self._vertical_thrust['none']
            ),
            ctrl.Rule(
                self._vertical_position['center'] & self._vertical_speed['down'],
                self._vertical_thrust['high']
            ),
            ctrl.Rule(
                self._vertical_position['center'],
                self._vertical_thrust['low']
            ),
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
