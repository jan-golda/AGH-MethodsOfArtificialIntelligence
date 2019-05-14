import numpy as np
import skfuzzy.control as ctrl
from pygame import Vector2

from moon_lander import settings


class AIController:

    def __init__(self):

        # inputs
        self._vertical_speed = ctrl.Antecedent(
            np.arange(-150, 150, 10),
            'vertical speed'
        )
        self._horizontal_speed = ctrl.Antecedent(
            np.arange(-150, 150, 10),
            'horizontal speed'
        )
        self._vertical_position = ctrl.Antecedent(
            np.arange(-settings.WORLD_SIZE[1], settings.WORLD_SIZE[1], 10),
            'vertical position'
        )
        self._horizontal_position = ctrl.Antecedent(
            np.arange(-settings.WORLD_SIZE[0], settings.WORLD_SIZE[0], 10),
            'horizontal position'
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
        self._vertical_speed.automf(names=['up', 'static', 'down'])
        self._horizontal_speed.automf(names=['left', 'static', 'right'])
        self._vertical_position.automf(names=['low', 'center', 'high'])
        self._horizontal_position.automf(names=['left', 'center', 'right'])
        self._vertical_thrust.automf(names=['none', 'very-low', 'low', 'high', 'very-high'])
        self._horizontal_thrust.automf(names=['very-left', 'left', 'none', 'right', 'very-right'])

        # rules
        self._rules = [

            # horizontal
            ctrl.Rule(
                self._horizontal_position['left'],
                self._horizontal_thrust['right']
            ),
            ctrl.Rule(
                self._horizontal_position['left'] & self._horizontal_speed['left'],
                self._horizontal_thrust['very-right']
            ),
            ctrl.Rule(
                self._horizontal_position['right'],
                self._horizontal_thrust['left']
            ),
            ctrl.Rule(
                self._horizontal_position['right'] & self._horizontal_speed['right'],
                self._horizontal_thrust['very-left']
            ),
            ctrl.Rule(
                self._horizontal_position['center'],
                self._horizontal_thrust['none']
            ),


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
