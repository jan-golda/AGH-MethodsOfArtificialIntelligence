import numpy as np
import skfuzzy.control as ctrl
from pygame import Vector2

from moon_lander import settings


class AIController:

    def __init__(self):

        # inputs

        self._h_speed = ctrl.Antecedent(
            np.arange(-50, 51, 10),
            'horizontal speed'
        )
        self._h_pos = ctrl.Antecedent(
            np.arange(-settings.WORLD_SIZE[0], settings.WORLD_SIZE[0]+1, 10),
            'horizontal position'
        )

        self._v_speed = ctrl.Antecedent(
            np.arange(-60, 61, 10),
            'vertical speed'
        )
        self._v_pos = ctrl.Antecedent(
            np.arange(-settings.WORLD_SIZE[1], settings.WORLD_SIZE[1]+1, 10),
            'vertical position'
        )

        # outputs
        self._v_thrust = ctrl.Consequent(
            np.arange(0.0, 1.0, 0.05),
            'vertical thrust'
        )
        self._h_thrust = ctrl.Consequent(
            np.arange(-1.0, 1.0, 0.05),
            'horizontal thrust'
        )

        # auto membership
        horizontal_names = ['very-left', 'left', 'center', 'right', 'very-right']
        self._h_speed.automf(names=horizontal_names)
        self._h_pos.automf(names=horizontal_names)
        self._h_thrust.automf(names=horizontal_names)

        vertical_names = ['very-low', 'low', 'center', 'high', 'very-high']
        self._v_speed.automf(names=vertical_names, invert=True)
        self._v_pos.automf(names=vertical_names, invert=True)
        self._v_thrust.automf(names=vertical_names)

        # rules
        
        vertical_rules = [
            ('center',      'very-low',     'very-high'),
            ('center',      'low',          'very-high'),
            ('center',      'center',       'center'),
            ('high',        'very-low',     'very-high'),
            ('high',        'low',          'very-high'),
            ('high',        'center',       'low'),
            ('high',        'high',         'low'),
            ('high',        'very-high',    'low'),
            ('very-high',   'very-low',     'center'),
            ('very-high',   'low',          'center'),
            ('very-high',   'center',       'low'),
            ('very-high',   'high',         'low'),
            ('very-high',   'very-high',    'low'),
        ]

        self._rules = []
        self._rules.append(ctrl.Rule(self._h_pos['very-left'], self._h_thrust['very-right']))
        self._rules.append(ctrl.Rule(self._h_pos['very-right'], self._h_thrust['very-left']))
        self._rules.append(ctrl.Rule(self._h_pos['very-left'] & self._h_speed['very-right'], self._h_thrust['right']))
        self._rules.append(ctrl.Rule(self._h_pos['very-right'] & self._h_speed['very-left'], self._h_thrust['left']))

        self._rules.append(ctrl.Rule(self._h_pos['left'], self._h_thrust['right']))
        self._rules.append(ctrl.Rule(self._h_pos['right'], self._h_thrust['left']))
        self._rules.append(ctrl.Rule(self._h_pos['left'] & self._h_speed['very-right'], self._h_thrust['left']))
        self._rules.append(ctrl.Rule(self._h_pos['right'] & self._h_speed['very-left'], self._h_thrust['right']))

        # centering horizontally close to landing
        self._rules.append(ctrl.Rule(self._h_pos['center'] & self._h_speed['very-right'], self._h_thrust['very-left']))
        self._rules.append(ctrl.Rule(self._h_pos['center'] & self._h_speed['right'], self._h_thrust['left']))
        self._rules.append(ctrl.Rule(self._h_pos['center'] & self._h_speed['center'], self._h_thrust['center']))
        self._rules.append(ctrl.Rule(self._h_pos['center'] & self._h_speed['left'], self._h_thrust['right']))
        self._rules.append(ctrl.Rule(self._h_pos['center'] & self._h_speed['very-left'], self._h_thrust['very-right']))

        # if is very high and not over landing slow down descend by little
        self._rules.append(ctrl.Rule(
            self._v_pos['very-high'] & (self._h_pos['very-left'] | self._h_pos['very-right']),
            self._v_thrust['high']
        ))

        # if is high and not over landing slow down descend
        self._rules.append(ctrl.Rule(
            (self._v_pos['high'] | self._v_pos['center']) & (self._h_pos['left'] | self._h_pos['right'] | self._h_pos['very-left'] | self._h_pos['very-right']),
            self._v_thrust['very-high']
        ))

        # fast corrections when very low
        self._rules.append(ctrl.Rule(
            self._v_pos['center'] & (self._h_pos['left'] | self._h_pos['very-left']),
            self._h_thrust['very-right']
        ))
        self._rules.append(ctrl.Rule(
            self._v_pos['center'] & (self._h_pos['right'] | self._h_pos['very-right']),
            self._h_thrust['very-left']
        ))

        self._rules += [
            ctrl.Rule(self._v_pos[p] & self._v_speed[s], self._v_thrust[t])
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
        self._v_speed.view()
        self._h_speed.view()
        self._v_pos.view()
        self._h_pos.view()
        self._v_thrust.view()
        self._h_thrust.view()

    def view_output(self):
        self._v_thrust.view(self._simulation)
        self._h_thrust.view(self._simulation)

    def get_vertical_thrust(self):
        return self._simulation.output['vertical thrust']

    def get_horizontal_thrust(self):
        return self._simulation.output['horizontal thrust']
