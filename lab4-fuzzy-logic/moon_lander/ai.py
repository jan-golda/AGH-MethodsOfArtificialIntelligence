import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from pygame import Vector2

from moon_lander import settings

NORM_H_POS = (-settings.WORLD_SIZE[0], settings.WORLD_SIZE[0])
NORM_V_POS = (0, settings.WORLD_SIZE[1])
NORM_H_SPEED = (-50.0, 50.0)
NORM_V_SPEED = (-50.0, 50.0)

UNIVERSE = np.arange(0.0, 1.01, 0.05)


class AIController:

    def __init__(self):
        self._init_variables()
        self._init_rules()
        self._init_simulation()

    def _init_simulation(self):
        self._ctrl = ctrl.ControlSystem(self._rules)
        self._simulation = ctrl.ControlSystemSimulation(self._ctrl)

    # =============================================================
    #  VARIABLES
    # =============================================================
    def _init_variables(self):

        # create variables
        self._h_pos = ctrl.Antecedent(UNIVERSE, 'horizontal position')
        self._v_pos = ctrl.Antecedent(UNIVERSE, 'vertical position')
        self._h_speed = ctrl.Antecedent(UNIVERSE, 'horizontal speed')
        self._v_speed = ctrl.Antecedent(UNIVERSE, 'vertical speed')
        self._h_thrust = ctrl.Consequent(UNIVERSE, 'horizontal thrust')
        self._v_thrust = ctrl.Consequent(UNIVERSE, 'vertical thrust')

        # define auto memberships
        self._h_thrust.automf(names=['very-left', 'left', 'center', 'right', 'very-right'])
        self._v_thrust.automf(names=['very-low', 'low', 'center', 'high', 'very-high'])

        # horizontal position
        self._h_pos['very-left']    = fuzz.trimf(self._h_pos.universe, [0.0, 0.0, 0.4])
        self._h_pos['left']         = fuzz.trimf(self._h_pos.universe, [0.0, 0.4, 0.5])
        self._h_pos['center']       = fuzz.trimf(self._h_pos.universe, [0.4, 0.5, 0.6])
        self._h_pos['right']        = fuzz.trimf(self._h_pos.universe, [0.5, 0.6, 1.0])
        self._h_pos['very-right']   = fuzz.trimf(self._h_pos.universe, [0.6, 1.0, 1.0])

        # vertical position
        self._v_pos['center']       = fuzz.trimf(self._v_pos.universe, [0.0, 0.0, 0.5])
        self._v_pos['high']         = fuzz.trimf(self._v_pos.universe, [0.0, 0.5, 1.0])
        self._v_pos['very-high']    = fuzz.trimf(self._v_pos.universe, [0.5, 1.0, 1.0])

        # horizontal speed
        self._h_speed['very-left']  = fuzz.trimf(self._h_speed.universe, [0.0, 0.0, 0.4])
        self._h_speed['left']       = fuzz.trimf(self._h_speed.universe, [0.0, 0.4, 0.5])
        self._h_speed['center']     = fuzz.trimf(self._h_speed.universe, [0.4, 0.5, 0.6])
        self._h_speed['right']      = fuzz.trimf(self._h_speed.universe, [0.5, 0.6, 1.0])
        self._h_speed['very-right'] = fuzz.trimf(self._h_speed.universe, [0.6, 1.0, 1.0])

        # vertical speed
        self._v_speed['very-high']  = fuzz.trimf(self._v_speed.universe, [0.0, 0.0, 0.4])
        self._v_speed['high']       = fuzz.trimf(self._v_speed.universe, [0.0, 0.4, 0.5])
        self._v_speed['center']     = fuzz.trimf(self._v_speed.universe, [0.4, 0.5, 0.6])
        self._v_speed['low']        = fuzz.trimf(self._v_speed.universe, [0.5, 0.6, 1.0])
        self._v_speed['very-low']   = fuzz.trimf(self._v_speed.universe, [0.6, 1.0, 1.0])

    # =============================================================
    #  RULES
    # =============================================================
    def _init_rules(self):
        self._rules = []

        # =============================================================
        #  VERTICAL
        # =============================================================
        # dont descent very fast
        self._rules.append(ctrl.Rule(
            self._v_speed['very-low'],
            self._v_thrust['very-high']
        ))

        # slow down when close to the landing
        self._rules.append(ctrl.Rule(
            self._v_speed['low'] & self._v_pos['center'],
            self._v_thrust['high']
        ))

        # keep descending if far from landing
        self._rules.append(ctrl.Rule(
            self._v_speed['low'] & (self._v_pos['high'] | self._v_pos['very-high']),
            self._v_thrust['center']
        ))

        # keep descending
        self._rules.append(ctrl.Rule(
            self._v_speed['center'],
            self._v_thrust['low']
        ))

        # dont go up!
        self._rules.append(ctrl.Rule(
            self._v_speed['high'],
            self._v_thrust['very-low']
        ))
        self._rules.append(ctrl.Rule(
            self._v_speed['very-high'],
            self._v_thrust['very-low']
        ))

        # =============================================================
        #  HORIZONTAL
        # =============================================================
        # center when very far from center
        self._rules.append(ctrl.Rule(
            self._h_pos['very-left'],
            self._h_thrust['right']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['very-right'],
            self._h_thrust['left']
        ))

        # but not too much
        self._rules.append(ctrl.Rule(
            self._h_pos['very-left'] & self._h_speed['very-right'],
            self._h_thrust['center']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['very-right'] & self._h_speed['very-left'],
            self._h_thrust['center']
        ))

        # center when close to the center
        self._rules.append(ctrl.Rule(
            self._h_pos['left'] & self._h_speed['center'],
            self._h_thrust['right']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['center'] & self._h_speed['left'],
            self._h_thrust['right']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['center'] & self._h_speed['center'],
            self._h_thrust['center']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['center'] & self._h_speed['right'],
            self._h_thrust['left']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['right'] & self._h_speed['center'],
            self._h_thrust['left']
        ))

        # slow down if close to the center and moving too fast
        self._rules.append(ctrl.Rule(
            self._h_pos['center'] & self._h_speed['very-left'],
            self._h_thrust['very-right']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['center'] & self._h_speed['left'],
            self._h_thrust['very-right']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['center'] & self._h_speed['right'],
            self._h_thrust['very-left']
        ))
        self._rules.append(ctrl.Rule(
            self._h_pos['center'] & self._h_speed['very-right'],
            self._h_thrust['very-left']
        ))

    # =============================================================
    #  INPUT / OUTPUT
    # =============================================================
    def input(self, pos: Vector2, speed: Vector2, landing: Vector2):
        """
        Sets input values for controller and performs computations
        :param pos: position of the lander
        :param speed: speed of the lander
        :param landing: position of the landing
        """

        # load inputs
        self._simulation.input['horizontal position'] = self._normalize(pos.x - landing.x, *NORM_H_POS)
        self._simulation.input['vertical position'] = self._normalize(landing.y - pos.y, *NORM_V_POS)
        self._simulation.input['horizontal speed'] = self._normalize(speed.x, *NORM_H_SPEED)
        self._simulation.input['vertical speed'] = self._normalize(speed.y, *NORM_V_SPEED)

        # compute outputs
        self._simulation.compute()

    def get_vertical_thrust(self) -> float:
        """
        Returns calculated vertical thrust
        :return: float in range [0.0, 1.0]
        """
        return self._simulation.output['vertical thrust']

    def get_horizontal_thrust(self) -> float:
        """
        Returns calculated horizontal thrust
        :return: float in range [-1.0, 1.0]
        """
        return self._denormalize(self._simulation.output['horizontal thrust'], -1.0, 1.0)

    # =============================================================
    #  UTILITY
    # =============================================================
    @staticmethod
    def _normalize(num: float, minimum: float, maximum: float) -> float:
        """
        Returns value normalized from range [minimum, maximum] to range [0.0, 1.0]
        :param num: number to normalize, float in range [minimum, maximum]
        :param minimum: number that corresponds to 0.0
        :param maximum: number that corresponds to 1.0
        :return: float in range [0.0, 1.0]
        """
        return (num - minimum) / (maximum - minimum)

    @staticmethod
    def _denormalize(num: float, minimum: float, maximum: float) -> float:
        """
        Returns value denormalized from range [0.0, 1.0] to range [minimum, maximum]
        :param num: number to denormalize, float in range [0.0, 1.0]
        :param minimum: number that corresponds to 0.0
        :param maximum: number that corresponds to 1.0
        :return: float in range [minimum, maximum]
        """
        return num * (maximum - minimum) + minimum
