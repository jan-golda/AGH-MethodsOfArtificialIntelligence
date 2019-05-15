import argparse
from threading import Thread

from moon_lander import settings
from moon_lander.ai import AIController
from moon_lander.game import Game
from scikit_fuzzy_visualizer import FuzzyVariablesVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Start Fuzzy Logic: Moon Lander game')

    parser.add_argument('--ai', action='store_true', help='Start with AI enabled')
    parser.add_argument('-s', '--speed', type=float, default=1., help='Physics speed multiplier')
    parser.add_argument('-v', '--visualize', action='store_true', help='Show fuzzy logic visualization')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # set physics speed
    settings.PHYSICS_SPEED *= args.speed

    # get controller if requested
    controller = None
    if args.ai:
        controller = AIController()

    # start game
    game = Game(controller)
    Thread(target=game.run).start()

    # start visualization if requested
    if args.ai and args.visualize:
        visualizer = FuzzyVariablesVisualizer(controller._simulation)
        visualizer.run()
