import argparse

from moon_lander.ai import AIController
from moon_lander.game import Game


def parse_args():
    parser = argparse.ArgumentParser(description='Start Fuzzy Logic: Moon Lander game')

    parser.add_argument('--ai', action='store_true', help='Start with AI enabled')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # get controller if requested
    controller = None
    if args.ai:
        controller = AIController()

    Game(controller).run()
