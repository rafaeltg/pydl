import argparse
import os
import os.path

import cli.operations as op
from pydl.models.utils.utilities import load_json

parser = argparse.ArgumentParser(prog='pydl_cli')

# create global arguments
common = argparse.ArgumentParser(add_help=False)
common.add_argument('-c', '--config', dest='config', default='', required=True,
                    help='JSON file with the parameters of the operation')
common.add_argument('-o', '--output', dest='output', default='',
                    help='Path to the folder where the output of the operation will be saved')


# Parser for each command
subparsers = parser.add_subparsers(help='commands')
subparsers.add_parser('fit', parents=[common], help='Fit operation').set_defaults(func=op.fit)
subparsers.add_parser('predict', parents=[common], help='Predict operation').set_defaults(func=op.predict)
subparsers.add_parser('transform', parents=[common], help='Transform operation').set_defaults(func=op.transform)
subparsers.add_parser('reconstruct', parents=[common], help='Reconstruct operation').set_defaults(func=op.reconstruct)
subparsers.add_parser('score', parents=[common], help='Score operation').set_defaults(func=op.score)
subparsers.add_parser('validate', parents=[common], help='Validate operation').set_defaults(func=op.validate)
subparsers.add_parser('optimize', parents=[common], help='Optimize operation').set_defaults(func=op.optimize)

args = parser.parse_args()


def run():
    output = args.output if args.output != '' else os.getcwd()
    args.func(load_json(args.config), output)


if __name__ == "__main__":
    run()
