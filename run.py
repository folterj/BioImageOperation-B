import argparse
from importlib import import_module
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser("BioImageOperation-B")
    parser.add_argument('--params',
                        required=True,
                        help='The location of the parameters file')
    args = parser.parse_args()
    with open(args.params, 'r') as file:
        params = yaml.safe_load(file)

    for operation0 in params['operations']:
        operation = next(iter(operation0))
        module = import_module(f'src.pipeline.{operation}')
        print(f'[Operation: {operation}]')
        module.run(params, operation0[operation])
