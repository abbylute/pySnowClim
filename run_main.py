"""Main script to run pySnowClim."""
import os
import sys
import argparse

# Add the 'code' directory to the Python path
sys.path.append('src/')

#import pdb; pdb.set_trace()

from runsnowclim_model import run_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='pySnowClim model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path',
                        help='Path where model inputs are located',
                        default='data/',
                        nargs='?')

    parser.add_argument('output',
                        help='Path where model outputs will be saved',
                        default='outputs/',
                        nargs='?')

    parser.add_argument('parameters',
                        help='Path where the parameters file is located',
                        default=None,
                        nargs='?')

    parser.add_argument('save_format',
                        help='File format to be saved (.npy or .nc)',
                        default=None,
                        nargs='?')

    args = vars(parser.parse_args())

    print('Starting pySnowClim model...')
    model_output = run_model(args['path'], args['parameters'], args['output'],
                             args['save_format'])
    print('pySnowClim model finished!')
