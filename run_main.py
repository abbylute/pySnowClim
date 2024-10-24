"""Main script to run pySnowClim."""
import os
import sys
# Add the 'code' directory to the Python path
sys.path.append('src/')

from runsnowclim_model import run_model


if __name__ == "__main__":

    print('Starting pySnowClim model...')
    
    model_output = run_model('data/', 'data/parameters.mat')
    print('pySnowClim model finished!')


