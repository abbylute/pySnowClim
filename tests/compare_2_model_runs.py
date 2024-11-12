"""
This script compares outputs from two snow model runs by extracting subsets of data 
from `.npy` files and comparing them with a predefined control dataset. The script 
is designed to validate the consistency of model output variables across different 
model runs.

### Script Overview:
1. **Input**: The script takes as input the path to the directory where the model outputs 
   (in `.npy` format) are stored. Optionally, it can also take a list of variables to 
   compare between the model runs.
2. **Data Extraction**: The script extracts subsets of data for specific time and spatial 
   indices from the model outputs.
3. **Comparison**: The script compares the extracted data from a new model run with a 
   known control dataset (stored as a `.nc` file). It checks if the datasets are identical 
   and identifies the first timestep where they diverge, printing this information to the console.

### Functionality:
- **_extract_result_subset(output_dir, time_indices, location_indices, variables)**:
  Extracts a spatiotemporal subset of model output data from `.npy` files located in the 
  specified directory. It returns an `xarray.Dataset` with the subsetted data for the 
  requested variables.

- **_compare_datasets(new, control, variables)**:
  Compares two `xarray.Dataset` objects—one from a new model run and one from a 
  predefined control dataset. The comparison is done across the specified variables, 
  and the function prints any discrepancies, including the first timestep when the 
  variables diverge.

- **_compare_2_model_runs(output_dir='none', variables='none')**:
  The main function that integrates the previous functions. It loads the control dataset, 
  extracts the relevant model output from `.npy` files in the given directory, and compares 
  the new model output with the control dataset.

### Execution:
When executed directly, the script:
1. Takes input from the command line for the directory containing model output files and 
   optionally for the list of variables to compare.
2. Loads the control dataset (`control.nc`) and identifies the time and location indices 
   for the comparison.
3. Extracts the new model outputs based on the provided indices and compares them with 
   the control dataset.
4. Prints the results of the comparison to the console, including any variables and 
   timesteps where discrepancies occur.

### Command-Line Arguments:
- **path**: The path to the directory containing the model output `.npy` files.
- **variables**: Optionally, a list of variables to compare. If not provided, a default list
  of variables will be used.
- **example of running the code from the command line and comparing two variables:
  python pySnowClim/tests/compare_2_model_runs.py 'pySnowClim/outputs/' SnowWaterEq SnowMelt

### Dependencies:
- **xarray**: For handling multi-dimensional data arrays.
- **numpy**: For loading and manipulating data from `.npy` files.
- **argparse**: For handling command-line arguments and providing a user-friendly interface.

### Use Case:
This script is useful for comparing model outputs from different runs of a snow model (or from 
different configurations of the same model). It is primarily designed to help users identify 
differences in model behavior and validate model results.

"""


import xarray as xr
import numpy as np
import argparse


def _extract_result_subset(output_dir, time_indices, location_indices, variables):
    """ 
    This function extracts a spatiotemporal subset of the SnowClim outputs
    to compare. It loads results from .npy files, extracts a subset, and saves the
    subset to a netcdf.

    Parameters:
    - output_dir: str, directory where the outputs are located
    - time_indices: array of temporal indices
    - location_indices: array of spatial indices
    - variables: list of variables to extract data for

    Returns:
    - x: xarray dataset containing the subsetted data for each variable
    """

    # create an empty xarray that can be filled with subset values
    x = xr.Dataset(None, coords=dict(time=time_indices, location=location_indices))

    for v in variables:
        # load the snowclim outputs
        output = np.load(output_dir + v + '.npy')

        # extract the subset
        output_subset = output[time_indices,:][:,location_indices]

        # assign the subset to the xarray dataset
        x[v] = xr.DataArray(output_subset, dims=("time","location"))
    return x


def _compare_datasets(new, control, variables):
    """
    Compare two xarray datasets of snow model outputs

    Parameters:
    - new: xarray dataset containing a subset of snow model outputs
    - control: xarray dataset containing a subset of snow model outputs from a known run
    - variables: list of snow model output variables to compare

    Returns:
    - prints to screen information regarding the comparison
    """

    # first check that we are comparing the same time and space indices
    if not all(new.time.values == control.time.values):
        print('time indices are not the same in the two datasets')
        return
    if not all(new.location.values == control.location.values):
        print('location indices are not the same in the two datasets')
        return

    # check if the whole dataset is equal
    if new.equals(control):
        print('New dataset is the same as the control dataset')

    # if the datasets are not equal, make a list of variables that are not equal
    # and the first time step when they start to diverge
    else:
        print('New dataset is NOT the same as the control dataset\n')
        print('Variables which are not the same include:')
        vars_not_equal = []
        for v in variables:
            if not new[v].equals(control[v]):
                tf = new[v]==control[v]
                times_with_diffs = np.sum(tf,1).to_numpy()
                timestep = np.argmax(times_with_diffs)
                print(v + ' starting at timestep ' + str(timestep))

def _compare_2_model_runs(output_dir='none', variables='none'):
    """
    Runs a simple comparison of two snow model output datasets
    """

    if output_dir == 'none':
        output_dir = './pySnowClim/outputs/'
    if variables == 'none':
        variables = ['SnowfallWaterEq','SnowWaterEq','SnowMelt',
             'LW_down','LW_up','SW_down','SW_up',
             'Q_latent','Q_sensible','Q_precip',
             'SnowTemp','Albedo','Energy','PackCC',
             'Sublimation','Condensation','RefrozenWater']

    # load the control dataset
    control = xr.open_dataset('./pySnowClim/tests/datasets/control.nc')

    # identify the time and location indices
    time_indices = control.time.values
    location_indices = control.location.values

    # subset the newly created model outputs
    x = _extract_result_subset(output_dir, time_indices, location_indices, variables)

    # compare
    _compare_datasets(x, control, variables)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compare outputs from two model runs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path',
                        help='Path where model outputs are located',
                        default='none')

    parser.add_argument('variables',
                        help='Variables to compare',
                        nargs='*',
                        const=None,
                        default='none')

    args = vars(parser.parse_args())

    _compare_2_model_runs(args['path'], args['variables'])
