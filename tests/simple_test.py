import xarray as xr
import numpy as np


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

def _run_simple_test(output_dir='none', variables='none'):
    """
    Runs a simple comparison of two snow model output datasets
    """

    if output_dir == 'none':
        output_dir = '/home/abbylute/pySnowClim/outputs/'
    if variables == 'none':
        variables = ['SnowfallWaterEq','SnowWaterEq','SnowMelt',
             'LW_down','LW_up','SW_down','SW_up',
             'Q_latent','Q_sensible','Q_precip',
             'SnowTemp','Albedo','Energy','PackCC',
             'Sublimation','Condensation','RefrozenWater']

    # load the control dataset
    control = xr.open_dataset('~/pySnowClim/tests/datasets/control.nc')

    # identify the time and location indices
    time_indices = control.time.values
    location_indices = control.location.values

    # subset the newly created model outputs
    x = _extract_result_subset(output_dir, time_indices, location_indices, variables)

    # compare
    _compare_datasets(x, control, variables)

_run_simple_test()
