"""
This script runs the SnowClim Model using an example of 4-hourly forcing data for the water year 2002
for an area in Central Idaho. It:
 - Sets default parameter values
 - Imports example forcing data
 - Runs the SnowClim model
 - Plots some results

The script can be adapted for different parameters, time periods, or datasets.
"""
import os
import pickle
import scipy.io
import numpy as np

from createParameterFile import create_dict_parameters
from snowclim_model import run_snowclim_model


def _load_forcing_data(data_dir):
    """
    Load meteorological forcing data and geospatial information from the specified directory.

    Args:
        data_dir (str): Path to the directory containing the forcing data files.

    Returns:
        dict: Dictionary containing the loaded variables (lat, lon, lrad, solar, tavg, ppt, vs, psfc, huss, relhum, tdmean).
    """
    # Load latitude, longitude, and elevation data
    latlonelev = scipy.io.loadmat(f'{data_dir}lat_lon_elev.mat')
    lat = latlonelev['lat']
    lon = latlonelev['lon']

    # Load meteorological data (forcing inputs)
    lrad = scipy.io.loadmat(f'{data_dir}lrad.mat')['lrad']
    solar = scipy.io.loadmat(f'{data_dir}solar.mat')['solar']
    tavg = scipy.io.loadmat(f'{data_dir}tavg.mat')['tavg']
    ppt = scipy.io.loadmat(f'{data_dir}ppt.mat')['ppt']
    vs = scipy.io.loadmat(f'{data_dir}vs.mat')['vs']
    psfc = scipy.io.loadmat(f'{data_dir}psfc.mat')['psfc']
    huss = scipy.io.loadmat(f'{data_dir}huss.mat')['huss']
    relhum = scipy.io.loadmat(f'{data_dir}relhum.mat')['relhum']
    tdmean = scipy.io.loadmat(f'{data_dir}tdmean.mat')['tdmean']

    # Return the data as a dictionary
    return {'coords':
            {
            'lat': lat,
            'lon': lon,
            'time': None,
            },
        'forcings':
            {
            'lrad': lrad,
            'solar': solar,
            'tavg': tavg,
            'ppt': ppt,
            'vs': vs,
            'psfc': psfc,
            'huss': huss,
            'relhum': relhum,
            'tdmean': tdmean
        }
    }


def _load_parameter_file(parameterfilename):
    """
    Load the parameters from a pickle file if it exists.

    Args:
        parameterfilename (str): Name of the file to load parameters from.

    Returns:
        dict or None: Dictionary of parameters if file exists, otherwise None.
    """
    if parameterfilename is None:
        parameters = create_dict_parameters()
    else:
        if os.path.exists(parameterfilename):
            print(f"Loading parameters from {parameterfilename}")
            mat_data = scipy.io.loadmat(parameterfilename)

            # Remove metadata fields that start with '__' (like '__header__', '__version__', etc.)
            parameters = {key: value for key, value in mat_data.items() if not key.startswith('__')}
            parameters = parameters['S']
        else:
            parameters = create_dict_parameters()

    return parameters


def _save_outputs(model_output, outputs_path=None):
    """
    Save model outputs to file.

    Args:
        model_output (class): class containing snow model outputs
        outputs_path (str): name of directory to save outputs to.

    Returns:
        nothing
    """
    if outputs_path is not None:
    # TODO: here or perhaps before the model runs, add a check that the output directory exists
    # TODO: for now, saving these as .npy because it is easy and avoids added complications.
    #       May want to save as netcdf eventually.
        n_locations = model_output[0].SnowWaterEq.shape[0]
        variables_to_save = [x for x in dir(model_output[0]) if not x.startswith('__')]
        for v in variables_to_save:
            var_data = np.empty((len(model_output),n_locations))
            for t in range(len(model_output)):
                var_data[t,:] = getattr(model_output[t],v)
            np.save(outputs_path + v + '.npy', var_data)



def run_model(forcings_path, parameters_path, outputs_path):

    print('Loading necessary files...')
    parameters = _load_parameter_file(parameters_path)
    forcings_data = _load_forcing_data(forcings_path)
    forcings_data['coords']['time'] = parameters['cal']

    print('Files loaded, running the model...')
    model_output = run_snowclim_model(forcings_data, parameters)
    _save_outputs(model_output, outputs_path)

    return model_output
