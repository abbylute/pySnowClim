# pySnowClim
Welcome to the `pySnowClim` repo.

## What is it?

`pySnowClim`  is a robust, efficient, and open-source Python implementation of
the original SnowClim model written in MATLAB.
`pySnowClim` addresses the practical need for a snow models that can accurately
simulate snow accumulation and melt across both large regions and fine spatial detail.
`pySnowClim` combines key physical processes with efficient algorithms to
provide high-resolution snowpack estimates, especially useful in complex terrain.
Many current snow models are either too simple, too computationally costly,
or not easy to use with modern workflows.

Users that might be interested on this package include researchers,
hydrologists, ecologists, students, and land
managers who need gridded information about snow for water resource
assessments, ecological studies, and climate impact analysis.

## Requirements
Before running pySnowClim, ensure that you have the following installed:

* Python 3.x
* Necessary Python packages (if any are needed, such as numpy, scipy, etc.)

## Running the Model
1. Prepare Input Data
To run the model, you need the following files:

* **Meteorological forcing data**: Store these files in a directory, such as `data/`.
These files provide necessary input data for the model (e.g., temperature, precipitation).
* **Parameter file**: This file defines model parameters required to run the simulation
(e.g., parameters.mat). Place this file in the appropriate directory (e.g., `data/`).
If there file is not present the code will use the default values based on
[Lute et al. 2022](https://doi.org/10.5194/gmd-15-5045-2022).

Ensure the files are organized as follows:
```
data/
├── parameters.mat         # Model parameters
├── lat_lon_elev.npy       # Latitude, longitude, elevation
├── lrad.npy               # Longwave radiation
├── solar.npy              # Solar radiation
├── tavg.npy               # Average temperature
├── ppt.npy                # Precipitation
└── other forcing files... # Other necessary meteorological data
```

2. Running the Model
To run the model, use the following command in the terminal:

```
python run_main.py
```

This will execute the pySnowClim model, which loads the required forcing data and
parameter file, then runs the snowpack simulation.

For more information about how to run the model:
```
python run_main.py --help
```

3. Main File Structure
The `run_main.py` script serves as the entry point for running the model.
Here's an overview of its structure:

* The script imports the `run_model` function from the `src/runsnowclim_model.py` file.
* It initializes the model by specifying the path to the forcing data directory (`data/`)
and the parameter file (`data/parameters.mat`).
* The model is then run using the run_model function.

4. Modifying the Model
If you wish to modify the paths or input files, adjust the `forcings_path` and
`parameters_path` in the main script.

Example:
```
python run_main.py path_to_forcing_data/  path_to_outputs/
```

5. Output
Once the model has run successfully, the results (such as snow melt, snow water
equivalent, snow depth, and other variables) will be returned by the run_model function.
These results can be further analyzed or saved to files, depending on your needs.
By default the files will be saved inside `/outputs` directory.

6. Troubleshooting
* Ensure that all the required input files are correctly placed in the `data/` directory.
* Make sure the paths provided in the `run_model` function are correct.
* If you encounter missing dependencies, install them using the `pip` or `conda`.
