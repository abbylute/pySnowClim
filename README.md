# pySnowClim

`pySnowClim` is a Python-based model to simulate snowpack evolution using various 
meteorological forcings. This guide will walk you through setting up and running the model.

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
ra = run_model('path_to_forcing_data/', 'path_to_parameters/parameters.mat')
```

5. Output
Once the model has run successfully, the results (such as snow melt, snow water 
equivalent, snow depth, and other variables) will be returned by the run_model function. 
These results can be further analyzed or saved to files, depending on your needs.

6. Troubleshooting
* Ensure that all the required input files are correctly placed in the `data/` directory.
* Make sure the paths provided in the `run_model` function are correct.
* If you encounter missing dependencies, install them using the `pip` or `conda`.
