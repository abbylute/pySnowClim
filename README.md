# pySnowClim

`pySnowClim`  is a robust, efficient, and open-source Python implementation of
the original SnowClim model written in MATLAB.
`pySnowClim` addresses the practical need for a snow models that can accurately
simulate snow accumulation and melt across both large regions and fine spatial detail.
`pySnowClim` combines key physical processes with efficient algorithms to
provide high-resolution snowpack estimates, especially useful in complex terrain.
Many current snow models are either too simple, too computationally costly,
or not easy to use with modern workflows.

The target audience of `pySnowClim` includes hydrologists, climatologists,
ecologists, water resource managers,
and students who need reliable snow modeling capabilities for applications such as:

- Research: Detailed energy balance studies and process investigations
- Operations: Water resource forecasting and management
- Education: Teaching snow physics and energy balance concepts
- Climate Studies: Long-term snow evolution under changing conditions
- Adaptation Planning: anticipating and planning adaptations to impacts of warming on snow-dependent species, ecosystems, and communities

More information about the package and the model can be found in the [documentation page](https://abbylute.github.io/pySnowClim/index.html) and [Lute et al., 2022](https://doi.org/10.5194/gmd-15-5045-2022).

![Snow model conceptual diagram.](docs/snowclim.png)

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- See the [requirements.txt](requirements.txt) file to check the required packages.

### Option 1: Install from source (Recommended for development)

1.Clone the repository:
```bash
git clone https://github.com/abbylute/pySnowClim.git
cd pySnowClim
```

2.Create a virtual environment (recommended):
```bash
python -m venv pysnowclim-env
source pysnowclim-env/bin/activate  # On Windows: pysnowclim-env\Scripts\activate
```
or with conda:
```bash
# Create environment with Python
conda create -n pysnowclim python=3.9
conda activate pysnowclim
```

3.Install the package and dependencies:
This installs the package in "editable" mode, so changes to the source code are immediately available.
```bash
pip install -e .
```
To verify that `pySnowClim` is installed correctly:
```bash
python verify_installation.py
```

### Option 2: Install for regular use
```bash
pip install git+https://github.com/abbylute/pySnowClim.git
```
To verify that `pySnowClim` is installed correctly:
```bash
python -c "from createParameterFile import create_dict_parameters;print(create_dict_parameters())"
```

## Running the Model

`pySnowClim` provides two main ways to run snow simulations:

1. Python API: Use the run_model function directly in Python scripts.
2. Command Line: Run simulations from the terminal using run_main.py.

Both approaches require meteorological forcing data and optionally accept custom parameter configurations.

### Input Data Requirements
`pySnowClim` requires time series of meteorological variables. The model accepts data in NetCDF format (Recommended) or legacy MATLAB `.mat` files.

Required Variables:
- **lrad** (kJ/m²/timestep):	Incoming longwave radiation
- **solar** (kJ/m²/timestep):	Incoming solar radiation
- **tavg** (°C):	Mean air temperature
- **ppt** (m/timestep):	Total precipitation
- **vs** (m/s):	Wind speed at reference height
- **psfc** (hPa):	Surface atmospheric pressure
- **huss** (kg/kg):	Specific humidity
- **relhum** (%):	Relative humidity
- **tdmean** (°C):	Dewpoint temperature

The forcing data should be organized as a NetCDF file with dimensions:
- **time**: Temporal dimension
- **lat**: Latitude dimension
- **lon**: Longitude dimension


### Python API Usage

The primary interface for programmatic use:

```python
from src.runsnowclim_model import run_model

# Basic simulation
results = run_model(
    forcings_path='forcing_data.nc',      # Input forcing data
    parameters_path='parameters.json',     # Custom parameters (optional)
    outputs_path='simulation_results/',    # Output directory
    save_format='.nc'                     # Save as NetCDF
)

# Results is a list of SnowModelVariables objects, one per timestep
print(f"Simulation completed: {len(results)} timesteps")
```

### Command Line Usage

For operational use and batch processing:

```bash
# Basic usage - uses all defaults
python run_main.py

# Specify input forcing file
python run_main.py forcing_data.nc

# Specify input and output directories
python run_main.py forcing_data.nc results/

# Full specification with custom parameters
python run_main.py forcing_data.nc results/ custom_params.json .nc

# Run with MATLAB-format inputs (legacy)
python run_main.py data/ results/ parameters.json .npy
```

#### Command Line Arguments:

1. ``forcings_path`` (optional): Path to forcing data

   - Default: ``'data/'``
   - Can be NetCDF file or directory with .mat files

2. ``output_path`` (optional): Output directory

   - Default: ``'outputs/'``
   - Directory will be created if it doesn't exist

3. ``parameters_path`` (optional): JSON parameter file

   - Default: ``None`` (uses model defaults)
   - Must be valid JSON format

4. ``save_format`` (optional): Output file format

   - Default: ``None`` (saves as .npy files)
   - Use ``'.nc'`` for NetCDF output

For more information about how to run the model:
```python
python run_main.py --help
```

## Examples

The [run_snowclim_example.py](examples/run_snowclim_example.py)` script provides a complete workflow that:

1. Loads meteorological forcing data and observations
2. Runs the `pySnowClim` model with default parameters
3. Compares model output with observed snow water equivalent
4. Generates validation plots and performance statistics

**Command Line Execution:**
```bash
cd examples
python run_snowclim_example.py
```
## Troubleshooting
### Installation Problems

- Verify Python version (3.8+) and required packages
- Check file paths and permissions
- Ensure NetCDF libraries are properly installed

### Input Data Issues

- Validate forcing data units and ranges
- Check for missing values or unrealistic extremes
- Verify coordinate system consistency

### Parameter Problems

- Use ``create_dict_parameters()`` to ensure proper parameter structure
- Check JSON syntax if using custom parameter files
- Validate parameter value ranges

### Memory Errors

- Reduce spatial domain size
- Increase available system memory

### Convergence Issues

- Check energy balance components for unrealistic values
- Enable stability corrections for turbulent flux calculations
- Adjust energy smoothing parameters for sub daily simulations

### Output Problems

- Ensure output directory exists and is writable
- Check available disk space
- Verify save format specification
