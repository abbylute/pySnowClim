User Guide
==========

This comprehensive guide covers all aspects of using pySnowClim for snow modeling applications.

.. contents::
   :local:
   :depth: 2

Getting Started
--------------

pySnowClim provides two main ways to run snow simulations:

1. **Python API**: Use the ``run_model`` function directly in Python scripts
2. **Command Line**: Run simulations from the terminal using ``run_main.py``

Both approaches require meteorological forcing data and optionally accept custom parameter configurations.

Input Data Requirements
-----------------------

Meteorological Forcing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

pySnowClim requires time series of meteorological variables. The model accepts data in NetCDF format or legacy MATLAB .mat files.

**Required Variables:**

.. list-table::
   :widths: 15 15 50
   :header-rows: 1

   * - Variable
     - Units
     - Description
   * - ``lrad``
     - kJ/m²/timestep
     - Incoming longwave radiation
   * - ``solar``
     - kJ/m²/timestep
     - Incoming solar radiation
   * - ``tavg``
     - °C
     - Mean air temperature
   * - ``ppt``
     - m/timestep
     - Total precipitation
   * - ``vs``
     - m/s
     - Wind speed at reference height
   * - ``psfc``
     - hPa
     - Surface atmospheric pressure
   * - ``huss``
     - kg/kg
     - Specific humidity
   * - ``relhum``
     - %
     - Relative humidity
   * - ``tdmean``
     - °C
     - Dewpoint temperature

**NetCDF Format (Recommended):**

The forcing data should be organized as a NetCDF file with dimensions:

- ``time``: Temporal dimension
- ``lat``: Latitude dimension
- ``lon``: Longitude dimension

Model Configuration
------------------

Parameter Management
~~~~~~~~~~~~~~~~~~~

pySnowClim uses a comprehensive parameter system that controls model physics and behavior. Parameters can be customized or use scientifically validated defaults.

**Creating Parameter Files:**

.. code-block:: python

   from src.createParameterFile import create_dict_parameters
   import json

   # Create parameters with custom values
   params = create_dict_parameters(
       # Temporal settings
       hours_in_ts=24,              # 24-hour timesteps (daily)

       # Albedo configuration
       albedo_option=2,             # Use Tarboton albedo model
       max_albedo=0.85,            # Maximum fresh snow albedo
       ground_albedo=0.25,         # Bare ground albedo

       # Physical parameters
       snow_dens_default=250,       # Default snow density (kg/m³)
       lw_max=0.1,                 # Maximum liquid water fraction

       # Energy balance
       stability=1,                # Enable atmospheric stability corrections
       windHt=10,                  # Wind measurement height (m)
       tempHt=2,                   # Temperature measurement height (m)

       # Advanced settings
       smooth_time_steps=1,        # Energy smoothing window
       E0_value=1,                 # Windless exchange coefficient
   )

   # Save to JSON file
   with open('my_parameters.json', 'w') as f:
       json.dump(params, f, indent=2, default=str)

**Key Parameter Categories:**

*Temporal Settings:*

- ``hours_in_ts``: Hours per model timestep (1-24)
- ``snowoff_month``, ``snowoff_day``: Annual snow reset date

*Albedo Configuration:*

- ``albedo_option``: 1=Essery, 2=Tarboton, 3=VIC model
- ``max_albedo``: Fresh snow albedo (0.7-0.95)
- ``ground_albedo``: Snow-free surface albedo (0.1-0.4)

*Physical Parameters:*

- ``snow_dens_default``: Initial snow density (150-400 kg/m³)
- ``lw_max``: Maximum liquid water content (0.05-0.15)
- ``snow_emis``: Snow emissivity (0.95-0.99)

*Turbulent Flux Settings:*

- ``stability``: Enable/disable atmospheric stability (0/1)
- ``windHt``, ``tempHt``: Measurement heights (m)
- ``z_0``, ``z_h``: Surface roughness lengths (m)

Running Simulations
------------------

Python API Usage
~~~~~~~~~~~~~~~

The primary interface for programmatic use:

.. code-block:: python

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


Command Line Usage
~~~~~~~~~~~~~~~~~

For operational use and batch processing:

.. code-block:: bash

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

**Command Line Arguments:**

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


Model Outputs
-------------

Output Variables
~~~~~~~~~~~~~~~

pySnowClim generates comprehensive outputs covering snow state, energy fluxes, and surface properties:

**Snow State Variables:**

- ``SnowWaterEq``: Snow water equivalent (mm)
- ``SnowDepth``: Snow depth (mm)
- ``SnowDensity``: Bulk snow density (kg/m³)
- ``SnowfallWaterEq``: New snowfall (mm/timestep)

**Energy and Mass Fluxes:**

- ``SnowMelt``: Snow melt rate (mm/timestep)
- ``Sublimation``: Sublimation rate (mm/timestep)
- ``Condensation``: Vapor condensation (mm/timestep)
- ``RefrozenWater``: Refrozen liquid water (mm/timestep)

**Water Balance:**

- ``Runoff``: Surface runoff (mm/timestep)
- ``PackWater``: Liquid water in snowpack (mm)
- ``RaininSnow``: Rain on snow (mm/timestep)

**Energy Components:**

- ``Energy``: Net energy flux (kJ/m²/timestep)
- ``Q_sensible``: Sensible heat flux (kJ/m²/timestep)
- ``Q_latent``: Latent heat flux (kJ/m²/timestep)
- ``SW_down``, ``SW_up``: Shortwave radiation fluxes
- ``LW_down``, ``LW_up``: Longwave radiation fluxes
- ``Q_precip``: Precipitation heat flux (kJ/m²/timestep)

**Surface Properties:**

- ``Albedo``: Surface albedo (dimensionless)
- ``SnowTemp``: Snow surface temperature (°C)
- ``PackCC``: Snowpack cold content (kJ/m²)

Output Formats
~~~~~~~~~~~~~

**NetCDF Format (Recommended):**

Each variable is saved as a separate NetCDF file with full metadata.

**NumPy Format:**

For programmatic access and analysis.


Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~

1. **Quality Control**: Ensure forcing data has no missing values or unrealistic extremes
2. **Temporal Consistency**: Use consistent timesteps throughout the simulation
3. **Spatial Consistency**: Maintain consistent coordinate systems and projections
4. **Units**: Verify all variables use the expected units (see requirements table)

Parameter Selection
~~~~~~~~~~~~~~~~~

1. **Default Parameters**: Start with defaults, which are calibrated for Western US conditions
2. **Regional Tuning**: Adjust key parameters based on local climate and snow conditions
3. **Sensitivity Testing**: Test model sensitivity to critical parameters
4. **Documentation**: Keep detailed records of parameter choices and justifications

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

1. **Timestep Selection**: Use daily timesteps unless sub-daily processes are critical
2. **Domain Size**: Balance spatial detail with computational requirements
3. **Memory Management**: Monitor memory usage for large domains
4. **Output Management**: Save only needed variables to reduce storage requirements

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Installation Problems:**

- Verify Python version (3.8+) and required packages
- Check file paths and permissions
- Ensure NetCDF libraries are properly installed

**Input Data Issues:**

- Validate forcing data units and ranges
- Check for missing values or unrealistic extremes
- Verify coordinate system consistency

**Parameter Problems:**

- Use ``create_dict_parameters()`` to ensure proper parameter structure
- Check JSON syntax if using custom parameter files
- Validate parameter value ranges

**Memory Errors:**

- Reduce spatial domain size
- Increase available system memory
- Use smaller timestep chunks for processing

**Convergence Issues:**

- Check energy balance components for unrealistic values
- Enable stability corrections for turbulent flux calculations
- Adjust energy smoothing parameters for sub daily simulations

**Output Problems:**

- Ensure output directory exists and is writable
- Check available disk space
- Verify save format specification
