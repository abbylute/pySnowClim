Quick Start Guide
=================

This guide will help you run your first pySnowClim simulation.

Basic Usage
-----------

1. Prepare Your Data
~~~~~~~~~~~~~~~~~~~

pySnowClim requires meteorological forcing data. Create a data directory structure:

.. code-block:: text

   data/
   ├── forcing_data.nc        # NetCDF with forcing variables
   └── parameters.json        # Model parameters (optional)

The forcing data should contain these variables:

* ``lrad`` - Longwave radiation (kJ/m²/timestep)
* ``solar`` - Solar radiation (kJ/m²/timestep)
* ``tavg`` - Air temperature (°C)
* ``ppt`` - Precipitation (m/timestep)
* ``vs`` - Wind speed (m/s)
* ``psfc`` - Surface pressure (hPa)
* ``huss`` - Specific humidity (kg/kg)
* ``relhum`` - Relative humidity (%)
* ``tdmean`` - Dewpoint temperature (°C)

2. Run the Model
~~~~~~~~~~~~~~~

.. code-block:: python

   from src.runsnowclim_model import run_model

   # Run with default parameters
   results = run_model(
       forcings_path='data/forcing_data.nc',
       parameters_path=None,  # Use defaults
       outputs_path='outputs/',
       save_format='.nc'
   )

3. Command Line Usage
~~~~~~~~~~~~~~~~~~~~

You can also run pySnowClim from the command line:

.. code-block:: bash

   python run_main.py data/forcing_data.nc outputs/ parameters.json .nc


Understanding the Output
-----------------------

The model will generate several output files in NetCDF format:

* ``SnowWaterEq.nc`` - Snow water equivalent (mm)
* ``SnowDepth.nc`` - Snow depth (mm)
* ``SnowMelt.nc`` - Snow melt (mm/timestep)
* ``Albedo.nc`` - Surface albedo (dimensionless)
* ``Energy.nc`` - Net energy flux (kJ/m²/timestep)
* And many more...

See :doc:`output_variables` for a complete description of all output variables.

Next Steps
----------

* Read the :doc:`user_guide` for detailed information about model configuration
* Explore :doc:`examples` for more advanced use cases
* Check the :doc:`api_reference` for complete function documentation
* Learn about the model physics in :doc:`model_description`
