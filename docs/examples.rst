Examples
========

This section demonstrates how to run the provided pySnowClim example and what results to expect.

Running the Example Script
--------------------------

Overview
~~~~~~~~

The `examples/run_snowclim_example.py` script provides a complete workflow that:

1. Loads meteorological forcing data and observations
2. Runs the pySnowClim model with default parameters
3. Compares model output with observed snow water equivalent
4. Generates validation plots and performance statistics

Required Data Files
~~~~~~~~~~~~~~~~~~~

Before running the example, ensure these files are in the `examples/` directory:

- ``forcings_example.nc`` - Meteorological forcing data
- ``target_example.nc`` - Observed snow water equivalent for validation

The forcing file contains the following variables:

.. list-table::
   :widths: 20 20 60
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
     - Wind speed
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

The observation file contains:

- ``swe`` - Observed snow water equivalent (mm)
- ``time`` - Time coordinate matching the forcing data period
- Optional metadata: station name, elevation, latitude, longitude

Running the Example
~~~~~~~~~~~~~~~~~~~

**Command Line Execution:**

.. code-block:: bash

   cd examples
   python run_snowclim_example.py


**Expected Console Output:**

.. code-block:: text

   ============================================================
   pySnowClim Example Run
   ============================================================
   Loading forcing data...
   Loading observation data...
   Forcing data time range: 2001-10-01T00:00:00 to 2002-09-30T00:00:00
   Observation data time range: 2001-10-01T00:00:00 to 2002-09-30T00:00:00
   Number of forcing time steps: 365
   Number of observation time steps: 365
   Running pySnowClim model...
   Loading necessary files...
   Parameter file undefined, using default parameters
   Files loaded, running the model...
   Model run completed!

   Comparing model results with observations...

   Comparison Statistics:
   Correlation: 0.876
   RMSE: 89.3 mm
   MAE: 67.2 mm
   Bias: -12.4 mm

   Validation plot saved to: examples/output/model_validation.png
   Validation statistics saved to: examples/output/validation_stats.txt

   Example run completed successfully!
   Results saved in: examples/output

Expected Results
----------------

Output Files
~~~~~~~~~~~~

The example generates the following files in `examples/output/`:

**Model Output Files (NetCDF format):**

- ``SnowWaterEq.nc`` - Snow water equivalent time series
- ``SnowDepth.nc`` - Snow depth time series
- ``SnowMelt.nc`` - Daily snowmelt amounts
- Additional variables (sublimation, condensation, runoff, etc.)

**Validation Products:**

- ``model_validation.png`` - Four-panel validation plot
- ``validation_stats.txt`` - Performance statistics summary

The example serves as both a validation tool and a template for setting up your own pySnowClim applications.
