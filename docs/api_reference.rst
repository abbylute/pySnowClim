API Reference
=============

This page provides detailed documentation for the main pySnowClim modules and functions.

Core Model Functions
-------------------

Main Model Runner
~~~~~~~~~~~~~~~~

.. automodule:: runsnowclim_model
   :members: run_model

The main entry point for running pySnowClim simulations. This module handles loading forcing data,
parameter files, and coordinating model execution.

Snow Climate Model Engine
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: snowclim_model
   :members: run_snowclim_model

The core snow climate model that performs the physics calculations and timestep iterations.

Parameter Configuration
----------------------

.. automodule:: createParameterFile
   :members: create_dict_parameters

Functions for creating and managing model parameter configurations.

Data Structures
---------------

The model uses several key data structure classes:

Model Variables
~~~~~~~~~~~~~~

.. autoclass:: SnowModelVariables.SnowModelVariables
   :members:

   Container for all model output variables including snow properties, energy fluxes, and surface conditions.

Snowpack State
~~~~~~~~~~~~~

.. autoclass:: SnowpackVariables.Snowpack
   :members:

   Manages the internal state of the snowpack including temperature, density, liquid water content, and other physical properties.

Precipitation Properties
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PrecipitationProperties.PrecipitationProperties
   :members:

   Handles precipitation phase partitioning and properties of rainfall and snowfall.

Constants and Configuration
---------------------------

.. automodule:: constants
   :members:

   Physical constants used throughout the model calculations.

Command Line Interface
---------------------

The model can be run from the command line using the main script:

.. code-block:: bash

   python run_main.py [forcings_path] [output_path] [parameters_path] [save_format]

Arguments:
  - ``forcings_path``: Path to forcing data file or directory (default: 'data/')
  - ``output_path``: Directory to save model outputs (default: 'outputs/')
  - ``parameters_path``: Path to parameters JSON file (default: None, uses defaults)
  - ``save_format``: Output format '.nc' or '.npy' (default: None, uses .npy)

