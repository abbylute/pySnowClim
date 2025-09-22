API Reference
=============

This page provides detailed documentation for the main pySnowClim modules and functions.

Core Model Functions
--------------------

Main Model Runner
~~~~~~~~~~~~~~~~~

.. automodule:: runsnowclim_model
   :members: run_model

The main entry point for running pySnowClim simulations. This module handles loading forcing data,
parameter files, and coordinating model execution.

SnowClim Model Engine
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: snowclim_model
   :members: run_snowclim_model

The core snow model that performs the physics calculations and timestep iterations.

Parameter Configuration
-----------------------

.. automodule:: createParameterFile
   :members: create_dict_parameters

Functions for creating and managing model parameter configurations.

Data Structures
---------------

The model uses several key data structure classes:

Model Variables
~~~~~~~~~~~~~~~

.. autoclass:: SnowModelVariables.SnowModelVariables
   :members:

   Container for all model output variables including snow properties, energy fluxes, and surface conditions.

Snowpack State
~~~~~~~~~~~~~~

.. autoclass:: SnowpackVariables.Snowpack
   :members:

   Manages the internal state of the snowpack including temperature, density, liquid water content, and other physical properties.

Precipitation Properties
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PrecipitationProperties.PrecipitationProperties
   :members:

   Handles precipitation phase partitioning and properties of rainfall and snowfall.
