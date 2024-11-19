"""
Snowpack Model Initialization Script

This script defines a `Snowpack` class used to manage the initialization and tracking of 
snowpack-related variables for climate or environmental models. The script provides two methods 
for initializing the snowpack variables:
    1. `initialize_snowpack_base()`: Initializes all variables except `lastpacktemp` and `snowage`.
    2. `initialize_full_snowpack()`: Initializes all variables, including `lastpacktemp` and `snowage`.

Attributes:
-----------
- lastalbedo: np.ndarray
    Albedo values initialized based on the ground albedo parameter.
- lastswe: np.ndarray
    Last snow water equivalent initialized to zeros.
- lastsnowdepth: np.ndarray
    Last snow depth initialized to zeros.
- packsnowdensity: np.ndarray
    Snow density initialized based on the default parameter.
- lastpackcc: np.ndarray
    Last pack cold content initialized to zeros.
- lastpackwater: np.ndarray
    Last pack water content initialized to zeros.
- lastpacktemp: np.ndarray
    Last pack temperature initialized to zeros (only with full initialization).
- snowage: np.ndarray
    Snow age initialized to zeros (only with full initialization).

Usage:
------
To initialize all variables including `lastpacktemp` and `snowage`:
>>> snowpack = Snowpack(n_lat, parameters)
>>> snowpack.initialize_full_snowpack(forcings_data)

To initialize everything except `lastpacktemp` and `snowage`:
>>> snowpack.initialize_snowpack_base(forcings_data)

"""

import numpy as np

class Snowpack:
    def __init__(self, n_lat, parameters):
        """
        Class to handle initialization and management of snowpack-related variables.

        Args:
        -----
        - n_lat: int
            Number of latitude points.
        - parameters: dict
            A dictionary of model parameters, including 'ground_albedo' and 'snow_dens_default'.
        """
        self.n_lat = n_lat
        self.ground_albedo = parameters['ground_albedo']
        self.snow_dens_default = parameters['snow_dens_default']


        # Initialize core snowpack variables (all except lastpacktemp and snowage)
        self.initialize_full_snowpack()


    def initialize_snowpack_base(self):
        """Initialize all core snowpack-related variables except 'lastpacktemp' and 'snowage'."""
        self.lastalbedo = np.ones(self.n_lat, dtype=np.float32) * self.ground_albedo
        self.lastswe = np.zeros(self.n_lat, dtype=np.float32)  # Snow Water Equivalent
        self.lastsnowdepth = np.zeros(self.n_lat, dtype=np.float32)
        self.packsnowdensity = np.ones(self.n_lat, dtype=np.float32) * self.snow_dens_default
        self.lastpackcc = np.zeros(self.n_lat, dtype=np.float32)  # Cold content
        self.lastpackwater = np.zeros(self.n_lat, dtype=np.float32)  # Pack water content
        self.rain_in_snow = np.full(self.n_lat, np.nan, dtype=np.float32) 


    def initialize_full_snowpack(self):
        """
        Initialize all snowpack variables, including 'lastpacktemp' and 'snowage'.

        Args:
        -----
        - forcings_data: dict
            Dictionary containing latitude data for initializing additional variables.
        """
        # Initialize base variables
        self.initialize_snowpack_base()

        # Initialize lastpacktemp and snowage
        self.lastpacktemp = np.zeros(self.n_lat, dtype=np.float32)
        self.snowage = np.zeros(self.n_lat, dtype=np.float32)

