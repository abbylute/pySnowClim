"""
This script calculates the dew point temperature at a given altitude based on specific humidity
and elevation.
"""

import numpy as np

def calc_dewpoint(shv, Z):
    """
    Calculates the dew point temperature (in Celsius) from specific humidity and elevation.

    Parameters:
    -----------
    - shv: Specific humidity in kg/kg.
    - Z: Elevation in meters.

    Returns:
    --------
    - Td: Dew point temperature in Celsius.
    """

    # CIMIS formula to calculate atmospheric pressure based on elevation
    # based on Eq14 here: https://cimis.water.ca.gov/Content/PDF/PM%20Equation.pdf
    pres = 1013.25 * ((293 - 0.0065 * Z) / 293) ** 5.26

    # Calculate vapor pressure based on specific humidity and pressure
    # see specific humidity equations here: 
    # https://pressbooks-dev.oer.hawaii.edu/atmo/chapter/chapter-4-water-vapor/
    e = shv * pres / 0.622

    # Calculate dew point temperature in Celsius
    # based on: 
    # Bolton, D., 1980: The computation of equivalent potential temperature. Mon. Wea. Rev.,
      # 108, 1046-1053, doi:10.1175/1520-0493(1980)108%3C1046:TCOEPT%3E2.0.CO;2.
    e1 = np.log(e / 6.112)
    Td = 243.5 * e1 / (17.67 - e1)

    return Td
