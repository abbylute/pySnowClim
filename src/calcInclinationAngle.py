"""
Calculates the solar zenith angle as a function of latitude and time of year, assuming time is solar noon.
"""

import numpy as np
from datetime import datetime

def calc_inclination_angle(lat, month, day=15):
    """
    Calculate the solar zenith angle based on latitude and date assuming time is solar noon.

    Parameters:
    -----------
    - lat: Latitude in degrees (array or scalar).
    - month: Month of the year (1 to 12).
    - day: Day of the month (default is 15).

    Returns:
    --------
    - incangle: Solar zenith angle in degrees.
    """

    # Calculate day of the year (doy)
    if month == 2 and day == 29:
        day = 28

    date = datetime(2013, month, day)
    doy = (date - datetime(2013, 1, 1)).days + 1  # days since start of year

    # Declination angle calculation
    decl_angle = 23.45 * np.sin(2 * np.pi * (284 + doy) / 365)

    # Solar inclination angle
    incangle = 90 - np.abs(lat - decl_angle)

    # Negative values are reset to 0° since this indicates nighttime/no sun
    incangle = np.maximum(0, incangle)

    return incangle
