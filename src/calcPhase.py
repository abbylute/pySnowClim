"""
Estimates precipitation phase (rain or snow) based on temperature and relative humidity using a bivariate logistic regression model from Jennings et al., 2018.
"""
import numpy as np

def calc_phase(temp_celsius, rh, binary):
    """
    Calculate the precipitation phase (rain or snow) based on temperature, relative humidity, and precipitation data.
    
    Parameters:
    -----------
    - temp_celsius: Air temperature in C (scalar or array).
    - precip: Precipitation amount (scalar or array).
    - rh: Relative humidity as a percentage (scalar or array).
    - binary: Boolean, whether to classify the phase as binary (1 for snow, 0 for rain).
    
    Returns:
    --------
    - snow: Precipitation classified as snow (in precipitation units). If `binary` is True, returns 1 for snow and 0 for rain.
    """
    # TODO add the below coefficients to the constant file
    # Coefficients from Jennings et al., 2018 (Table 2)
    a = -10.04
    b = 1.41
    g = 0.09

    # Calculate probability of snow
    psnow = 1.0 / (1.0 + np.exp(a + b * temp_celsius + g * rh))

    # TODO break this function to remove the binary variable
    # If binary is True, classify as 0 (rain) or 1 (snow)
    if binary:
        psnow = np.where(psnow >= 0.5, 1, 0)

    precip = np.ones(temp_celsius.shape)
    # Return snow precipitation amount (or binary classification if `binary` is True)
    snow = psnow * precip
    
    return snow.astype(np.float32)
