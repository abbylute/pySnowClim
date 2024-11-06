"""
Calculates sublimation and evaporation processes, updating snow water equivalent (SWE), 
snow depth, snow density, and cold content (cc) based on energy flux (E) and snowpack conditions.
"""
import numpy as np
import constants as const

def calc_sublimation(E, SnowWaterEq, SnowDepth, SnowDensity, lastsnowtemp, lastpackcc, 
                     SnowDensDefault, sublimation, condensation, lastpackwater):
    """
    Update snowpack properties by calculating sublimation and evaporation.

    Parameters:
    -----------
    - E: Energy flux (kg/m²/s).
    - SnowWaterEq: Snow water equivalent (mm or m³ of water).
    - SnowDepth: Depth of snow (m).
    - SnowDensity: Density of snow (kg/m³).
    - lastsnowtemp: Temperature of the snow (°C).
    - lastpackcc: Cold content of the snowpack (kJ/kg).
    - SnowDensDefault: Default snow density for new snow (kg/m³).
    - sublimation: Current sublimation value (kg/m²).
    - condensation: Current condensation value (kg/m²).
    - lastpackwater: Amount of water in the snowpack (kg/m²).

    Returns:
    --------
    - Updated sublimation, condensation, SnowWaterEq, SnowDepth, lastpackcc, SnowDensity, lastpackwater.
    """
    # Calculate sublimation and evaporation
    Sublimation = np.zeros(E.size)
    Evaporation = np.zeros(E.size)
    Sublimation[lastsnowtemp < 0] = -E[lastsnowtemp < 0] / const.WATERDENS  # Sublimation when snow temp < 0°C
    Evaporation[lastsnowtemp == 0] = -E[lastsnowtemp == 0] / const.WATERDENS  # Evaporation at 0°C

    initialSWE = SnowWaterEq
    # Update SWE, snow depth, and snow density
    has_sublimation = SnowWaterEq > Sublimation  # Sublimation occurs, update SWE, snow depth, cc
    no_snow_left = SnowWaterEq <= Sublimation  # Complete sublimation, no snow left
    # f = np.where(SnowWaterEq > Sublimation)[0]  # Sublimation occurs, update SWE, snow depth, cc
    # f1 = np.where(SnowWaterEq <= Sublimation)[0]  # Complete sublimation, no snow left

    # For non-complete sublimation
    SnowWaterEq[has_sublimation] -= Sublimation[has_sublimation]
    SnowDepth[has_sublimation] = SnowWaterEq[has_sublimation] / SnowDensity[has_sublimation] * const.WATERDENS
    
    cc_sublimation = has_sublimation & (Sublimation>0)  #only update cc for sublimation, not condensation
    lastpackcc[cc_sublimation] *= SnowWaterEq[cc_sublimation]/initialSWE[cc_sublimation]

    # Complete sublimation
    Sublimation[no_snow_left] = SnowWaterEq[no_snow_left]
    SnowWaterEq[no_snow_left] = 0
    SnowDepth[no_snow_left] = 0
    lastpackcc[no_snow_left] = 0
    SnowDensity[no_snow_left] = SnowDensDefault

    # Output sublimation and condensation
    b = Sublimation > 0
    sublimation[b] = Sublimation[b]
    condensation[~b] = Sublimation[~b]

    # Update packwater by subtracting evaporation
    lastpackwater = np.maximum(0, lastpackwater - Evaporation)

    return sublimation, condensation, SnowWaterEq, SnowDepth, lastpackcc, SnowDensity, lastpackwater
