"""
Calculates sublimation and evaporation processes, updating snow water equivalent (SWE), 
snow depth, snow density, and cold content (cc) based on energy flux (E) and snowpack conditions.
"""
def calc_sublimation(E, SnowWaterEq, SnowDepth, SnowDensity, lastsnowtemp, lastpackcc, 
                     SnowDensDefault, sublimation, condensation, lastpackwater, WaterDens):
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
    - WaterDens: Density of water (kg/m³).

    Returns:
    --------
    - Updated sublimation, condensation, SnowWaterEq, SnowDepth, lastpackcc, SnowDensity, lastpackwater.
    """

    # Latent heat of sublimation (kJ/kg)
    lambdaS = 2834.1 - 0.29 * lastsnowtemp - 0.004 * lastsnowtemp ** 2 

    # Calculate sublimation and evaporation
    Sublimation = np.zeros(len(E))
    Evaporation = np.zeros(len(E))
    Sublimation[lastsnowtemp < 0] = -E[lastsnowtemp < 0] / WaterDens  # Sublimation when snow temp < 0°C
    Evaporation[lastsnowtemp == 0] = -E[lastsnowtemp == 0] / WaterDens  # Evaporation at 0°C

    # Update SWE, snow depth, and snow density
    f = np.where(SnowWaterEq > Sublimation)[0]  # Sublimation occurs, update SWE, snow depth, cc
    f1 = np.where(SnowWaterEq <= Sublimation)[0]  # Complete sublimation, no snow left

    # For non-complete sublimation
    initialSWE = SnowWaterEq[f]
    SnowWaterEq[f] -= Sublimation[f]
    SnowDepth[f] = SnowWaterEq[f] / SnowDensity[f] * WaterDens

    # Complete sublimation
    Sublimation[f1] = SnowWaterEq[f1]
    SnowWaterEq[f1] = 0
    SnowDepth[f1] = 0
    lastpackcc[f1] = 0
    SnowDensity[f1] = SnowDensDefault

    # Output sublimation and condensation
    b = Sublimation > 0
    sublimation[b] += Sublimation[b]
    condensation[~b] += Sublimation[~b]

    # Update packwater by subtracting evaporation
    lastpackwater = np.maximum(0, lastpackwater - Evaporation)

    return sublimation, condensation, SnowWaterEq, SnowDepth, lastpackcc, SnowDensity, lastpackwater
