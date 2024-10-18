"""
This script distributes excess energy in the snowpack to allow refreezing of rain and meltwater, updates the snow water equivalent (SWE), snowpack cold content (CC), and snow density.
"""

import numpy as np

def calc_energy_to_refreezing(lastpackwater, lastswe, lastpackcc, lastsnowdepth, 
                              WaterDens, LatHeatFreez, RefrozenWater, packsnowdensity):
    """
    Distributes excess energy to refreeze rain and meltwater in the snowpack, updates SWE, cold content, and snow density.

    Parameters:
    -----------
    - lastpackwater: Current water content in the snowpack (array).
    - lastswe: Snow water equivalent (array).
    - lastpackcc: Snowpack cold content (array).
    - lastsnowdepth: Current snow depth (array).
    - WaterDens: Density of water (constant).
    - LatHeatFreez: Latent heat of fusion (constant).
    - RefrozenWater: Amount of refrozen water (array).
    - packsnowdensity: Snowpack density (array).

    Returns:
    --------
    - lastpackwater: Updated water content in the snowpack (array).
    - lastswe: Updated snow water equivalent (array).
    - lastpackcc: Updated cold content (array).
    - packsnowdensity: Updated snow density (array).
    - RefrozenWater: Updated amount of refrozen water (array).
    """
    
    # Condition for refreezing: available water, snow, cold content, and density < 550 kg/m^3
    b = (lastpackwater > 0) & (lastswe > 0) & (lastpackcc < 0) & (packsnowdensity < 550)
    
    # Potential energy from refreezing
    Prf = np.zeros_like(lastpackwater)
    Prf[b] = WaterDens * LatHeatFreez * lastpackwater[b]

    # 1. If cold content exceeds refreezing potential energy, freeze all water and update cold content
    bb = b & (-lastpackcc >= Prf)
    lastswe[bb] += lastpackwater[bb]
    RefrozenWater[:, bb] = lastpackwater[bb]
    lastpackwater[bb] = 0
    lastpackcc[bb] += Prf[bb]
    Prf[bb] = 0

    # 2. If cold content is insufficient for full refreezing, freeze what is possible
    bb = (lastpackwater > 0) & (lastswe > 0) & (lastpackcc < 0) & (-lastpackcc < Prf) & (packsnowdensity < 550)
    RefrozenWater[:, bb] = -lastpackcc[bb] / (WaterDens * LatHeatFreez)
    lastswe[bb] += RefrozenWater[:, bb]
    lastpackwater[bb] -= RefrozenWater[:, bb]
    lastpackcc[bb] = 0

    # Update snow density, assuming snow depth remains unchanged
    packsnowdensity = lastswe * WaterDens / lastsnowdepth

    return lastpackwater, lastswe, lastpackcc, packsnowdensity, RefrozenWater
