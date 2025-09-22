Model Description
=================

This section provides an overview of the physical processes and model structure implemented in pySnowClim.

Model Overview
--------------

pySnowClim is a physically-based energy balance snow model that simulates snow accumulation and melt.
The model is implemented as a Python translation of the original SnowClim model written in MATLAB
`(Lute et al., 2022) <https://doi.org/10.5194/gmd-15-5045-2022>`_ ,
maintaining the same core physics while providing improved
computational efficiency and modern data handling capabilities.

**Core Model Philosophy:**

The model operates on the principle that snow evolution is fundamentally controlled by energy exchanges at the snow surface.
All physical processes (melting, sublimation, refreezing, densification) are driven by the surface energy balance,
making the model suitable for a wide range of climatic conditions and applications.

**Model Structure:**

pySnowClim consists of three main components:

1. **Model Engine** (``snowclim_model.py``): Core physics calculations and timestep processing
2. **Model Runner** (``runsnowclim_model.py``): Data management, I/O operations, and model coordination
3. **Command Interface** (``run_main.py``): Command-line interface for operational use

Fundamental Physics
-------------------

Energy Balance Framework
~~~~~~~~~~~~~~~~~~~~~~~~

The model is built around the surface energy balance equation:

.. math::

   Q_{net} = SW_{in} - SW_{out} + LW_{in} - LW_{out} + H + E_{i} + E_{w} + G + P

Where:

- :math:`SW` = shortwave radiation fluxes
- :math:`LW` = longwave radiation fluxes
- :math:`H` = sensible heat flux
- :math:`E` = latent heat flux of ice (i) and water (w)
- :math:`G` = ground heat flux
- :math:`P` = advected heat flux from precipitation


Mass Balance
~~~~~~~~~~~~

Mass balance of the solid :math:`(M_{s})` and liquid :math:`(M_{l})` portions of the snowpack are governed by:

.. math::

   M_{s} = M_{snow} + M_{ref} - M_{melt} + M_{dep} - M_{sub}

.. math::

   M_{l} = M_{rain} - M_{ref} + M_{melt} - M_{runoff} + M_{cond} - M_{evap}

Where:

- :math:`M_{snow}` = mass of new snowfall
- :math:`M_{ref}` = mass of the snowpack liquid water that has been refrozen
- :math:`M_{melt}` = mass of snow that has melted
- :math:`M_{dep}` = mass of deposition
- :math:`M_{sub}` = mass of sublimation
- :math:`M_{rain}` = mass of rain added to the snowpack
- :math:`M_{runoff}` = mass of liquid water that has left the snowpack as runoff
- :math:`M_{cond}` = mass of condensation
- :math:`M_{evap}` = mass of evaporation

Model Implementation
--------------------

Core Model Engine
~~~~~~~~~~~~~~~~~

**File**: ``snowclim_model.py``

The main model engine implements the physics through the ``run_snowclim_model()`` function, which processes each timestep through several phases:

**1. Precipitation Processing**

Determines rain/snow partitioning using temperature and humidity:

- Logistic regression model separates rain from snow
- Fresh snow density calculated from air temperature
- New snow temperature and cold content computed

**2. Energy Balance Calculation**

Computes all surface energy flux components:

- Shortwave radiation (accounting for albedo)
- Longwave radiation (Stefan-Boltzmann law)
- Turbulent fluxes (bulk aerodynamic formulas)
- Ground heat flux (constant value)
- Precipitation heat flux

**3. Energy Distribution**

Available energy is allocated sequentially:

- **Cold Content**: Energy to warm snowpack to 0°C
- **Refreezing**: Energy to refreeze liquid water
- **Melting**: Energy to melt snow/ice
- **Sublimation**: Mass exchange with atmosphere

**4. State Updates**

Physical properties are updated:

- Snow density evolution through compaction
- Liquid water content and drainage
- Surface albedo changes
- Internal snowpack temperature


Key Model Features
------------------

Albedo Parameterizations
~~~~~~~~~~~~~~~~~~~~~~~~

Three albedo options accommodate different applications:

- **Option 1**: Essery et al. (2013)
- **Option 2**: Utah Energy Balance (UEB)
- **Option 3**: VIC model

Snow Density Evolution
~~~~~~~~~~~~~~~~~~~~~~

Realistic density changes through:

- Fresh snow density as function of temperature
- Compaction based on overburden pressure and temperature
- Density updates after new snowfall events

Advanced Energy Balance
~~~~~~~~~~~~~~~~~~~~~~~

Energy processing includes:

- Atmospheric stability corrections for turbulent fluxes
- Energy smoothing to reduce numerical instabilities
- "Cold content tax" system for improved convergence
- Multiple measurement height corrections

Liquid Water Processes
~~~~~~~~~~~~~~~~~~~~~~

Comprehensive liquid water handling:

- Rain-on-snow events
- Internal liquid water storage and drainage
- Refreezing when snowpack has cold content
- Runoff generation with realistic drainage rates

Model Validation and Applications
---------------------------------

Scientific Basis
~~~~~~~~~~~~~~~~

pySnowClim physics are based on established snow science literature:

- Energy balance formulations from classical snow physics texts
- Parameterizations from peer-reviewed publications
- Validation against field observations and other snow models
- Calibration using SNOTEL network data across western United States

Appropriate Applications
~~~~~~~~~~~~~~~~~~~~~~~~

The model is suitable for:

- **Research**: Detailed energy balance studies and process investigations
- **Operations**: Water resource forecasting and management
- **Education**: Teaching snow physics and energy balance concepts
- **Climate Studies**: Long-term snow evolution under changing conditions
- **Spatial Scales**: Point locations to continental domains
- **Temporal Scales**: Sub-daily to multi-decadal simulations
- **Environments**: All snow climates from maritime to continental


Model Limitations
-----------------

**Physical Limitations**:

- Single-layer snowpack with separate surface and pack temperatures (but no internal temperature gradients)
- No explicit snow grain evolution
- Vegetation not included
- Ground heat flux assumed constant
- No snow redistribution via gravity or wind


For comprehensive scientific background, algorithm details, and validation results,
please refer to the original SnowClim publication `(Lute et al., 2022) <https://doi.org/10.5194/gmd-15-5045-2022>`_.
