Model Description
=================

This section provides an overview of the physical processes and model structure implemented in pySnowClim. For detailed scientific background and validation, please refer to the original SnowClim publications.

.. contents::
   :local:
   :depth: 2

Model Overview
--------------

pySnowClim is a physically-based energy balance snow model that simulates snow accumulation, metamorphism, and ablation processes. The model is implemented as a Python translation of the original SnowClim model, maintaining the same core physics while providing improved computational efficiency and modern data handling capabilities.

**Core Model Philosophy:**

The model operates on the principle that snow evolution is fundamentally controlled by energy exchanges at the snow surface. All physical processes (melting, sublimation, refreezing, densification) are driven by the surface energy balance, making the model suitable for a wide range of climatic conditions and applications.

**Model Structure:**

pySnowClim consists of three main components:

1. **Model Engine** (``snowclim_model.py``): Core physics calculations and timestep processing
2. **Model Runner** (``runsnowclim_model.py``): Data management, I/O operations, and model coordination
3. **Command Interface** (``run_main.py``): Command-line interface for operational use

Fundamental Physics
------------------

Energy Balance Framework
~~~~~~~~~~~~~~~~~~~~~~~

The model is built around the surface energy balance equation:

.. math::

   Q_{net} = SW_{in} - SW_{out} + LW_{in} - LW_{out} + H + LE + G + Q_p

Where:
- :math:`SW` = shortwave radiation fluxes
- :math:`LW` = longwave radiation fluxes
- :math:`H` = sensible heat flux
- :math:`LE` = latent heat flux
- :math:`G` = ground heat flux
- :math:`Q_p` = precipitation heat flux

The net energy (:math:`Q_{net}`) is then distributed among competing physical processes in order of priority.

Mass Balance
~~~~~~~~~~~

Snow mass changes are governed by:

.. math::

   \frac{dSWE}{dt} = Snowfall - Melt - Sublimation + Refreezing

Where all terms are in water equivalent units (mm/timestep).

Model Implementation
-------------------

Core Model Engine
~~~~~~~~~~~~~~~~

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

- **Cold Content**: Energy to warm snowpack to 0°C (first priority)
- **Refreezing**: Energy to refreeze liquid water (second priority)
- **Melting**: Energy to melt snow/ice (third priority)
- **Sublimation**: Mass exchange with atmosphere

**4. State Updates**

Physical properties are updated:

- Snow density evolution through compaction
- Liquid water content and drainage
- Surface albedo changes
- Internal snowpack temperature


Key Model Features
-----------------

Albedo Parameterizations
~~~~~~~~~~~~~~~~~~~~~~~

Three albedo options accommodate different applications:

- **Option 1**: Essery et al. (2013) 
- **Option 2**: Tarboton (Utah) 
- **Option 3**: VIC model 

Snow Density Evolution
~~~~~~~~~~~~~~~~~~~~~

Realistic density changes through:

- Fresh snow density as function of temperature
- Compaction based on overburden pressure and temperature
- Density updates after new snowfall events

Advanced Energy Balance
~~~~~~~~~~~~~~~~~~~~~~

Sophisticated energy processing includes:

- Atmospheric stability corrections for turbulent fluxes
- Energy smoothing to reduce numerical instabilities
- "Cold content tax" system for improved convergence
- Multiple measurement height corrections

Liquid Water Processes
~~~~~~~~~~~~~~~~~~~~~

Comprehensive liquid water handling:

- Rain-on-snow events
- Internal liquid water storage and drainage
- Refreezing when snowpack has cold content
- Runoff generation with realistic drainage rates

Model Validation and Applications
--------------------------------

Scientific Basis
~~~~~~~~~~~~~~~

pySnowClim physics are based on established snow science literature:

- Energy balance formulations from classical snow physics texts
- Parameterizations from peer-reviewed publications
- Validation against field observations and other snow models
- Calibration using SNOTEL network data across western United States

Appropriate Applications
~~~~~~~~~~~~~~~~~~~~~~~

The model is suitable for:

- **Research**: Detailed energy balance studies and process investigations
- **Operations**: Water resource forecasting and management
- **Education**: Teaching snow physics and energy balance concepts
- **Climate Studies**: Long-term snow evolution under changing conditions
- **Spatial Scales**: Point locations to continental domains
- **Temporal Scales**: Sub-daily to multi-decadal simulations
- **Environments**: All snow climates from maritime to continental


Model Limitations
----------------

**Physical Limitations**:

- Single-layer snowpack (no internal temperature gradients)
- No explicit snow grain evolution or metamorphism
- Simplified treatment of snow-vegetation interactions
- Ground heat flux assumed constant

**Technical Limitations**:

- Requires complete meteorological forcing datasets
- No built-in downscaling or gap-filling capabilities
- Output storage can be large for long simulations over large domains


For comprehensive scientific documentation, algorithm details, and validation results, please refer to the original SnowClim publications.
