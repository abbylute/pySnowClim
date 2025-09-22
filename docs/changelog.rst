Changelog
=========

All notable changes to pySnowClim will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[0.1.0] - 2024-XX-XX
--------------------

Initial Release
~~~~~~~~~~~~~~

**Added**

*Core Model Features:*

- Complete Python implementation of the SnowClim snow model
- Energy balance snow modeling with comprehensive physics
- Support for gridded and point-scale simulations
- Three albedo parameterization options (Essery, Tarboton, VIC)
- Atmospheric stability corrections for turbulent fluxes
- Comprehensive snow density evolution modeling
- Liquid water content tracking and refreezing processes
- Cold content and energy distribution algorithms

*Input/Output Capabilities:*

- NetCDF input format support for forcing data
- Legacy MATLAB .mat file support for backward compatibility
- NetCDF output format with full metadata
- NumPy binary output format for analysis
- Flexible parameter configuration system via JSON files
- Command-line interface for operational use

*Physical Processes:*

- Precipitation phase partitioning (rain/snow) using logistic regression
- Fresh snow density calculation as function of temperature
- Snow compaction following Anderson (1976) parameterization
- Multiple albedo aging models for different applications
- Sublimation and evaporation processes with proper thermodynamics
- Rain-on-snow processes and liquid water drainage
- Ground heat flux and energy balance closure

*Data Structures:*

- ``SnowModelVariables`` class for comprehensive output management
- ``SnowpackVariables`` class for internal snowpack state tracking
- ``PrecipitationProperties`` class for precipitation processing
- Efficient vectorized operations for multi-point simulations

*Utilities and Tools:*

- Parameter file creation utilities with scientifically-validated defaults
- Time series to daily mean conversion functions
- Model comparison and validation tools
- Comprehensive test suite for quality assurance

*Documentation:*

- Complete Sphinx documentation with installation, user guide, and API reference
- Detailed model description with physics background
- Output variable descriptions with units and typical ranges
- Contributing guidelines for community development
- Example workflows and use cases

**Technical Specifications**

*System Requirements:*

- Python 3.8+ compatibility
- Core dependencies: numpy, xarray, scipy, netCDF4, pandas
- Cross-platform support (Windows, macOS, Linux)

*Scientific Validation:*

- Physics based on peer-reviewed snow science literature
- Parameter defaults calibrated against western US SNOTEL network
- Energy and mass balance conservation enforced

**Configuration Options**

*Model Physics:*

- Multiple albedo parameterizations for different research applications
- Configurable turbulent flux formulations with stability corrections
- Adjustable snow density parameters and compaction rates

*Operational Settings:*

- Flexible timestep configuration (1-24 hours)
- Adjustable output variables and formats
- Vectorized calculations for computational efficiency
- Configurable simulation periods and restart capabilities

**Quality Assurance**

*Testing Framework:*

- Unit tests for individual physics functions
- Integration tests for complete model workflows

*Technical Constraints:*

- Requires complete meteorological forcing datasets
- No built-in spatial downscaling or gap-filling capabilities
- Memory usage scales with domain size and output retention
- Limited parallel processing optimization in initial release
- Output storage can be large for long simulations over large domains


**Migration Notes**

*From Original MATLAB SnowClim:*

- Identical physics implementation ensures result consistency
- Enhanced computational efficiency through vectorization
- Improved data handling with modern Python libraries
- Maintained backward compatibility for existing workflows

*Parameter Files:*

- JSON format replaces MATLAB .mat parameter files
- Automatic conversion utilities provided for existing parameter sets
- Enhanced parameter validation and error checking
- Comprehensive default parameter sets for common applications

---

**Version History Summary**

- **v0.1.0**: Initial Python implementation with complete feature set
- **Future versions**: Will follow semantic versioning with detailed changelogs


**Acknowledgments**

This Python implementation builds upon the original SnowClim model and acknowledges the contributions of the broader snow modeling community, including field data providers, algorithm developers, and validation researchers.

For detailed technical information about model physics and implementation, please refer to the original SnowClim publication `(Lute et al., 2022) <https://doi.org/10.5194/gmd-15-5045-2022>`_  and the comprehensive model documentation.
