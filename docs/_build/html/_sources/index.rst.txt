pySnowClim Documentation
========================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
   :alt: License

Welcome to pySnowClim, a robust, efficient, and open-source Python implementation
of the SnowClim model. This package provides tools for simulating
snow accumulation, melting, sublimation, and energy balance processes across
various spatial and temporal scales.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quick_start
   user_guide
   api_reference
   examples
   model_description
   output_variables
   contributing
   changelog

Overview
--------

``pySnowClim`` addresses the practical need for snow models that can accurately
simulate snow accumulation and melt across both large regions and fine spatial detail.
The model combines key physical processes with efficient algorithms to provide
high-resolution snowpack estimates, especially useful in complex terrain.


Key Features
-----------

* **Energy Balance Modeling**: Comprehensive energy balance calculations including sensible heat, latent heat, and radiation
* **Multiple Albedo Options**: Three different albedo parameterizations (Essery et al. 2013, Utah Snow Model, VIC model)
* **Snow Physics**: Realistic snow density evolution, refreezing processes, and liquid water content
* **Flexible Input/Output**: Support for NetCDF and NumPy array formats
* **Efficient**: Vectorized operations for processing multiple grid points simultaneously
* **Open Source**: MIT licensed for academic and commercial use


Quick Navigation
---------------

* :doc:`installation` - Get started with installing pySnowClim
* :doc:`quick_start` - Run your first simulation
* :doc:`api_reference` - Complete API documentation
* :doc:`examples` - Example workflows and use cases

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
