Installation
============

Requirements
-----------

pySnowClim requires Python 3.8 or later and the following packages:

* numpy
* xarray
* scipy
* netCDF4
* pandas
* requests

Installation Methods
-------------------

From Source
~~~~~~~~~~~

To install from source, clone the repository and install:

.. code-block:: bash

   git clone https://github.com/abbylute/pySnowClim.git
   cd pySnowClim
   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

For development, install with additional dependencies:

.. code-block:: bash

   git clone https://github.com/abbylute/pySnowClim.git
   cd pySnowClim
   pip install -e ".[dev]"

Verification
-----------

Test your installation by running:

.. code-block:: bash

   python verify_instalation.py

