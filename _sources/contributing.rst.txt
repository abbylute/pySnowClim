Contributing to pySnowClim
==========================

We welcome contributions to pySnowClim! This document provides guidelines for contributing code, documentation, bug reports, and feature requests.


Getting Started
---------------

Types of Contributions
~~~~~~~~~~~~~~~~~~~~~~

We welcome several types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new capabilities or improvements
- **Code Contributions**: Implement bug fixes, new features, or optimizations
- **Documentation**: Improve documentation, examples, and tutorials
- **Testing**: Add test cases, validate model behavior, or benchmark performance
- **Scientific Validation**: Compare against observations or other models

Prerequisites
~~~~~~~~~~~~~

Before contributing, please:

1. **Familiarize yourself** with pySnowClim by running examples and reading documentation
2. **Check existing issues** to avoid duplicate work
3. **Read this contributing guide** completely
4. **Set up a development environment** as described below

Development Setup
-----------------

Environment Setup
~~~~~~~~~~~~~~~~~

1. **Fork and Clone the Repository**:

.. code-block:: bash

   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/pySnowClim.git
   cd pySnowClim

2. **Create Development Environment**:

.. code-block:: bash

   # Create virtual environment
   python -m venv pysnowclim-dev
   source pysnowclim-dev/bin/activate  # On Windows: pysnowclim-dev\Scripts\activate

   # Install in development mode with dev dependencies
   pip install -e ".[dev]"

3. **Verify Installation**:

.. code-block:: bash

   # Run basic tests
   python -m pytest tests/

   # Test basic model functionality
   python run_main.py

Code Standards
~~~~~~~~~~~~~~

**Python Style:**

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Maximum line length: 88 characters (Black formatter default)

**Documentation Style:**

- Use Google or NumPy docstring format
- Include parameter types and descriptions
- Provide usage examples for complex functions
- Update relevant documentation files

**Code Organization:**

- Keep functions focused and modular
- Maintain separation between physics calculations and I/O operations
- Use appropriate error handling and validation
- Add comments for complex algorithms or physics


Getting Help
~~~~~~~~~~~~

If you need help contributing:

1. **Check Documentation**: Review existing documentation and examples
2. **Search Issues**: Look for similar problems or discussions
3. **Ask Questions**: Open a GitHub issue or discussion
4. **Start Small**: Begin with simple contributions to learn the workflow


Thank you for contributing to pySnowClim! Your contributions help advance snow science and support the broader research community.
