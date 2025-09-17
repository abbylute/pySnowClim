# pySnowClim Examples

This directory contains example scripts and data for running the pySnowClim model.

## Files

- `run_snowclim_example.py` - Main example script that demonstrates how to run pySnowClim
- `forcings_example.nc` - Example forcing data (temperature, precipitation, radiation, etc.)
- `target_example.nc` - Observed snow water equivalent data for validation
- `output/` - Directory where model outputs and validation plots are saved

## Running the Example

1. Make sure you have the required data files in this directory:
   - `forcings_example.nc`
   - `target_example.nc`

2. Run the example script:
   ```bash
   cd examples
   python run_snowclim_example.py
