"""
Example script for running pySnowClim model with sample data.

This script:
1. Loads forcing data from forcings_example.nc
2. Runs the pySnowClim model
3. Compares model output with observations from target_example.nc
4. Creates plots showing the comparison
"""
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# Add the src directory to Python path to import pySnowClim modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runsnowclim_model import run_model


def load_and_validate_data(forcings_file, observations_file):
    """
    Load and validate the forcing and observation data.

    Parameters:
    -----------
    forcings_file : str
        Path to the forcing NetCDF file
    observations_file : str
        Path to the observations NetCDF file

    Returns:
    --------
    forcings_ds : xarray.Dataset
        Forcing data
    obs_ds : xarray.Dataset
        Observation data
    """
    print("Loading forcing data...")
    forcings_ds = xr.open_dataset(forcings_file)

    print("Loading observation data...")
    obs_ds = xr.open_dataset(observations_file)

    print(f"Forcing data time range: {forcings_ds.time.values[0]} to {forcings_ds.time.values[-1]}")
    print(f"Observation data time range: {obs_ds.time.values[0]} to {obs_ds.time.values[-1]}")
    print(f"Number of forcing time steps: {len(forcings_ds.time)}")
    print(f"Number of observation time steps: {len(obs_ds.time)}")

    return forcings_ds, obs_ds


def run_pysnowclim_example(forcings_file, output_dir):
    """
    Run the pySnowClim model with example data.

    Parameters:
    -----------
    forcings_file : str
        Path to the forcing NetCDF file
    output_dir : str
        Directory to save model outputs

    Returns:
    --------
    model_output : list
        Model output for each timestep
    """
    print("Running pySnowClim model...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the model
    _ = run_model(
        forcings_path=forcings_file,
        parameters_path=None,  # Use default parameters
        outputs_path=output_dir,
        save_format='.nc'
    )

    print("Model run completed!")


def load_model_results(output_dir):
    """
    Load model results from the output directory.

    Parameters:
    -----------
    output_dir : str
        Directory containing model outputs

    Returns:
    --------
    swe_data : xarray.DataArray
        Snow water equivalent results
    """
    swe_file = os.path.join(output_dir, 'SnowWaterEq.nc')
    if os.path.exists(swe_file):
        swe_data = xr.open_dataarray(swe_file)
        return swe_data
    else:
        print(f"SWE output file not found: {swe_file}")
        return None


def compare_with_observations(model_swe, obs_ds, output_dir):
    """
    Compare model results with observations and create plots.

    Parameters:
    -----------
    model_swe : xarray.DataArray
        Model snow water equivalent
    obs_ds : xarray.Dataset
        Observation dataset
    output_dir : str
        Directory to save comparison plots
    """
    print("Comparing model results with observations...")

    # Extract observed SWE (convert from mm to m if needed)
    obs_swe = obs_ds.swe.values
    obs_time = obs_ds.time.values

    # Extract model SWE - take the spatial mean if there are multiple grid points
    if model_swe.ndim > 1:
        model_swe_values = model_swe.mean(dim=['lat', 'lon']).values
    else:
        model_swe_values = model_swe.values.flatten()

    model_time = model_swe.time.values

    # Align time series (interpolate model to observation times if needed)
    if len(model_time) != len(obs_time):
        print("Warning: Model and observation time series have different lengths")
        print(f"Model timesteps: {len(model_time)}, Observation timesteps: {len(obs_time)}")

        # Simple alignment - truncate to minimum length
        min_len = min(len(model_time), len(obs_time))
        model_swe_values = model_swe_values[:min_len]
        obs_swe = obs_swe[:min_len]
        model_time = model_time[:min_len]
        obs_time = obs_time[:min_len]

    # Calculate statistics
    correlation = np.corrcoef(model_swe_values, obs_swe)[0, 1]
    rmse = np.sqrt(np.mean((model_swe_values - obs_swe)**2))
    mae = np.mean(np.abs(model_swe_values - obs_swe))
    bias = np.mean(model_swe_values - obs_swe)

    print("\nComparison Statistics:")
    print(f"Correlation: {correlation:.3f}")
    print(f"RMSE: {rmse:.3f} mm")
    print(f"MAE: {mae:.3f} mm")
    print(f"Bias: {bias:.3f} mm")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Time series plot
    axes[0, 0].plot(pd.to_datetime(obs_time), obs_swe, 'b-', label='Observed', linewidth=2)
    axes[0, 0].plot(pd.to_datetime(model_time), model_swe_values, 'r-', label='Modeled', linewidth=1)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Snow Water Equivalent (mm)')
    axes[0, 0].set_title('Snow Water Equivalent Time Series')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Scatter plot
    axes[0, 1].scatter(obs_swe, model_swe_values, alpha=0.6)
    max_val = max(np.max(obs_swe), np.max(model_swe_values))
    axes[0, 1].plot([0, max_val], [0, max_val], 'k--', label='1:1 line')
    axes[0, 1].set_xlabel('Observed SWE (mm)')
    axes[0, 1].set_ylabel('Modeled SWE (mm)')
    axes[0, 1].set_title(f'Scatter Plot (r = {correlation:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Residuals plot
    residuals = model_swe_values - obs_swe
    axes[1, 0].plot(pd.to_datetime(obs_time), residuals, 'g-')
    axes[1, 0].axhline(y=0, color='k', linestyle='--')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Residuals (mm)')
    axes[1, 0].set_title('Model Residuals')
    axes[1, 0].grid(True)

    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residuals (mm)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residuals Distribution')
    axes[1, 1].grid(True)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, 'model_validation.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Validation plot saved to: {plot_file}")
    #plt.show()

    # Save comparison statistics to file
    stats_file = os.path.join(output_dir, 'validation_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("pySnowClim Model Validation Statistics\n")
        f.write("="*40 + "\n")
        f.write(f"Station: {obs_ds.station_name.values if 'station_name' in obs_ds else 'Unknown'}\n")
        f.write(f"Elevation: {obs_ds.elevation.values if 'elevation' in obs_ds else 'Unknown'} m\n")
        f.write(f"Latitude: {obs_ds.latitude.values if 'latitude' in obs_ds else 'Unknown'}°\n")
        f.write(f"Longitude: {obs_ds.longitude.values if 'longitude' in obs_ds else 'Unknown'}°\n")
        f.write(f"Time period: {obs_time[0]} to {obs_time[-1]}\n")
        f.write(f"Number of data points: {len(obs_swe)}\n\n")
        f.write(f"Correlation coefficient: {correlation:.3f}\n")
        f.write(f"Root Mean Square Error: {rmse:.3f} mm\n")
        f.write(f"Mean Absolute Error: {mae:.3f} mm\n")
        f.write(f"Bias (model - observed): {bias:.3f} mm\n")
        f.write(f"Max observed SWE: {np.max(obs_swe):.1f} mm\n")
        f.write(f"Max modeled SWE: {np.max(model_swe_values):.1f} mm\n")

    print(f"Validation statistics saved to: {stats_file}")


def main():
    """Main function to run the example."""

    # Set up file paths
    example_dir = os.path.dirname(__file__)
    forcings_file = os.path.join(example_dir, 'forcings_example.nc')
    observations_file = os.path.join(example_dir, 'target_example.nc')
    output_dir = os.path.join(example_dir, 'output')

    print("="*60)
    print("pySnowClim Example Run")
    print("="*60)

    # Check if input files exist
    if not os.path.exists(forcings_file):
        print(f"Error: Forcing file not found: {forcings_file}")
        print("Please ensure 'forcings_example.nc' is in the examples folder.")
        return

    if not os.path.exists(observations_file):
        print(f"Error: Observation file not found: {observations_file}")
        print("Please ensure 'target_example.nc' is in the examples folder.")
        return

    try:
        # Load and validate data
        forcings_ds, obs_ds = load_and_validate_data(forcings_file, observations_file)

        # Run the model
        run_pysnowclim_example(forcings_file, output_dir)

        # Load model results
        model_swe = load_model_results(output_dir)

        if model_swe is not None:
            # Compare with observations
            compare_with_observations(model_swe, obs_ds, output_dir)
            print("\nExample run completed successfully!")
            print(f"Results saved in: {output_dir}")
        else:
            print("Error: Could not load model results for comparison.")

    except Exception as e:
        print(f"Error running example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
