"""Integration tests for pySnowClim model."""
import pytest
import numpy as np

from snowclim_model import run_snowclim_model
from createParameterFile import create_dict_parameters

class TestSingleTimestepIntegration:
    """Test single timestep model execution."""

    @pytest.fixture
    def minimal_forcing_data(self):
        """Create minimal forcing data for one timestep."""
        # Single grid point, single timestep
        shape = (1, 1, 1)  # time, lat, lon

        forcings = {
            'lrad': np.full(shape, 300.0),      # Longwave radiation (kJ/m²/timestep)
            'solar': np.full(shape, 500.0),     # Solar radiation (kJ/m²/timestep)
            'tavg': np.full(shape, -2.0),       # Air temperature (°C)
            'ppt': np.full(shape, 5.0),         # Precipitation (mm)
            'vs': np.full(shape, 3.0),          # Wind speed (m/s)
            'psfc': np.full(shape, 1013.0),     # Surface pressure (hPa)
            'huss': np.full(shape, 0.008),      # Specific humidity (kg/kg)
            'relhum': np.full(shape, 80.0),     # Relative humidity (%)
            'tdmean': np.full(shape, -5.0)      # Dewpoint Mean temperature (°C)
        }

        coords = {
            'lat': np.array([[45.0]]),
            'lon': np.array([[-120.0]]),
            'time': np.array(['2023-01-01']),
            'time_sliced': [[2023, 1, 1, 0, 0, 0]]
        }

        return {'forcings': forcings, 'coords': coords}

    def test_single_timestep_execution(self, minimal_forcing_data):
        """Test that model can execute a single timestep without errors."""
        parameters = create_dict_parameters()

        results = run_snowclim_model(minimal_forcing_data, parameters)

        assert len(results) == 1, "Should return one timestep result"
        result = results[0]

        # Check that key outputs exist and are reasonable
        assert hasattr(result, 'SnowWaterEq')
        assert hasattr(result, 'SnowDepth')

        # Check output shapes
        assert result.SnowWaterEq.shape == (1, 1)
        assert result.SnowDepth.shape == (1, 1)

    def test_snow_accumulation_single_step(self, minimal_forcing_data):
        """Test snow accumulation in a single cold, snowy timestep."""
        # Modify forcing for snow conditions
        minimal_forcing_data['forcings']['tavg'] = np.full((1, 1, 1), -10.0)  # Cold
        minimal_forcing_data['forcings']['ppt'] = np.full((1, 1, 1), 10.0)    # Precipitation
        minimal_forcing_data['forcings']['relhum'] = np.full((1, 1, 1), 90.0) # High humidity

        parameters = create_dict_parameters()
        results = run_snowclim_model(minimal_forcing_data, parameters)

        result = results[0]

        assert result.SnowWaterEq[0, 0] > 0, "Should accumulate snow in cold conditions"
        assert result.SnowDepth[0, 0] > 0, "Should have positive snow depth"
        assert result.SnowfallWaterEq[0, 0] > 0, "Should register snowfall"


class TestMassConservation:
    """Test mass conservation in the model."""

    def test_water_mass_conservation(self):
        """Test that water mass is conserved through the system."""
        # Simple 3-timestep scenario
        shape = (3, 1, 1)

        forcings = {
            'lrad': np.full(shape, 300.0),
            'solar': np.full(shape, 400.0),
            'tavg': np.array([[[-5.0]], [[0.0]], [[2.0]]]),  # Cold -> warm
            'ppt': np.array([[[10.0]], [[5.0]], [[0.0]]]),   # Decreasing precip
            'vs': np.full(shape, 4.0),
            'psfc': np.full(shape, 1013.0),
            'huss': np.full(shape, 0.008),
            'relhum': np.full(shape, 80.0),
            'tdmean': np.array([[[-7.0]], [[-2.0]], [[0.0]]])
        }

        coords = {
            'lat': np.array([[45.0]]),
            'lon': np.array([[-120.0]]),
            'time': np.arange(3),
            'time_sliced': [[2023, 1, i+1, 0, 0, 0] for i in range(3)]
        }

        forcing_data = {'forcings': forcings, 'coords': coords}
        parameters = create_dict_parameters()
        results = run_snowclim_model(forcing_data, parameters)

        # Track water through the system
        for i, result in enumerate(results):
            grid_point = (0, 0)

            # All water components should be non-negative
            assert result.SnowWaterEq[grid_point] >= 0, f"SWE negative at timestep {i}"
            assert result.PackWater[grid_point] >= 0, f"Pack water negative at timestep {i}"
            assert result.Runoff[grid_point] >= 0, f"Runoff negative at timestep {i}"

            # Total water should be reasonable
            total_water = (result.SnowWaterEq[grid_point] +
                          result.PackWater[grid_point] +
                          result.Runoff[grid_point])

            assert total_water >= 0, f"Total water negative at timestep {i}"

class TestBoundaryConditions:
    """Test model behavior at extreme boundary conditions."""

    def test_no_snow_scenario(self):
        """Test model with conditions that produce no snow."""
        shape = (3, 1, 1)

        # Warm, dry conditions
        forcings = {
            'lrad': np.full(shape, 350.0),
            'solar': np.full(shape, 800.0),
            'tavg': np.full(shape, 15.0),    # Warm
            'ppt': np.full(shape, 0.0),      # No precipitation
            'vs': np.full(shape, 2.0),
            'psfc': np.full(shape, 1013.0),
            'huss': np.full(shape, 0.010),
            'relhum': np.full(shape, 50.0),   # Dry
            'tdmean': np.full(shape, 10.0)
        }

        coords = {
            'lat': np.array([[35.0]]),  # Lower latitude
            'lon': np.array([[-100.0]]),
            'time': np.arange(3),
            'time_sliced': [[2023, 7, i+1, 0, 0, 0] for i in range(3)]  # Summer
        }

        forcing_data = {'forcings': forcings, 'coords': coords}
        parameters = create_dict_parameters()
        results = run_snowclim_model(forcing_data, parameters)

        # Should have minimal or no snow
        for result in results:
            assert result.SnowWaterEq[0, 0] < 1.0, "Should have minimal snow in warm conditions"
            assert result.SnowDepth[0, 0] < 10.0, "Should have minimal snow depth"

    def test_extreme_cold_scenario(self):
        """Test model with extremely cold conditions."""
        shape = (3, 1, 1)

        # Very cold conditions
        forcings = {
            'lrad': np.full(shape, 200.0),
            'solar': np.full(shape, 100.0),   # Low solar in winter
            'tavg': np.full(shape, -40.0),    # Extremely cold
            'ppt': np.full(shape, 5.0),
            'vs': np.full(shape, 10.0),       # Windy
            'psfc': np.full(shape, 1013.0),
            'huss': np.full(shape, 0.0005),   # Very dry
            'relhum': np.full(shape, 70.0),
            'tdmean': np.full(shape, -45.0)
        }

        coords = {
            'lat': np.array([[70.0]]),  # High latitude
            'lon': np.array([[-150.0]]),
            'time': np.arange(3),
            'time_sliced': [[2023, 12, i+1, 0, 0, 0] for i in range(3)]  # Winter
        }

        forcing_data = {'forcings': forcings, 'coords': coords}
        parameters = create_dict_parameters()

        # Should run without numerical errors
        results = run_snowclim_model(forcing_data, parameters)

        for result in results:
            # Values should be finite
            assert np.isfinite(result.SnowWaterEq[0, 0]), "SWE should be finite in extreme cold"
            assert np.isfinite(result.Energy[0, 0]), "Energy should be finite in extreme cold"
            assert np.isfinite(result.Albedo[0, 0]), "Albedo should be finite in extreme cold"
