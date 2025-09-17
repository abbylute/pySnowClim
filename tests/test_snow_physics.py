"""Test snow physics calculations."""
import pytest
import numpy as np

from calcFreshSnowDensity import calc_fresh_snow_density
from calcPhase import calc_phase
from calcSnowDensity import calc_snow_density
from calcSnowDensityAfterSnow import calc_snow_density_after_snow


def assert_physical_bounds(values, min_val=None, max_val=None, variable_name=""):
    """Helper to check physical bounds."""
    if min_val is not None:
        assert np.all(
            values >= min_val), f"{variable_name} has values below {min_val}"
    if max_val is not None:
        assert np.all(
            values <= max_val), f"{variable_name} has values above {max_val}"


def assert_no_nan_or_inf(values, variable_name=""):
    """Helper to check for NaN or infinite values."""
    assert np.all(np.isfinite(values)
                  ), f"{variable_name} contains NaN or infinite values"


sec_in_timestep = 3600


class TestFreshSnowDensity:
    """Test fresh snow density calculations."""

    def test_density_at_freezing(self):
        """Test density calculation at 0°C."""
        density = calc_fresh_snow_density(0.0)
        assert_physical_bounds(density, min_val=50, max_val=500,
                               variable_name="Fresh snow density")

    def test_density_cold_conditions(self):
        """Test density at very cold temperatures."""
        temps = np.array([-30, -20, -10, -5])
        densities = calc_fresh_snow_density(temps)

        assert len(densities) == len(temps)
        assert_physical_bounds(densities, min_val=50, max_val=500)
        assert_no_nan_or_inf(densities, "Fresh snow density")

        # Colder should generally mean lower density (at very cold temps)
        assert densities[0] <= densities[-1], "Density should increase with temperature"

    def test_density_warm_conditions(self):
        """Test density at warm (but below freezing) conditions."""
        temps = np.array([-2, -1, -0.5])
        densities = calc_fresh_snow_density(temps)

        # At warmer temps, density should be higher
        assert np.all(
            densities > 100), "Density should be higher at warmer temps"

    def test_fresh_snow_density_cold(self):
        """Test fresh snow density at very cold temperatures"""
        temp = -20.0  # °C
        density = calc_fresh_snow_density(temp)
        assert density >= 50, "Density should be at least minimum value"


class TestPrecipitationPhase:
    """Test precipitation phase determination."""

    def test_phase_cold_snow(self):
        """Test phase calculation favoring snow."""
        psnow = calc_phase(-10.0, 80.0)
        assert 0.7 <= psnow <= 1.0, "Should be mostly snow at -10°C"

    def test_phase_all_rain(self):
        """Test phase calculation for conditions favoring rain"""
        temp = 10.0   # °C
        rh = 50.0     # %
        psnow = calc_phase(temp, rh)
        assert 0.0 <= psnow <= 0.2, "Should be mostly rain at 10°C"

    def test_phase_boundary_valid(self):
        """Test that phase probability is always valid."""
        temps = np.linspace(-20, 20, 10)
        rh_values = np.linspace(20, 100, 10)

        for temp in temps:
            for rh in rh_values:
                psnow = calc_phase(temp, rh)
                assert 0.0 <= psnow <= 1.0, f"Invalid probability at T={temp}, RH={rh}"


class TestSnowDensityEvolution:
    """Test snow density compaction and evolution."""

    def test_density_compaction(self):
        """Test that snow density increases over time."""
        initial_density = 200.0  # kg/m³
        swe = 0.5  # m
        temp = -5.0  # °C

        new_density = calc_snow_density(
            swe, temp, initial_density, sec_in_timestep)

        assert new_density >= initial_density, "Density should not decrease during compaction"
        assert_physical_bounds(new_density, min_val=50, max_val=800)

    def test_density_temperature_effect(self):
        """Test that warmer temperatures lead to more compaction."""
        initial_density = 150.0
        swe = 0.3

        cold_density = calc_snow_density(
            swe, -20.0, initial_density, sec_in_timestep)
        warm_density = calc_snow_density(
            swe, -2.0, initial_density, sec_in_timestep)

        assert warm_density >= cold_density, "Warmer temps should cause more compaction"

    def test_density_after_snowfall(self):
        """Test density calculation after new snowfall."""
        old_swe, new_swe = 0.3, 0.1
        old_density, new_snow_density = 300.0, 100.0

        combined_density = calc_snow_density_after_snow(
            old_swe, new_swe, old_density, new_snow_density
        )

        # Should be weighted average
        expected = (old_swe + new_swe) / \
            ((old_swe/old_density) + (new_swe/new_snow_density))
        assert abs(combined_density - expected) < 1e-6
        assert_physical_bounds(combined_density, min_val=50, max_val=600)

    def test_density_after_snowfall_array(self, sample_domain_2d):
        """Test density calculation with arrays."""
        shape = sample_domain_2d

        old_swe = np.full(shape, 0.2)
        new_swe = np.full(shape, 0.05)
        old_density = np.full(shape, 250.0)
        new_density = np.full(shape, 120.0)

        combined = calc_snow_density_after_snow(
            old_swe, new_swe, old_density, new_density)

        assert combined.shape == shape
        assert_physical_bounds(combined, min_val=50, max_val=600)
        # Combined should be between old and new densities
        assert np.all(combined >= new_density)
        assert np.all(combined <= old_density)
