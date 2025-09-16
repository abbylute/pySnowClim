"""Test energy balance calculations."""
import pytest
import numpy as np

from calcTurbulentFluxes import calc_turbulent_fluxes
from calcLongwave import calc_longwave
from calcEnergyToCC import calc_energy_to_cc

def assert_physical_bounds(values, min_val=None, max_val=None, variable_name=""):
    """Helper to check physical bounds."""
    if min_val is not None:
        assert np.all(values >= min_val), f"{variable_name} has values below {min_val}"
    if max_val is not None:
        assert np.all(values <= max_val), f"{variable_name} has values above {max_val}"

def assert_no_nan_or_inf(values, variable_name=""):
    """Helper to check for NaN or infinite values."""
    assert np.all(np.isfinite(values)), f"{variable_name} contains NaN or infinite values"

sec_in_timestep = 3600

class TestTurbulentFluxes:
    """Test turbulent flux calculations."""

    def test_turbulent_fluxes_basic(self, default_parameters, sample_domain_2d):
        """Test basic turbulent flux calculation."""
        shape = sample_domain_2d

        # Input data
        wind_speed = np.full(shape, 5.0)
        snow_temp = np.full(shape, -2.0)
        air_temp = np.full(shape, 0.0)
        pressure = np.full(shape, 1013.0)
        humidity = np.full(shape, 0.008)        

        H, E, EV = calc_turbulent_fluxes(
            default_parameters, wind_speed, snow_temp, air_temp,
            pressure, humidity, sec_in_timestep
        )

        # Check outputs have correct shape
        assert H.shape == shape
        assert E.shape == shape
        assert EV.shape == shape

        # Check for finite values
        assert_no_nan_or_inf(H, "Sensible heat flux")
        assert_no_nan_or_inf(E, "Mass flux")
        assert_no_nan_or_inf(EV, "Latent heat flux")

    def test_zero_wind_speed(self, default_parameters, sample_domain_2d):
        """Test behavior with zero wind speed."""
        shape = sample_domain_2d

        wind_speed = np.zeros(shape)
        snow_temp = np.full(shape, -2.0)
        air_temp = np.full(shape, 0.0)
        pressure = np.full(shape, 1001.0)
        humidity = np.full(shape, 0.008)

        H, E, EV = calc_turbulent_fluxes(
            default_parameters, wind_speed, snow_temp, air_temp,
            pressure, humidity, sec_in_timestep
        )

        # With zero wind, fluxes should be zero or very small
        assert np.all(np.abs(H) < 1e-6), "Sensible heat should be ~0 with no wind"
        assert np.all(np.abs(E) < 1e-6), "Mass flux should be ~0 with no wind"

    def test_temperature_gradient_effect(self, default_parameters):
        """Test that temperature gradients affect sensible heat flux."""
        # Warm air, cold snow
        H1, _, _ = calc_turbulent_fluxes(
            default_parameters,
            wind_speed=np.array([5.0]),
            lastsnowtemp=np.array([-10.0]),
            tavg=np.array([5.0]),
            psfc=np.array([1013.0]),
            huss=np.array([0.008]),
            sec_in_ts=sec_in_timestep
        )

        # Cold air, cold snow (smaller gradient)
        H2, _, _ = calc_turbulent_fluxes(
            default_parameters,
            wind_speed=np.array([5.0]),
            lastsnowtemp=np.array([-10.0]),
            tavg=np.array([-5.0]),
            psfc=np.array([1013.0]),
            huss=np.array([0.008]),
            sec_in_ts=sec_in_timestep
        )

        assert np.abs(H1[0]) > np.abs(H2[0]), "Larger temp gradient should give larger flux"

class TestLongwaveRadiation:
    """Test longwave radiation calculations."""

    def test_longwave_calculation(self):
        """Test longwave radiation computation."""
        emissivity = 0.98
        temp = -5.0  # °C
        lw_down = 300.0  # kJ/m²

        lw_up = calc_longwave(emissivity, temp, lw_down, sec_in_timestep)

        assert lw_up > 0, "Upward longwave should be positive"
        assert_no_nan_or_inf(lw_up, "Longwave radiation")

    def test_longwave_array_input(self, sample_domain_2d):
        """Test with array inputs."""
        shape = sample_domain_2d

        emissivity = 0.98
        temp = np.full(shape, -2.0)
        lw_down = np.full(shape, 300.0)

        lw_up = calc_longwave(emissivity, temp, lw_down, sec_in_timestep)

        assert lw_up.shape == shape
        assert np.all(lw_up > 0)
        assert_no_nan_or_inf(lw_up, "Longwave radiation array")

    def test_emissivity_effect(self):
        """Test that emissivity affects longwave radiation."""
        temp = 0.0
        lw_down = 300.0

        lw_high_emis = calc_longwave(0.98, temp, lw_down, sec_in_timestep)
        lw_low_emis = calc_longwave(0.85, temp, lw_down, sec_in_timestep)

        assert lw_high_emis != lw_low_emis, "Emissivity should affect longwave radiation"

class TestEnergyToColdContent:
    """Test energy distribution to cold content."""

    def test_energy_to_cc_basic(self, sample_domain_2d):
        """Test basic energy to cold content calculation."""
        shape = sample_domain_2d

        lastpackcc = np.full(shape, -100.0)  # Negative = cold
        lastenergy = np.full(shape, 50.0)    # Positive = available energy
        CCenergy = np.zeros(shape)

        new_cc, new_energy, new_cc_energy = calc_energy_to_cc(
            lastpackcc, lastenergy, CCenergy
        )

        assert new_cc.shape == shape
        assert new_energy.shape == shape
        assert new_cc_energy.shape == shape

        # Energy should have been used to warm the pack
        assert np.all(new_cc >= lastpackcc), "Cold content should decrease (become less negative)"
        assert np.all(new_energy <= lastenergy), "Energy should be consumed"

    def test_no_cold_content(self, sample_domain_2d):
        """Test when there's no cold content."""
        shape = sample_domain_2d

        lastpackcc = np.zeros(shape)  # No cold content
        lastenergy = np.full(shape, 100.0)
        CCenergy = np.zeros(shape)

        new_cc, new_energy, new_cc_energy = calc_energy_to_cc(
            lastpackcc, lastenergy, CCenergy
        )

        # Nothing should change
        np.testing.assert_array_equal(new_cc, lastpackcc)
        np.testing.assert_array_equal(new_energy, lastenergy)
