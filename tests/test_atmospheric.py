"""Test atmospheric calculations."""
import pytest
import numpy as np

from calcSpecificHumidity import calculate_specific_humidity
from calcInclinationAngle import calc_inclination_angle
from calcLatHeatVap import calculate_lat_heat_vap
from calcLatHeatSub import calculate_lat_heat_sub

def assert_physical_bounds(values, min_val=None, max_val=None, variable_name=""):
    """Helper to check physical bounds."""
    if min_val is not None:
        assert np.all(values >= min_val), f"{variable_name} has values below {min_val}"
    if max_val is not None:
        assert np.all(values <= max_val), f"{variable_name} has values above {max_val}"

def assert_no_nan_or_inf(values, variable_name=""):
    """Helper to check for NaN or infinite values."""
    assert np.all(np.isfinite(values)), f"{variable_name} contains NaN or infinite values"


class TestSpecificHumidity:
    """Test specific humidity calculations."""

    def test_specific_humidity_basic(self):
        """Test basic specific humidity calculation."""
        td = 5.0      # °C
        pressure = 1013.0  # mb

        sh = calculate_specific_humidity(td, pressure)

        assert sh > 0, "Specific humidity should be positive"
        assert_physical_bounds(sh, min_val=0, max_val=0.05)

    def test_specific_humidity_temperature_effect(self):
        """Test effect of temperature on specific humidity."""
        pressure = 1013.0

        sh_cold = calculate_specific_humidity(-10.0, pressure)
        sh_warm = calculate_specific_humidity(20.0, pressure)

        assert sh_warm > sh_cold, "Warmer air should hold more moisture"

class TestInclinationAngle:
    """Test solar inclination angle calculations."""

    def test_inclination_angle_seasonal(self):
        """Test seasonal variation in inclination angle."""
        lat = 45.0

        summer_angle = calc_inclination_angle(lat, 6, 21)  # Summer solstice
        winter_angle = calc_inclination_angle(lat, 12, 21)  # Winter solstice

        assert_physical_bounds(summer_angle, min_val=0, max_val=90,
                               variable_name="Summer inclination angle")

        assert_physical_bounds(winter_angle, min_val=0, max_val=90,
                               variable_name="Winter inclination angle")

        assert summer_angle > winter_angle, "Summer should have higher sun angle"

    def test_inclination_angle_array(self):
        """Test with array of latitudes."""
        lats = np.array([30, 45, 60])
        month = 3  # Equinox
        day = 21

        angles = calc_inclination_angle(lats, month, day)

        assert len(angles) == len(lats)
        assert_physical_bounds(angles, min_val=0, max_val=90)

    def test_inclination_angle_physics(self):
        """Test physical constraints of solar inclination angle."""
        # At the poles during winter, sun angle should be very low
        arctic_winter = calc_inclination_angle(80.0, 12, 21)  # 80°N, winter solstice
        assert arctic_winter < 20, "Arctic winter sun should be very low"

        # At equator during equinox, sun should be high
        equator_equinox = calc_inclination_angle(0.0, 3, 21)  # Equator, spring equinox
        assert 60 < equator_equinox < 90, "Equatorial sun should be high at equinox"

        # Test southern hemisphere
        south_summer = calc_inclination_angle(-30.0, 12, 21)  # 30°S, summer solstice
        north_summer = calc_inclination_angle(30.0, 6, 21)    # 30°N, summer solstice

        # Both should be relatively high (summer conditions)
        assert south_summer > 50 and north_summer > 50

class TestLatentHeat:
    """Test latent heat calculations."""

    def test_latent_heat_vaporization(self):
        """Test latent heat of vaporization calculation."""
        temps = np.array([-15, -10, -5, 0])  # °C

        lh_vap = calculate_lat_heat_vap(temps)

        assert_physical_bounds(lh_vap, min_val=2400, max_val=2600,
                             variable_name="Latent heat of vaporization")
        assert_no_nan_or_inf(lh_vap, "Latent heat vaporization")

        assert np.all(np.diff(lh_vap) < 0), "Latent heat should decrease with temperature"

    def test_latent_heat_sublimation(self):
        """Test latent heat of sublimation calculation."""
        temps = np.array([-20, -10, -5, 0])  # °C

        lh_sub = calculate_lat_heat_sub(temps)

        assert_physical_bounds(lh_sub, min_val=2800, max_val=2850,
                             variable_name="Latent heat of sublimation")
        assert_no_nan_or_inf(lh_sub, "Latent heat sublimation")

        lh_vap = calculate_lat_heat_vap(temps)
        assert np.all(lh_sub > lh_vap), "Sublimation should require more energy than vaporization"

    def test_latent_heat_temperature_sensitivity(self):
        """Test temperature sensitivity of latent heat."""
        temp_cold = -20.0
        temp_warm = 0.0

        lh_sub_cold = calculate_lat_heat_sub(temp_cold)
        lh_sub_warm = calculate_lat_heat_sub(temp_warm)

        assert lh_sub_cold > lh_sub_warm, "Latent heat should be higher at colder temps"

        diff_percent = (lh_sub_cold - lh_sub_warm) / lh_sub_warm * 100
        assert diff_percent < 5, "Temperature effect should be modest (<5%)"
