"""Basic integration and package-level tests."""
import pytest

def test_package_import():
    """Ensure the pySnowclim package can be imported."""
    try:
        import snowclim_model
        import runsnowclim_model
        import constants
    except ImportError as e:
        pytest.fail(f"Failed to import pySnowclim package: {e}")

def test_constants_availability():
    """Test that key constants are accessible."""
    import constants as const

    assert hasattr(const, 'WATERDENS')
    assert const.WATERDENS == 1000
    assert hasattr(const, 'LATHEATFREEZ')
    assert const.LATHEATFREEZ == 333.3

def test_model_run_basic_functionality():
    """Basic test that model can be instantiated without errors."""
    from createParameterFile import create_dict_parameters

    parameters = create_dict_parameters()
    assert isinstance(parameters, dict)
    assert 'hours_in_ts' in parameters
    assert 'albedo_option' in parameters
    assert 'max_albedo' in parameters
