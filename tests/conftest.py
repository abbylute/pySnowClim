"""Shared test fixtures."""
import pytest

@pytest.fixture
def default_parameters():
    """Provide default parameters for testing."""
    from createParameterFile import create_dict_parameters
    return create_dict_parameters()

@pytest.fixture
def sample_domain_2d():
    """Provide a small 2D domain for testing."""
    return (3, 3)
