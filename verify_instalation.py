"""Verify pySnowClim installation."""

def verify_installation():
    """Verify that pySnowClim is properly installed."""
    try:
        # Test basic imports
        import numpy
        print("✓ numpy imported successfully")
        
        import xarray  
        print("✓ xarray imported successfully")
        
        import scipy
        print("✓ scipy imported successfully")
        
        # Test pySnowClim imports
        try:
            from src import run_snowclim_model, create_dict_parameters
            print("✓ pySnowClim core functions imported successfully")
        except ImportError:
            # Fallback for development installation
            import sys
            sys.path.append('src/')
            from snowclim_model import run_snowclim_model
            from createParameterFile import create_dict_parameters
            print("✓ pySnowClim functions imported successfully (development mode)")
        
        # Test parameter creation
        params = create_dict_parameters()
        print("✓ Parameter creation successful")
        
        print("\n🎉 Installation verification successful!")
        print("pySnowClim is ready to use.")
        
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    verify_installation() 
