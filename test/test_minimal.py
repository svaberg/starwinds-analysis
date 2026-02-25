def test_import():
    try:
        import pyvista as pv    
    except ImportError:
        assert False, "PyVista could not be imported"
