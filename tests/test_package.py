"""Package initialization tests."""


def test_package_imports():
    """Test that all main modules can be imported without errors."""
    # Test main package import
    import agentic_lab

    # Verify main attributes exist
    assert hasattr(agentic_lab, "__version__")
    assert hasattr(agentic_lab, "__author__")
    assert hasattr(agentic_lab, "__description__")
