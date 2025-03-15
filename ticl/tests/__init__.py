"""
TiCL Test Suite

The tests are organized into the following directories:

- baselines: Tests for baseline models and comparison methods
- core: Core functionality tests (dataloading, model interfaces, etc.)
- evaluation: Model evaluation and metrics tests
- models: Model architecture and component tests
- prediction: Tests for model prediction interfaces
- priors: Tests for prior distribution implementations
- semantic: Tests for semantic feature functionality
- training: Tests for model training workflows
- utils: Utility function tests

Run tests with pytest:
    python -m pytest            # Run all tests
    python -m pytest ticl/tests/semantic/  # Run only semantic tests
"""