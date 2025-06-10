# Squid Microscope Control System - Test Suite

This directory contains comprehensive tests for the Squid microscope control system, covering both the core `SquidController` and the `Hypha service` components.

## Test Structure

### Test Files

1. **`test_squid_controller.py`** - Comprehensive tests for the SquidController class
   - Initialization and configuration tests
   - Stage movement and positioning tests
   - Image acquisition and camera control tests
   - Autofocus functionality tests
   - Well plate navigation tests
   - Simulation mode tests
   - Hardware integration tests
   - Multi-channel imaging tests
   - Service integration scenarios

2. **`test_hypha_service.py`** - Tests for the Hypha service layer
   - Service initialization and setup tests
   - API endpoint functionality tests
   - Task status management tests
   - Parameter management tests
   - Error handling tests
   - Permission checking tests

### Test Categories

Tests are marked with pytest markers to allow selective running:

- `@pytest.mark.simulation` - Tests that require simulation mode
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.local` - Tests that require local server setup
- `@pytest.mark.hardware` - Tests that require real hardware (currently skipped)

## Running Tests

### Prerequisites

1. **Install test dependencies:**
   ```bash
   pip install pytest pytest-asyncio pytest-cov pytest-timeout
   ```

2. **Environment setup:**
   ```bash
   # Set environment variables for local testing (optional)
   export HYPHA_TEST_LOCAL=1  # Enable local server tests
   ```

### Basic Test Execution

**Run all tests:**
```bash
pytest
```

**Run specific test file:**
```bash
pytest tests/test_squid_controller.py
pytest tests/test_hypha_service.py
```

**Run tests with specific markers:**
```bash
pytest -m simulation          # Only simulation tests
pytest -m "not slow"          # Exclude slow tests
pytest -m "simulation and not slow"  # Simulation tests that aren't slow
```

### Advanced Test Options

**Run with coverage:**
```bash
pytest --cov=squid_control --cov=start_hypha_service --cov-report=html
```

**Run with verbose output:**
```bash
pytest -v
```

**Run specific test function:**
```bash
pytest tests/test_squid_controller.py::test_controller_initialization
pytest tests/test_hypha_service.py::test_task_status_management -v
```

**Run tests in parallel (if pytest-xdist is installed):**
```bash
pip install pytest-xdist
pytest -n auto
```

**Run with timeout protection:**
```bash
pytest --timeout=300  # 5 minute timeout per test
```

### Continuous Integration

For CI/CD environments, use:
```bash
pytest --maxfail=5 --tb=short --strict-markers
```

## Test Configuration

### pytest.ini

The `pytest.ini` file contains default configuration:
- Asyncio mode set to auto for async test support
- Warning filters to reduce noise
- Verbose output and duration reporting
- Strict marker enforcement

### Environment Variables

Set these environment variables to customize test behavior:

- `HYPHA_TEST_LOCAL=1` - Enable tests that require local server setup
- `SQUID_TEST_HARDWARE=1` - Enable hardware tests (requires real microscope)
- `SQUID_TEST_TIMEOUT=600` - Set custom test timeout in seconds

## Test Development Guidelines

### Writing New Tests

1. **Use appropriate fixtures:**
   ```python
   async def test_my_feature(sim_controller_fixture):
       async for controller in sim_controller_fixture:
           # Your test code here
           break
   ```

2. **Mark tests appropriately:**
   ```python
   @pytest.mark.simulation
   @pytest.mark.slow
   async def test_long_running_simulation():
       pass
   ```

3. **Test error conditions:**
   ```python
   async def test_error_handling():
       with pytest.raises(ExpectedException):
           # Code that should raise exception
           pass
   ```

4. **Use mocking for external dependencies:**
   ```python
   @patch('external.dependency')
   async def test_with_mock(mock_dependency):
       mock_dependency.return_value = "test_value"
       # Test code
   ```

### Testing Best Practices

1. **Always test simulation mode first** - Follow the "SIMULATION FIRST" principle
2. **Use descriptive test names** that explain what is being tested
3. **Test both success and failure scenarios**
4. **Verify state changes** after operations
5. **Clean up resources** in fixtures and teardown
6. **Use appropriate assertions** with meaningful error messages

### Comprehensive Test Coverage

The `test_squid_controller.py` file provides comprehensive coverage including:

- **Core Functionality**: Initialization, configuration, and basic operations
- **Stage Control**: Movement, positioning, and well plate navigation
- **Image Acquisition**: Multi-channel imaging, exposure control, and camera management
- **Autofocus Systems**: Both contrast-based and reflection-based autofocus
- **Simulation Mode**: Complete simulation testing with virtual sample data
- **Error Handling**: Edge cases and error conditions
- **State Management**: Controller state and parameter synchronization

## Troubleshooting

### Common Issues

1. **Asyncio errors**: Ensure all async functions use `await` properly
2. **Fixture scope issues**: Use appropriate fixture scopes for resource sharing
3. **Import errors**: Ensure the squid_control package is in PYTHONPATH
4. **Configuration errors**: Check that configuration files are accessible
5. **Simulation timeouts**: Some tests may take longer in simulation mode

### Debug Mode

Run tests with debug output:
```bash
pytest -v -s --log-cli-level=DEBUG
```

### Skipping Tests

Skip specific tests temporarily:
```python
@pytest.mark.skip(reason="Under development")
async def test_new_feature():
    pass
```

### Test Data

Tests use simulated data with consistent parameters:
- Default sample data: `agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38`
- Pixel size: 0.333 micrometers
- Drift correction: X=-1.6, Y=-2.1
- Reference Z position: From SIMULATED_CAMERA.ORIN_Z

## Contributing

When contributing new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate documentation and comments
3. Ensure tests are deterministic and reproducible
4. Add tests for both positive and negative scenarios
5. Update this README if adding new test categories or requirements

## Monitoring and Reporting

### Coverage Reports

Generate HTML coverage reports:
```bash
pytest --cov=squid_control --cov=start_hypha_service --cov-report=html
open htmlcov/index.html
```

### Performance Testing

Monitor test performance with duration reporting:
```bash
pytest --durations=0  # Show all test durations
```

For questions about the test suite, refer to the main project documentation or contact the development team. 