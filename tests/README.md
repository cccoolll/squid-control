# Squid Microscope Control System - Test Suite

This directory contains comprehensive tests for the Squid microscope control system, covering both the core `SquidController` and the `Hypha service` components, along with advanced features like WebRTC video streaming, Zarr data management, and artifact uploads.

## Test Structure

### Test Files

1. **`test_squid_controller.py`** (84KB, 1855 lines) - Comprehensive tests for the SquidController class
   - Initialization and configuration tests
   - Stage movement and positioning tests
   - Image acquisition and camera control tests
   - Autofocus functionality tests
   - Well plate navigation tests
   - Simulation mode tests
   - Hardware integration tests
   - Multi-channel imaging tests
   - Service integration scenarios
   - Stage velocity control tests
   - Plate scanning with custom illumination settings

2. **`test_hypha_service.py`** (88KB, 2119 lines) - Tests for the Hypha service layer
   - Service initialization and setup tests
   - API endpoint functionality tests
   - Task status management tests
   - Parameter management tests
   - Error handling tests
   - Permission checking tests
   - Video buffering functionality tests
   - Well location detection tests
   - Microscope configuration management tests
   - Comprehensive service lifecycle tests

3. **`test_webrtc_e2e.py`** (72KB, 1656 lines) - End-to-end WebRTC video streaming tests
   - WebRTC service registration and connectivity
   - Video track creation and management
   - Real-time video streaming functionality
   - Metadata transmission via data channels
   - Video frame processing and compression
   - Cross-platform WebRTC compatibility tests
   - Performance and latency measurements

4. **`test_zip_upload_endpoints.py`** (52KB, 1200 lines) - Zarr dataset upload and artifact management tests
   - Gallery and dataset creation tests
   - Zarr file upload functionality
   - Experiment data management
   - Artifact manager integration tests
   - Multi-well canvas upload tests
   - Dataset metadata and organization tests

5. **`test_connection.py`** (963B, 33 lines) - Basic connectivity tests
   - Hypha server connection validation
   - Authentication and token verification
   - Network connectivity checks

### Test Categories

Tests are marked with pytest markers to allow selective running:

- `@pytest.mark.simulation` - Tests that require simulation mode
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.local` - Tests that require local server setup
- `@pytest.mark.hardware` - Tests that require real hardware (currently skipped)
- `@pytest.mark.integration` - Tests that require network access and external services
- `@pytest.mark.unit` - Fast unit tests that don't require external dependencies
- `@pytest.mark.asyncio` - Async tests requiring asyncio event loop

## Running Tests

### Prerequisites

1. **Install test dependencies:**
   ```bash
   pip install pytest pytest-asyncio pytest-cov pytest-timeout pytest-xdist
   ```

2. **Environment setup:**
   ```bash
   # Set environment variables for integration testing (optional)
   export AGENT_LENS_WORKSPACE_TOKEN="your_token_here"  # For integration tests
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
pytest tests/test_webrtc_e2e.py
pytest tests/test_zip_upload_endpoints.py
```

**Run tests with specific markers:**
```bash
pytest -m simulation          # Only simulation tests
pytest -m "not slow"          # Exclude slow tests
pytest -m "simulation and not slow"  # Simulation tests that aren't slow
pytest -m unit                # Only unit tests
pytest -m integration         # Only integration tests (requires tokens)
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

**Run with performance profiling:**
```bash
pytest --durations=0  # Show all test durations
```

### Continuous Integration

For CI/CD environments, use:
```bash
pytest --maxfail=5 --tb=short --strict-markers
```

## Test Configuration

### pytest.ini

The `pytest.ini` file contains comprehensive configuration:
- **Asyncio Mode**: Strict mode for async test support
- **Test Discovery**: Automatic discovery of test files and functions
- **Markers**: Predefined markers for different test types
- **Output Options**: Verbose output, duration reporting, timeout protection
- **Warning Filters**: Suppress deprecation warnings and noise
- **Logging**: Configured logging levels and formats

### conftest.py

The `conftest.py` file provides:
- **Event Loop Management**: Proper asyncio event loop setup and cleanup
- **Task Cleanup**: Automatic cleanup of async tasks after each test
- **Integration Test Filtering**: Skip integration tests without required tokens
- **Fixture Management**: Shared fixtures across test modules

### Environment Variables

Set these environment variables to customize test behavior:

- `AGENT_LENS_WORKSPACE_TOKEN` - Required for integration tests with Hypha services
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
7. **Test async operations properly** with proper await statements
8. **Handle WebRTC and video streaming tests** with appropriate timeouts

### Comprehensive Test Coverage

The test suite provides comprehensive coverage including:

#### Core Functionality
- **Initialization**: Controller setup, configuration loading, simulation mode detection
- **Stage Control**: Movement, positioning, velocity control, well plate navigation
- **Image Acquisition**: Multi-channel imaging, exposure control, camera management
- **Autofocus Systems**: Both contrast-based and reflection-based autofocus

#### Advanced Features
- **WebRTC Video Streaming**: Real-time video transmission, metadata handling, performance testing
- **Zarr Data Management**: Canvas creation, image stitching, multi-scale pyramid support
- **Artifact Management**: Dataset uploads, gallery organization, experiment management
- **Service Integration**: Hypha RPC services, API endpoints, task status tracking

#### Simulation Mode
- **Virtual Hardware**: Complete simulation of microscope components
- **Zarr Image Manager**: Virtual sample data access and processing
- **Performance Testing**: Latency measurements, frame rate analysis
- **Error Simulation**: Hardware failure scenarios, network issues

#### Error Handling
- **Edge Cases**: Boundary conditions, invalid inputs, resource exhaustion
- **Network Issues**: Connection failures, timeout handling, retry logic
- **Hardware Failures**: Camera errors, stage movement failures, illumination issues
- **Service Failures**: API endpoint errors, authentication failures

## Troubleshooting

### Common Issues

1. **Asyncio errors**: Ensure all async functions use `await` properly
2. **Fixture scope issues**: Use appropriate fixture scopes for resource sharing
3. **Import errors**: Ensure the squid_control package is in PYTHONPATH
4. **Configuration errors**: Check that configuration files are accessible
5. **Simulation timeouts**: Some tests may take longer in simulation mode
6. **WebRTC issues**: Browser compatibility, network restrictions, firewall settings
7. **Integration test failures**: Missing tokens, network connectivity, service availability

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
- **Default sample data**: `agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38`
- **Pixel size**: 0.333 micrometers
- **Drift correction**: X=-1.6, Y=-2.1
- **Reference Z position**: From SIMULATED_CAMERA.ORIN_Z
- **WebRTC settings**: 5 FPS default, 750x750 frame size
- **Video buffering**: 5-frame buffer, 1-second idle timeout

## Performance Testing

### WebRTC Performance
- **Latency Measurement**: Frame acquisition and transmission timing
- **Bandwidth Analysis**: Compression ratios and data transfer rates
- **Memory Usage**: Buffer management and resource cleanup
- **Cross-Platform Compatibility**: Browser and device testing

### Service Performance
- **API Response Times**: Endpoint latency and throughput
- **Concurrent Operations**: Multi-user scenario testing
- **Resource Management**: Memory leaks and cleanup verification
- **Error Recovery**: Timeout and retry mechanism testing

## Contributing

When contributing new tests:

1. **Follow the existing test structure** and naming conventions
2. **Add appropriate documentation** and comments
3. **Ensure tests are deterministic** and reproducible
4. **Add tests for both positive and negative scenarios**
5. **Update this README** if adding new test categories or requirements
6. **Test both simulation and hardware code paths** where applicable
7. **Include performance benchmarks** for new features
8. **Add integration tests** for new service endpoints

## Monitoring and Reporting

### Coverage Reports

The project maintains comprehensive coverage reporting:

**Local Coverage Generation:**
```bash
# Terminal coverage report
pytest --cov=squid_control --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=squid_control --cov-report=html:htmlcov --cov-report=term-missing

# Generate XML coverage for integration tools
pytest --cov=squid_control --cov-report=xml:coverage.xml --cov-report=term-missing
```

### Coverage Thresholds

The project maintains coverage quality standards:
- **Minimum Acceptable**: 40% (Orange/Warning level)
- **Good Coverage**: 60% (Green/Passing level)
- **Current Coverage**: ~39% (as of last test run)

**Improving Coverage:**
- Focus on testing core functionality in `squid_control/control/`
- Add tests for edge cases and error conditions
- Test both simulation and hardware code paths
- Prioritize testing public APIs and critical business logic
- Include WebRTC and video streaming edge cases
- Test Zarr data management and artifact uploads

### Performance Monitoring

Monitor test performance with duration reporting:
```bash
pytest --durations=0  # Show all test durations
```

For questions about the test suite, refer to the main project documentation or contact the development team. 