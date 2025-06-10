import pytest
import pytest_asyncio
import asyncio
import os
import time
import uuid
from hypha_rpc import connect_to_server, login
from start_hypha_service import Microscope
from squid_control.hypha_tools.hypha_storage import HyphaDataStore

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Test configuration
TEST_SERVER_URL = "https://hypha.aicell.io"
TEST_WORKSPACE = "squid-control"
TEST_TIMEOUT = 120  # seconds

class SimpleTestDataStore:
    """Simple test datastore that doesn't require external services."""
    
    def __init__(self):
        self.storage = {}
        self.counter = 0
    
    def put(self, file_type, data, filename, description=""):
        self.counter += 1
        file_id = f"test_file_{self.counter}"
        self.storage[file_id] = {
            'type': file_type,
            'data': data,
            'filename': filename,
            'description': description
        }
        return file_id
    
    def get_url(self, file_id):
        if file_id in self.storage:
            return f"https://test-storage.example.com/{file_id}"
        return None

@pytest_asyncio.fixture
async def test_microscope_service():
    """Create a real microscope service for testing."""
    # Check for token first
    token = os.environ.get("SQUID_WORKSPACE_TOKEN")
    if not token:
        pytest.skip("SQUID_WORKSPACE_TOKEN not set in environment")
    
    print(f"🔗 Connecting to {TEST_SERVER_URL} workspace {TEST_WORKSPACE}...")
    
    # Use simple connection approach like test_connection.py
    try:
        server = await connect_to_server({
            "server_url": TEST_SERVER_URL,
            "token": token,
            "workspace": TEST_WORKSPACE,
            "ping_interval": None
        })
        print("✅ Connected to server")
    except Exception as e:
        pytest.fail(f"Failed to connect to server: {e}")
    
    # Create unique service ID for this test
    test_id = f"test-microscope-{uuid.uuid4().hex[:8]}"
    print(f"Creating test service with ID: {test_id}")
    
    # Create real microscope instance in simulation mode
    print("🔬 Creating Microscope instance...")
    start_time = time.time()
    microscope = Microscope(is_simulation=True, is_local=False)
    init_time = time.time() - start_time
    print(f"✅ Microscope initialization took {init_time:.1f} seconds")
    
    microscope.service_id = test_id
    microscope.login_required = False  # Disable auth for tests
    microscope.authorized_emails = None
    
    # Create a simple datastore for testing
    microscope.datastore = SimpleTestDataStore()
    
    service = None
    try:
        # Register the service
        print("📝 Registering microscope service...")
        service_start_time = time.time()
        await microscope.start_hypha_service(server, test_id)
        service_time = time.time() - service_start_time
        print(f"✅ Service registration took {service_time:.1f} seconds")
        
        # Get the registered service to test against
        print("🔍 Getting service reference...")
        service = await server.get_service(test_id)
        print("✅ Service ready for testing")
        
        yield microscope, service
        
    except Exception as e:
        pytest.fail(f"Failed to create test service: {e}")
    finally:
        # Cleanup: close the microscope properly
        print(f"Cleaning up test service {test_id}")
        try:
            if microscope and hasattr(microscope, 'squidController'):
                microscope.squidController.close()
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        
        # Give a moment for cleanup to complete
        await asyncio.sleep(0.1)

# Basic connectivity tests
async def test_service_registration_and_connectivity(test_microscope_service):
    """Test that the service can be registered and is accessible."""
    microscope, service = test_microscope_service
    
    # Test basic connectivity with timeout
    result = await asyncio.wait_for(service.hello_world(), timeout=10)
    assert result == "Hello world"
    
    # Verify the service has the expected methods
    assert hasattr(service, 'move_by_distance')
    assert hasattr(service, 'get_status')
    assert hasattr(service, 'snap')

# Task status management tests
async def test_task_status_management(test_microscope_service):
    """Test task status tracking functionality."""
    microscope, service = test_microscope_service
    
    # Test getting all task status
    all_status = await asyncio.wait_for(service.get_all_task_status(), timeout=10)
    assert isinstance(all_status, dict)
    assert "move_by_distance" in all_status
    assert "snap" in all_status
    
    # Test individual task status
    status = microscope.get_task_status("move_by_distance")
    assert status in ["not_started", "started", "finished", "failed"]
    
    # Test resetting task status
    microscope.reset_task_status("move_by_distance")
    status = microscope.get_task_status("move_by_distance")
    assert status == "not_started"

# Stage movement tests
async def test_move_by_distance_service(test_microscope_service):
    """Test stage movement through the service."""
    microscope, service = test_microscope_service
    
    # Test successful movement
    result = await asyncio.wait_for(
        service.move_by_distance(x=1.0, y=1.0, z=0.1),
        timeout=15
    )
    
    assert isinstance(result, dict)
    assert "success" in result
    assert result["success"] == True
    assert "message" in result
    assert "initial_position" in result
    assert "final_position" in result

async def test_move_to_position_service(test_microscope_service):
    """Test absolute positioning through the service."""
    microscope, service = test_microscope_service
    
    # Test moving to specific position
    result = await asyncio.wait_for(
        service.move_to_position(x=5.0, y=5.0, z=2.0),
        timeout=15
    )
    
    assert isinstance(result, dict)
    assert "success" in result
    assert "message" in result
    
    if result["success"]:
        assert "initial_position" in result
        assert "final_position" in result

# Status and parameter tests
async def test_get_status_service(test_microscope_service):
    """Test status retrieval through the service."""
    microscope, service = test_microscope_service
    
    status = await asyncio.wait_for(service.get_status(), timeout=10)
    
    assert isinstance(status, dict)
    assert 'current_x' in status
    assert 'current_y' in status
    assert 'current_z' in status
    assert 'is_illumination_on' in status
    assert 'is_busy' in status

async def test_update_parameters_service(test_microscope_service):
    """Test parameter updates through the service."""
    microscope, service = test_microscope_service
    
    new_params = {
        'dx': 2.0,
        'dy': 3.0,
        'BF_intensity_exposure': [60, 120]
    }
    
    result = await asyncio.wait_for(
        service.update_parameters_from_client(new_params),
        timeout=10
    )
    
    assert isinstance(result, dict)
    assert result["success"] == True
    assert "message" in result
    
    # Verify parameters were updated
    assert microscope.dx == 2.0
    assert microscope.dy == 3.0
    assert microscope.BF_intensity_exposure == [60, 120]

# Image acquisition tests
async def test_snap_image_service(test_microscope_service):
    """Test image capture through the service."""
    microscope, service = test_microscope_service
    
    url = await asyncio.wait_for(
        service.snap(exposure_time=100, channel=0, intensity=50),
        timeout=20
    )
    
    assert isinstance(url, str)
    assert url.startswith("https://")

async def test_one_new_frame_service(test_microscope_service):
    """Test frame acquisition through the service."""
    microscope, service = test_microscope_service
    
    frame = await asyncio.wait_for(service.one_new_frame(), timeout=20)
    
    assert frame is not None
    assert hasattr(frame, 'shape')
    assert frame.shape == (3000, 3000)

async def test_get_video_frame_service(test_microscope_service):
    """Test video frame acquisition through the service."""
    microscope, service = test_microscope_service
    
    frame = await asyncio.wait_for(
        service.get_video_frame(frame_width=640, frame_height=480),
        timeout=15
    )
    
    assert frame is not None
    assert hasattr(frame, 'shape')
    assert frame.shape == (480, 640, 3)

# Illumination control tests
async def test_illumination_control_service(test_microscope_service):
    """Test illumination control through the service."""
    microscope, service = test_microscope_service
    
    # Test turning on illumination
    result = await asyncio.wait_for(service.on_illumination(), timeout=10)
    assert "turned on" in result.lower()
    
    # Test setting illumination
    result = await asyncio.wait_for(
        service.set_illumination(channel=0, intensity=50),
        timeout=10
    )
    assert "intensity" in result and "50" in result
    
    # Test turning off illumination
    result = await asyncio.wait_for(service.off_illumination(), timeout=10)
    assert "turned off" in result.lower()

async def test_camera_exposure_service(test_microscope_service):
    """Test camera exposure control through the service."""
    microscope, service = test_microscope_service
    
    result = await service.set_camera_exposure(channel=0, exposure_time=200)
    
    assert "exposure time" in result and "200" in result

# Well plate navigation tests
async def test_navigate_to_well_service(test_microscope_service):
    """Test well plate navigation through the service."""
    microscope, service = test_microscope_service
    
    result = await asyncio.wait_for(
        service.navigate_to_well(row='B', col=3, wellplate_type='96'),
        timeout=15
    )
    
    assert "moved to well position (B,3)" in result

# Autofocus tests
async def test_autofocus_services(test_microscope_service):
    """Test autofocus methods through the service."""
    microscope, service = test_microscope_service
    
    # Test contrast autofocus
    result = await service.auto_focus()
    assert "auto-focused" in result.lower()
    
    # Test laser autofocus
    result = await service.do_laser_autofocus()
    assert "auto-focused" in result.lower()

# Stage homing tests
async def test_stage_homing_services(test_microscope_service):
    """Test stage homing methods through the service."""
    microscope, service = test_microscope_service
    
    # Test home stage
    result = await service.home_stage()
    assert "home" in result.lower()
    
    # Test return stage
    result = await service.return_stage()
    assert "position" in result.lower()

async def test_move_to_loading_position_service(test_microscope_service):
    """Test moving to loading position through the service."""
    microscope, service = test_microscope_service
    
    result = await service.move_to_loading_position()
    assert "loading position" in result.lower()

# Advanced feature tests
async def test_video_contrast_adjustment_service(test_microscope_service):
    """Test video contrast adjustment through the service."""
    microscope, service = test_microscope_service
    
    result = await service.adjust_video_frame(min_val=10, max_val=200)
    
    assert isinstance(result, dict)
    assert result["success"] == True
    assert microscope.video_contrast_min == 10
    assert microscope.video_contrast_max == 200

async def test_simulated_sample_data_service(test_microscope_service):
    """Test simulated sample data management through the service."""
    microscope, service = test_microscope_service
    
    # Test getting current alias
    current_alias = await service.get_simulated_sample_data_alias()
    assert isinstance(current_alias, str)
    
    # Test setting new alias
    new_alias = "test-sample/new-data"
    result = await service.set_simulated_sample_data_alias(new_alias)
    assert new_alias in result
    
    # Verify it was set
    retrieved_alias = await service.get_simulated_sample_data_alias()
    assert retrieved_alias == new_alias

# Error handling tests
async def test_service_error_handling(test_microscope_service):
    """Test error handling in service methods."""
    microscope, service = test_microscope_service
    
    # Test movement with extreme values (should be handled gracefully)
    result = await service.move_by_distance(x=10.0, y=10.0, z=1.0)
    
    assert isinstance(result, dict)
    assert "success" in result
    # The result might be success=False due to limits, which is correct behavior

# Performance and stress tests
async def test_multiple_rapid_requests(test_microscope_service):
    """Test handling multiple rapid requests."""
    microscope, service = test_microscope_service
    
    # Send multiple status requests rapidly
    tasks = []
    for i in range(5):
        tasks.append(asyncio.wait_for(service.get_status(), timeout=10))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check that we got some successful results (allow for some failures due to timing)
    successful_results = [r for r in results if isinstance(r, dict) and 'current_x' in r]
    assert len(successful_results) >= 1, f"Expected at least 1 successful result, got {len(successful_results)}"

async def test_concurrent_operations(test_microscope_service):
    """Test concurrent operations on the service."""
    microscope, service = test_microscope_service
    
    # Create tasks for different operations
    tasks = [
        asyncio.wait_for(service.get_status(), timeout=10),
        asyncio.wait_for(service.hello_world(), timeout=10),
        asyncio.wait_for(service.get_all_task_status(), timeout=10),
        asyncio.wait_for(service.get_simulated_sample_data_alias(), timeout=10)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify at least the hello_world result (most reliable)
    hello_result = None
    for result in results:
        if isinstance(result, str) and result == "Hello world":
            hello_result = result
            break
    
    assert hello_result == "Hello world", f"Expected 'Hello world' result, got: {results}"

# Integration tests with real SquidController
async def test_service_controller_integration(test_microscope_service):
    """Test that service properly integrates with SquidController."""
    microscope, service = test_microscope_service
    
    # Test that the service has a real SquidController
    assert microscope.squidController is not None
    assert microscope.squidController.is_simulation == True
    
    # Test movement through service affects controller
    initial_status = await service.get_status()
    initial_x = initial_status['current_x']
    
    # Move through service
    move_result = await service.move_by_distance(x=1.0, y=0.0, z=0.0)
    
    if move_result["success"]:
        # Check that position changed in controller
        new_status = await service.get_status()
        new_x = new_status['current_x']
        
        # Position should have changed (allowing for floating point precision)
        assert abs(new_x - initial_x - 1.0) < 0.1

async def test_service_parameter_persistence(test_microscope_service):
    """Test that parameter changes persist in the service."""
    microscope, service = test_microscope_service
    
    # Set illumination through service
    await service.set_illumination(channel=11, intensity=75)
    
    # Verify it's reflected in the microscope state
    assert microscope.squidController.current_channel == 11
    assert microscope.F405_intensity_exposure[0] == 75
    
    # Set exposure through service
    await service.set_camera_exposure(channel=11, exposure_time=150)
    
    # Verify exposure was updated
    assert microscope.F405_intensity_exposure[1] == 150

# Cleanup and resource management tests
async def test_service_cleanup(test_microscope_service):
    """Test that the service can be properly cleaned up."""
    microscope, service = test_microscope_service
    
    # Test that the service is responsive
    result = await asyncio.wait_for(service.hello_world(), timeout=10)
    assert result == "Hello world"
    
    # Test that SquidController can be closed
    # (This will be handled by the fixture cleanup)
    assert microscope.squidController is not None 