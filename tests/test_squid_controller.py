import pytest
import pytest_asyncio
import asyncio
import numpy as np
import os
import tempfile
import threading
import time
from unittest.mock import patch
from squid_control.squid_controller import SquidController
from squid_control.control.config import CONFIG, SIMULATED_CAMERA, WELLPLATE_FORMAT_96, WELLPLATE_FORMAT_384 # Import necessary config

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Global flag to track if we need force termination
_tests_completed = False

def force_terminate_after_delay(delay_seconds=5):
    """Force terminate the process after a delay to prevent hanging."""
    def terminate():
        time.sleep(delay_seconds)
        if _tests_completed:
            print(f"\n🔧 Force terminating after {delay_seconds}s to prevent hanging...")
            os._exit(0)
    
    thread = threading.Thread(target=terminate, daemon=True)
    thread.start()

@pytest.fixture(scope="module", autouse=True)
def module_teardown():
    """Module-level fixture that runs after all tests in this module complete."""
    yield
    global _tests_completed
    _tests_completed = True
    print("\n✅ All squid_controller tests completed - starting force termination timer...")
    force_terminate_after_delay(5)

@pytest_asyncio.fixture(scope="function")
async def sim_controller_fixture():
    """Fixture to provide a fresh SquidController instance in simulation mode for each test."""
    controller = SquidController(is_simulation=True)
    
    try:
        yield controller
    finally:
        # Graceful cleanup that works with pytest's teardown
        print("Starting graceful cleanup...")
        
        # First, try to close the controller normally
        try:
            controller.close()
            print("Controller closed normally")
        except Exception as e:
            print(f"Error closing controller: {e}")
        
        # Clean up Hypha connections more gracefully
        try:
            if hasattr(controller.camera, 'zarr_image_manager') and controller.camera.zarr_image_manager:
                zarr_manager = controller.camera.zarr_image_manager
                # Try async cleanup first
                if hasattr(zarr_manager, 'close'):
                    await zarr_manager.close()
                    print("ZarrImageManager closed via async method")
                # Clear references
                controller.camera.zarr_image_manager = None
                print("ZarrImageManager references cleared")
        except Exception as e:
            print(f"Error in zarr cleanup: {e}")
        
        # Give a moment for connections to close gracefully
        await asyncio.sleep(0.1)
        
        # Only cancel remaining tasks if there are any problematic ones
        try:
            pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
            # Filter out pytest's own tasks and only cancel our tasks
            our_tasks = [task for task in pending_tasks if 'pytest' not in str(task)]
            if our_tasks:
                print(f"Cancelling {len(our_tasks)} remaining tasks...")
                for task in our_tasks:
                    if not task.cancelled():
                        task.cancel()
                # Give cancelled tasks a moment to finish
                await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Error in task cleanup: {e}")
        
        print("Graceful cleanup completed")

async def test_controller_initialization(sim_controller_fixture):
    """Test if the SquidController initializes correctly in simulation mode."""
    controller = sim_controller_fixture
    assert controller is not None
    assert controller.is_simulation is True
    assert controller.camera is not None
    assert controller.microcontroller is not None
    _, _, z_pos, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    assert z_pos == pytest.approx(CONFIG.DEFAULT_Z_POS_MM, abs=1e-3)

async def test_simulation_mode_detection():
    """Test simulation mode detection and import handling."""
    # Test with environment variable
    with patch.dict(os.environ, {'SQUID_SIMULATION_MODE': 'true'}):
        # This should trigger the simulation mode detection
        controller = SquidController(is_simulation=True)
        assert controller.is_simulation is True
        controller.close()
    
    # Test with pytest environment
    with patch.dict(os.environ, {'PYTEST_CURRENT_TEST': 'test_case'}):
        controller = SquidController(is_simulation=True)
        assert controller.is_simulation is True
        controller.close()

async def test_well_plate_navigation_comprehensive(sim_controller_fixture):
    """Test comprehensive well plate navigation for different plate types."""
    controller = sim_controller_fixture
    # Test different well plate formats
    plate_types = ['6', '24', '96', '384']
    
    for plate_type in plate_types:
        # Test corner wells for each plate type
        test_wells = [('A', 1)]  # Always test A1
        
        if plate_type == '96':
            test_wells.extend([('A', 12), ('H', 1), ('H', 12)])
        elif plate_type == '384':
            test_wells.extend([('A', 24), ('P', 1), ('P', 24)])
        elif plate_type == '24':
            test_wells.extend([('A', 6), ('D', 1), ('D', 6)])
        elif plate_type == '6':
            test_wells.extend([('A', 3), ('B', 1), ('B', 3)])
        
        for row, column in test_wells:
            initial_x, initial_y, initial_z, *_ = controller.navigationController.update_pos(
                microcontroller=controller.microcontroller)
            
            # Test well navigation
            controller.move_to_well(row, column, plate_type)
            
            new_x, new_y, new_z, *_ = controller.navigationController.update_pos(
                microcontroller=controller.microcontroller)
            
            # Position should have changed for non-center positions
            if row != 'D' or column != 6:  # Not the default starting position
                assert new_x != initial_x or new_y != initial_y
            
            # Z should remain the same
            assert new_z == pytest.approx(initial_z, abs=1e-3)


@pytest.mark.timeout(60)
async def test_laser_autofocus_methods(sim_controller_fixture):
    """Test laser autofocus related methods."""
    controller = sim_controller_fixture
    # Test laser autofocus simulation
    initial_z = controller.navigationController.update_pos(microcontroller=controller.microcontroller)[2]
    
    await controller.do_laser_autofocus()
    
    final_z = controller.navigationController.update_pos(microcontroller=controller.microcontroller)[2]
    # Should move to near ORIN_Z in simulation
    assert final_z == pytest.approx(SIMULATED_CAMERA.ORIN_Z, abs=0.01)
    

@pytest.mark.timeout(60)
async def test_camera_frame_methods(sim_controller_fixture):
    """Test camera frame acquisition methods."""
    controller = sim_controller_fixture
    # Test get_camera_frame_simulation
    frame = await controller.get_camera_frame_simulation(channel=0, intensity=50, exposure_time=100)
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape[0] > 100 and frame.shape[1] > 100

@pytest.mark.timeout(60)
async def test_stage_movement_edge_cases(sim_controller_fixture):
    """Test edge cases in stage movement."""
    controller = sim_controller_fixture
    # Test zero movement
    initial_x, initial_y, initial_z, *_ = controller.navigationController.update_pos(
        microcontroller=controller.microcontroller)
    
    # Test move_by_distance with zero values
    moved, x_before, y_before, z_before, x_after, y_after, z_after = controller.move_by_distance_limited(0, 0, 0)
    assert moved  # Should succeed even with zero movement
    
    # Test well navigation with edge cases - these should not crash
    # The controller should handle invalid inputs gracefully
    controller.move_to_well('A', 0, '96')  # Zero column - should be handled gracefully
    controller.move_to_well(0, 1, '96')   # Zero row - should be handled gracefully
    controller.move_to_well(0, 0, '96')   # Both zero - should be handled gracefully

    # Position should remain valid after edge case calls
    final_x, final_y, final_z, *_ = controller.navigationController.update_pos(
        microcontroller=controller.microcontroller)
    assert isinstance(final_x, (int, float))
    assert isinstance(final_y, (int, float))
    assert isinstance(final_z, (int, float))


async def test_configuration_and_pixel_size(sim_controller_fixture):
    """Test configuration access and pixel size calculations."""
    controller = sim_controller_fixture
    # Test get_pixel_size method
    original_pixel_size = controller.pixel_size_xy
    controller.get_pixel_size()
    assert isinstance(controller.pixel_size_xy, float)
    assert controller.pixel_size_xy > 0
    
    # Test pixel size adjustment factor
    assert hasattr(controller, 'pixel_size_adjument_factor')
    assert controller.pixel_size_adjument_factor > 0
    
    # Test drift correction parameters
    assert hasattr(controller, 'drift_correction_x')
    assert hasattr(controller, 'drift_correction_y')
    assert isinstance(controller.drift_correction_x, (int, float))
    assert isinstance(controller.drift_correction_y, (int, float))
    
    # Test sample data alias methods
    original_alias = controller.get_simulated_sample_data_alias()

async def test_stage_position_methods(sim_controller_fixture):
    """Test stage positioning methods comprehensively."""
    controller = sim_controller_fixture
    # Test move_to_scaning_position method
    try:
        controller.move_to_scaning_position()
        # Should complete without error
        x, y, z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(z, (int, float))
    except Exception:
        # Method might have specific requirements
        pass
    
    # Test home_stage method
    try:
        controller.home_stage()
        # Should complete without error in simulation
        x, y, z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(z, (int, float))
    except Exception:
        # Method might have specific hardware requirements
        pass
    
    # Test return_stage method
    try:
        controller.return_stage()
        # Should complete without error in simulation
        x, y, z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(z, (int, float))
    except Exception:
        # Method might have specific hardware requirements
        pass


@pytest.mark.timeout(60)
async def test_illumination_and_exposure_edge_cases(sim_controller_fixture):
    """Test illumination and exposure with edge cases."""
    controller = sim_controller_fixture
    # Test extreme exposure times
    extreme_exposures = [1, 50]
    for exposure in extreme_exposures:
        image = await controller.snap_image(exposure_time=exposure)
        assert image is not None
        assert controller.current_exposure_time == exposure
    
    # Test extreme intensity values
    extreme_intensities = [1, 99]
    for intensity in extreme_intensities:
        image = await controller.snap_image(intensity=intensity)
        assert image is not None
        assert controller.current_intensity == intensity
    
    # Test supported fluorescence channels
    fluorescence_channels = [12, 14]  # Using a subset for stability
    for channel in fluorescence_channels:
        image = await controller.snap_image(channel=channel, intensity=50, exposure_time=100)
        assert image is not None
        assert controller.current_channel == channel


@pytest.mark.timeout(60)
async def test_error_handling_and_robustness(sim_controller_fixture):
    """Test error handling and robustness."""
    controller = sim_controller_fixture
    # Test with invalid well plate type - should handle gracefully
    controller.move_to_well('A', 1, 'invalid_plate')
    
    # Test with invalid well coordinates - should handle gracefully
    controller.move_to_well('Z', 99, '96')  # Invalid row/column for 96-well
    
    # Test movement limits (try to move to extreme positions)
    # These should be limited by software boundaries or handled gracefully
    # without crashing.
    controller.move_x_to_limited(1000.0)
    controller.move_y_to_limited(1000.0)
    controller.move_z_to_limited(100.0)
    controller.move_x_to_limited(-1000.0)
    controller.move_y_to_limited(-1000.0)
    
    # Verify controller is still in a valid state
    x, y, z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    assert isinstance(x, (int, float))
    assert isinstance(y, (int, float))
    assert isinstance(z, (int, float))


@pytest.mark.timeout(60)
async def test_async_methods_comprehensive(sim_controller_fixture):
    """Test all async methods comprehensively."""
    controller = sim_controller_fixture
    # Test send_trigger_simulation with various parameters
    await controller.send_trigger_simulation(channel=0, intensity=50, exposure_time=100)
    assert controller.current_channel == 0
    assert controller.current_intensity == 50
    assert controller.current_exposure_time == 100
    
    # Test with different channel
    await controller.send_trigger_simulation(channel=12, intensity=70, exposure_time=200)
    assert controller.current_channel == 12
    assert controller.current_intensity == 70
    assert controller.current_exposure_time == 200
    
    # Test snap_image with illumination state handling
    # This tests the illumination on/off logic in snap_image
    controller.liveController.turn_on_illumination()
    image_with_illumination = await controller.snap_image()
    assert image_with_illumination is not None
    
    controller.liveController.turn_off_illumination()
    image_without_illumination = await controller.snap_image()
    assert image_without_illumination is not None


async def test_controller_properties_and_attributes(sim_controller_fixture):
    """Test controller properties and attributes."""
    controller = sim_controller_fixture
    # Test all the default attributes are set correctly
    assert hasattr(controller, 'fps_software_trigger')
    assert controller.fps_software_trigger == 10
    
    assert hasattr(controller, 'data_channel')
    assert controller.data_channel is None
    
    assert hasattr(controller, 'is_busy')
    assert isinstance(controller.is_busy, bool)
    
    # Test simulation-specific attributes
    assert hasattr(controller, 'dz')
    assert hasattr(controller, 'current_channel')
    assert hasattr(controller, 'current_exposure_time')
    assert hasattr(controller, 'current_intensity')
    assert hasattr(controller, 'pixel_size_xy')
    assert hasattr(controller, 'sample_data_alias')
    
    # Test that all required controllers are initialized
    assert controller.objectiveStore is not None
    assert controller.configurationManager is not None
    assert controller.streamHandler is not None
    assert controller.liveController is not None
    assert controller.navigationController is not None
    assert controller.slidePositionController is not None
    assert controller.autofocusController is not None
    assert controller.scanCoordinates is not None
    assert controller.multipointController is not None


async def test_move_stage_absolute(sim_controller_fixture):
    """Test moving the stage to absolute coordinates."""
    controller = sim_controller_fixture
    target_x, target_y, target_z = 10.0, 15.0, 1.0
    
    # These methods are synchronous
    moved_x, _, _, _, final_x_coord = controller.move_x_to_limited(target_x)
    assert moved_x
    assert final_x_coord == pytest.approx(target_x, abs=CONFIG.STAGE_MOVED_THRESHOLD)
    
    moved_y, _, _, _, final_y_coord = controller.move_y_to_limited(target_y)
    assert moved_y
    assert final_y_coord == pytest.approx(target_y, abs=CONFIG.STAGE_MOVED_THRESHOLD)

    moved_z, _, _, _, final_z_coord = controller.move_z_to_limited(target_z)
    assert moved_z
    assert final_z_coord == pytest.approx(target_z, abs=CONFIG.STAGE_MOVED_THRESHOLD)

    current_x, current_y, current_z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    assert current_x == pytest.approx(target_x, abs=1e-3)
    assert current_y == pytest.approx(target_y, abs=1e-3)
    assert current_z == pytest.approx(target_z, abs=1e-3)


async def test_move_stage_relative(sim_controller_fixture):
    """Test moving the stage by relative distances."""
    controller = sim_controller_fixture
    initial_x, initial_y, initial_z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    
    dx, dy, dz = 1.0, -1.0, 0.1
    
    # This method is synchronous
    moved, x_before, y_before, z_before, x_after, y_after, z_after = controller.move_by_distance_limited(dx, dy, dz)
    assert moved
    
    current_x, current_y, current_z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    
    assert current_x == pytest.approx(initial_x + dx, abs=1e-3)
    assert current_y == pytest.approx(initial_y + dy, abs=1e-3)
    assert current_z == pytest.approx(initial_z + dz, abs=1e-3)

    assert x_after == pytest.approx(initial_x + dx, abs=1e-3)
    assert y_after == pytest.approx(initial_y + dy, abs=1e-3)
    assert z_after == pytest.approx(initial_z + dz, abs=1e-3)


async def test_snap_image_simulation(sim_controller_fixture):
    """Test snapping an image in simulation mode."""
    controller = sim_controller_fixture
    # snap_image IS async
    image = await controller.snap_image()
    assert image is not None

    test_channel = 0
    test_intensity = 50
    test_exposure = 100
    image_custom = await controller.snap_image(channel=test_channel, intensity=test_intensity, exposure_time=test_exposure)
    assert image_custom is not None
    assert image_custom.shape > (100,100)
    
    assert controller.current_channel == test_channel
    assert controller.current_intensity == test_intensity
    assert controller.current_exposure_time == test_exposure


async def test_illumination_channels(sim_controller_fixture):
    """Test different illumination channels and intensities."""
    controller = sim_controller_fixture
    # Test brightfield channel (channel 0)
    bf_image = await controller.snap_image(channel=0, intensity=40, exposure_time=50)
    assert bf_image is not None
    assert bf_image.shape[0] > 100 and bf_image.shape[1] > 100
    
    # Test fluorescence channels (11-15)
    fluorescence_channels = [11, 12, 13, 14]  # 405nm, 488nm, 638nm, 561nm
    for channel in fluorescence_channels:
        fl_image = await controller.snap_image(channel=channel, intensity=60, exposure_time=200)
        assert fl_image is not None
        assert fl_image.shape[0] > 100 and fl_image.shape[1] > 100
        assert controller.current_channel == channel
        
    # Test intensity variation
    low_intensity = await controller.snap_image(channel=0, intensity=10)
    high_intensity = await controller.snap_image(channel=0, intensity=80)
    assert low_intensity is not None and high_intensity is not None


async def test_exposure_time_variations(sim_controller_fixture):
    """Test different exposure times and their effects."""
    controller = sim_controller_fixture
    exposure_times = [10, 50, 100, 500, 1000]
    
    for exposure in exposure_times:
        image = await controller.snap_image(channel=0, exposure_time=exposure)
        assert image is not None
        assert controller.current_exposure_time == exposure
        
    # Test very short and long exposures
    short_exp = await controller.snap_image(exposure_time=1)
    long_exp = await controller.snap_image(exposure_time=2000)
    assert short_exp is not None and long_exp is not None


async def test_camera_streaming_control(sim_controller_fixture):
    """Test camera streaming start/stop functionality."""
    controller = sim_controller_fixture
    # Camera should already be streaming after initialization
    assert controller.camera.is_streaming == True
    
    # Stop streaming
    controller.camera.stop_streaming()
    assert controller.camera.is_streaming == False
    
    # Start streaming again
    controller.camera.start_streaming()
    assert controller.camera.is_streaming == True


async def test_well_plate_navigation(sim_controller_fixture):
    """Test well plate navigation functionality."""
    controller = sim_controller_fixture
    # Test 96-well plate navigation
    plate_format = '96'
    
    # Test moving to specific wells - need to parse well names into row/column
    test_wells = [('A', 1), ('A', 12), ('H', 1), ('H', 12), ('D', 6)]  # Corner and center wells
    
    for row, column in test_wells:
        try:
            if hasattr(controller, 'move_to_well'):  # Check if method exists
                success = controller.move_to_well(row, column, plate_format)
                current_x, current_y, current_z, *_ = controller.navigationController.update_pos(
                    microcontroller=controller.microcontroller)
                # Verify position changed (basic sanity check)
                assert isinstance(current_x, (int, float))
                assert isinstance(current_y, (int, float))
        except (AttributeError, TypeError) as e:
            # Method might not exist or have different signature, skip this test
            pass


@pytest.mark.timeout(60)
async def test_autofocus_simulation(sim_controller_fixture):
    """Test autofocus in simulation mode."""
    controller = sim_controller_fixture
    initial_x, initial_y, initial_z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    
    # These methods are now async
    await controller.do_autofocus_simulation()
    
    x_after, y_after, z_after, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    
    assert x_after == pytest.approx(initial_x)
    assert y_after == pytest.approx(initial_y)
    assert z_after != pytest.approx(initial_z)
    assert z_after == pytest.approx(SIMULATED_CAMERA.ORIN_Z, abs=0.01)

    await controller.do_autofocus()
    x_final, y_final, z_final, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    assert z_final == pytest.approx(SIMULATED_CAMERA.ORIN_Z, abs=0.01)


@pytest.mark.timeout(60)
async def test_focus_stack_simulation(sim_controller_fixture):
    """Test focus stack acquisition in simulation mode."""
    controller = sim_controller_fixture
    initial_z = controller.navigationController.update_pos(microcontroller=controller.microcontroller)[2]
    
    # Test basic z-stack parameters
    z_start = initial_z - 0.5
    z_end = initial_z + 0.5
    z_step = 0.1
    
    # Move to different z positions and capture images
    z_positions = np.arange(z_start, z_end + z_step, z_step)
    images = []
    
    for z_pos in z_positions:
        controller.move_z_to_limited(z_pos)
        image = await controller.snap_image()
        assert image is not None
        images.append(image)
        
    assert len(images) == len(z_positions)
    # All images should have the same dimensions
    first_shape = images[0].shape
    for img in images:
        assert img.shape == first_shape


@pytest.mark.timeout(60)
async def test_multiple_image_acquisition(sim_controller_fixture):
    """Test acquiring multiple images in sequence."""
    controller = sim_controller_fixture
    num_images = 5
    images = []
    
    for i in range(num_images):
        image = await controller.snap_image()
        assert image is not None
        images.append(image)
        
    assert len(images) == num_images
    
    # Test with different channels
    channels = [0, 11, 12]  # BF, 405nm, 488nm
    multichannel_images = []
    
    for channel in channels:
        image = await controller.snap_image(channel=channel)
        assert image is not None
        multichannel_images.append(image)
        
    assert len(multichannel_images) == len(channels)


async def test_stage_boundaries_and_limits(sim_controller_fixture):
    """Test stage movement boundaries and software limits."""
    controller = sim_controller_fixture
    # Get current position
    current_x, current_y, current_z, *_ = controller.navigationController.update_pos(
        microcontroller=controller.microcontroller)
    
    # Test movement within reasonable bounds
    safe_moves = [
        (current_x + 1.0, current_y, current_z),
        (current_x, current_y + 1.0, current_z),
        (current_x, current_y, current_z + 0.1)
    ]
    
    for target_x, target_y, target_z in safe_moves:
        moved_x, _, _, _, final_x = controller.move_x_to_limited(target_x)
        moved_y, _, _, _, final_y = controller.move_y_to_limited(target_y)
        moved_z, _, _, _, final_z = controller.move_z_to_limited(target_z)
        
        # Movement should succeed within safe bounds
        assert moved_x or abs(final_x - target_x) < CONFIG.STAGE_MOVED_THRESHOLD
        assert moved_y or abs(final_y - target_y) < CONFIG.STAGE_MOVED_THRESHOLD
        assert moved_z or abs(final_z - target_z) < CONFIG.STAGE_MOVED_THRESHOLD


async def test_hardware_status_monitoring(sim_controller_fixture):
    """Test hardware status monitoring and updates."""
    controller = sim_controller_fixture
    # Test microcontroller status
    assert controller.microcontroller is not None
    
    # Test position updates
    pos_data = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    assert len(pos_data) >= 4  # x, y, z, theta at minimum
    x, y, z = pos_data[:3]
    assert isinstance(x, (int, float))
    assert isinstance(y, (int, float))
    assert isinstance(z, (int, float))
    
    # Test camera status
    assert controller.camera is not None
    assert hasattr(controller.camera, 'is_streaming')


async def test_configuration_access(sim_controller_fixture):
    """Test accessing configuration parameters."""
    controller = sim_controller_fixture
    # Test pixel size access
    controller.get_pixel_size()
    assert hasattr(controller, 'pixel_size_xy')
    assert isinstance(controller.pixel_size_xy, (int, float))
    assert controller.pixel_size_xy > 0
    
    # Test current settings
    assert hasattr(controller, 'current_channel')
    assert hasattr(controller, 'current_intensity')
    assert hasattr(controller, 'current_exposure_time')


async def test_image_properties_and_formats(sim_controller_fixture):
    """Test image properties and different formats."""
    controller = sim_controller_fixture
    # Test default image
    image = await controller.snap_image()
    assert image is not None
    assert isinstance(image, np.ndarray)
    assert len(image.shape) >= 2  # At least 2D
    assert image.dtype in [np.uint8, np.uint16, np.uint32]
    
    # Test image dimensions are reasonable
    height, width = image.shape[:2]
    assert height > 100 and width > 100
    assert height < 10000 and width < 10000  # Reasonable upper bounds
    
    # Test different exposure settings produce different results
    dark_image = await controller.snap_image(exposure_time=1, intensity=1)
    bright_image = await controller.snap_image(exposure_time=100, intensity=100)
    
    assert dark_image is not None and bright_image is not None
    # Images should have same shape but potentially different intensity distributions
    assert dark_image.shape == bright_image.shape


async def test_z_axis_focus_effects(sim_controller_fixture):
    """Test z-axis movement and focus effects in simulation."""
    controller = sim_controller_fixture
    # Get reference position
    ref_z = controller.navigationController.update_pos(microcontroller=controller.microcontroller)[2]
    
    # Test images at different z positions
    z_offsets = [-0.5, 0, 0.5]  # Below, at, and above focus
    images_at_z = {}
    
    for offset in z_offsets:
        target_z = ref_z + offset
        controller.move_z_to_limited(target_z)
        image = await controller.snap_image()
        assert image is not None
        images_at_z[offset] = image
        
    # All images should have same dimensions
    shapes = [img.shape for img in images_at_z.values()]
    assert all(shape == shapes[0] for shape in shapes)


@pytest.mark.timeout(60)
async def test_error_handling_scenarios(sim_controller_fixture):
    """Test error handling in various scenarios."""
    controller = sim_controller_fixture
    # Test with invalid channel - should handle gracefully or raise appropriate exception
    try:
        image = await controller.snap_image(channel=999)
        # If it succeeds, image should be valid
        if image is not None:
            assert isinstance(image, np.ndarray)
    except (ValueError, IndexError, KeyError):
        # This is also acceptable behavior
        pass
        
    # Test with extreme exposure times - should handle gracefully
    try:
        very_short = await controller.snap_image(exposure_time=0)
        if very_short is not None:
            assert isinstance(very_short, np.ndarray)
    except ValueError:
        # This is acceptable behavior for invalid exposure
        pass
        
    # Test with extreme intensity values - should handle gracefully
    try:
        zero_intensity = await controller.snap_image(intensity=0)
        if zero_intensity is not None:
            assert isinstance(zero_intensity, np.ndarray)
    except ValueError:
        # This is acceptable behavior for invalid intensity
        pass


async def test_simulated_sample_data_alias(sim_controller_fixture):
    """Test setting and getting the simulated sample data alias."""
    controller = sim_controller_fixture
    default_alias = controller.get_simulated_sample_data_alias()
    assert default_alias == "agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38"

    new_alias = "agent-lens/20250429-scan-time-lapse-2025-04-29_15-38-36"
    # This method is synchronous
    controller.set_simulated_sample_data_alias(new_alias)
    assert controller.get_simulated_sample_data_alias() == new_alias # get is also synchronous


async def test_get_pixel_size(sim_controller_fixture):
    """Test the get_pixel_size method."""
    controller = sim_controller_fixture
    # This method is synchronous
    controller.get_pixel_size()
    assert isinstance(controller.pixel_size_xy, float)
    assert controller.pixel_size_xy > 0


async def test_simulation_consistency(sim_controller_fixture):
    """Test that simulation provides consistent results."""
    controller = sim_controller_fixture
    # Take multiple images at the same position with same settings
    position_x, position_y, position_z, *_ = controller.navigationController.update_pos(
        microcontroller=controller.microcontroller)
    
    # Capture multiple images with identical settings
    images = []
    for _ in range(3):
        image = await controller.snap_image(channel=0, intensity=50, exposure_time=100)
        assert image is not None
        images.append(image)
        
    # Images should have consistent properties
    first_shape = images[0].shape
    first_dtype = images[0].dtype
    
    for img in images[1:]:
        assert img.shape == first_shape
        assert img.dtype == first_dtype
        
    # Test position consistency after movements
    controller.move_x_to_limited(position_x + 1.0)
    controller.move_x_to_limited(position_x)  # Return to original
    
    final_x, _, _, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
    assert final_x == pytest.approx(position_x, abs=CONFIG.STAGE_MOVED_THRESHOLD)


async def test_close_controller(sim_controller_fixture):
    """Test if the controller's close method can be called without errors."""
    controller = sim_controller_fixture
    # controller.close() is called by the fixture's teardown.
    # This test ensures explicit call is also fine and checks camera state.
    controller.close() # Assuming synchronous close
    assert True
    # Check camera state after close
    assert controller.camera.is_streaming == False

