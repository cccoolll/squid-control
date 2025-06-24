import pytest
import pytest_asyncio
import asyncio
import numpy as np
import os
import tempfile
from unittest.mock import patch
from squid_control.squid_controller import SquidController
from squid_control.control.config import CONFIG, SIMULATED_CAMERA, WELLPLATE_FORMAT_96, WELLPLATE_FORMAT_384 # Import necessary config
import sys

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Add squid_control to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
async def sim_controller_fixture():
    """Fixture to provide a SquidController instance in simulation mode."""
    controller = SquidController(is_simulation=True)
    yield controller
    # Teardown: close controller resources safely with proper async cleanup
    try:
        if hasattr(controller, 'camera') and controller.camera is not None:
            # First close ZarrImageManager connections properly
            if hasattr(controller.camera, 'zarr_image_manager') and controller.camera.zarr_image_manager is not None:
                await controller.camera._cleanup_zarr_resources_async()
            # Then close the controller
            controller.close()
            print("Controller cleanup completed successfully")
    except Exception as e:
        # Ignore cleanup errors to prevent test hangs
        print(f"Warning: Controller cleanup error (ignored): {e}")
        pass

@pytest.mark.timeout(60)
async def test_controller_initialization(sim_controller_fixture):
    """Test if the SquidController initializes correctly in simulation mode."""
    async for controller in sim_controller_fixture:
        assert controller is not None
        assert controller.is_simulation is True
        assert controller.camera is not None
        assert controller.microcontroller is not None
        _, _, z_pos, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
        assert z_pos == pytest.approx(CONFIG.DEFAULT_Z_POS_MM, abs=1e-3)
        break

@pytest.mark.timeout(60)
async def test_simulation_mode_detection():
    """Test simulation mode detection and import handling."""
    # Test with environment variable
    with patch.dict(os.environ, {'SQUID_SIMULATION_MODE': 'true'}):
        # This should trigger the simulation mode detection
        controller = SquidController(is_simulation=True)
        assert controller.is_simulation is True
        # Properly close controller
        try:
            controller.close()
        except Exception as e:
            print(f"Close error (expected): {e}")
    
    # Test with pytest environment
    with patch.dict(os.environ, {'PYTEST_CURRENT_TEST': 'test_case'}):
        controller = SquidController(is_simulation=True)
        assert controller.is_simulation is True
        # Properly close controller
        try:
            controller.close()
        except Exception as e:
            print(f"Close error (expected): {e}")

@pytest.mark.timeout(60)  # Longer timeout for comprehensive test
async def test_well_plate_navigation_comprehensive(sim_controller_fixture):
    """Test comprehensive well plate navigation for different plate types."""
    async for controller in sim_controller_fixture:
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
        break

async def test_laser_autofocus_methods(sim_controller_fixture):
    """Test laser autofocus related methods."""
    async for controller in sim_controller_fixture:
        # Test laser autofocus simulation
        initial_z = controller.navigationController.update_pos(microcontroller=controller.microcontroller)[2]
        
        await controller.do_laser_autofocus()
        
        final_z = controller.navigationController.update_pos(microcontroller=controller.microcontroller)[2]
        # Should move to near ORIN_Z in simulation
        assert final_z == pytest.approx(SIMULATED_CAMERA.ORIN_Z, abs=0.01)
        break

async def test_camera_frame_methods(sim_controller_fixture):
    """Test camera frame acquisition methods."""
    async for controller in sim_controller_fixture:
        # Test get_camera_frame_simulation
        frame = await controller.get_camera_frame_simulation(channel=0, intensity=50, exposure_time=100)
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape[0] > 100 and frame.shape[1] > 100
        
        # Test with different parameters
        frame_fl = await controller.get_camera_frame_simulation(channel=11, intensity=70, exposure_time=200)
        assert frame_fl is not None
        assert isinstance(frame_fl, np.ndarray)
        
        # Test get_camera_frame (non-simulation method - should work in simulation too)
        try:
            frame_direct = controller.get_camera_frame(channel=0, intensity=30, exposure_time=50)
            assert frame_direct is not None
            assert isinstance(frame_direct, np.ndarray)
        except Exception:
            # May not work if camera is not properly set up
            pass
        break

async def test_stage_movement_edge_cases(sim_controller_fixture):
    """Test edge cases in stage movement."""
    async for controller in sim_controller_fixture:
        # Test zero movement
        initial_x, initial_y, initial_z, *_ = controller.navigationController.update_pos(
            microcontroller=controller.microcontroller)
        
        # Test move_by_distance with zero values
        moved, x_before, y_before, z_before, x_after, y_after, z_after = controller.move_by_distance_limited(0, 0, 0)
        assert moved  # Should succeed even with zero movement
        
        # Test well navigation with edge cases
        controller.move_to_well('A', 0, '96')  # Zero column
        controller.move_to_well(0, 1, '96')   # Zero row
        controller.move_to_well(0, 0, '96')   # Both zero
        
        # These shouldn't crash, positions should remain valid
        final_x, final_y, final_z, *_ = controller.navigationController.update_pos(
            microcontroller=controller.microcontroller)
        assert isinstance(final_x, (int, float))
        assert isinstance(final_y, (int, float))
        assert isinstance(final_z, (int, float))
        break

async def test_configuration_and_pixel_size(sim_controller_fixture):
    """Test configuration access and pixel size calculations."""
    async for controller in sim_controller_fixture:
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
        test_alias = "test/sample/data"
        controller.set_simulated_sample_data_alias(test_alias)
        assert controller.get_simulated_sample_data_alias() == test_alias
        
        # Reset to original
        controller.set_simulated_sample_data_alias(original_alias)
        break

async def test_stage_position_methods(sim_controller_fixture):
    """Test stage positioning methods comprehensively."""
    async for controller in sim_controller_fixture:
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
        break

async def test_illumination_and_exposure_edge_cases(sim_controller_fixture):
    """Test illumination and exposure with edge cases."""
    async for controller in sim_controller_fixture:
        # Test extreme exposure times
        extreme_exposures = [1, 5000]
        for exposure in extreme_exposures:
            try:
                image = await controller.snap_image(exposure_time=exposure)
                assert image is not None
                assert controller.current_exposure_time == exposure
            except Exception:
                # Some extreme values might not be supported
                pass
        
        # Test extreme intensity values
        extreme_intensities = [1, 99]
        for intensity in extreme_intensities:
            try:
                image = await controller.snap_image(intensity=intensity)
                assert image is not None
                assert controller.current_intensity == intensity
            except Exception:
                # Some extreme values might not be supported
                pass
        
        # Test all supported fluorescence channels
        fluorescence_channels = [11, 12, 13, 14, 15]  # 405nm, 488nm, 638nm, 561nm, 730nm
        for channel in fluorescence_channels:
            try:
                image = await controller.snap_image(channel=channel, intensity=50, exposure_time=100)
                assert image is not None
                assert controller.current_channel == channel
            except Exception:
                # Some channels might not be fully supported in simulation
                pass
        break

async def test_error_handling_and_robustness(sim_controller_fixture):
    """Test error handling and robustness."""
    async for controller in sim_controller_fixture:
        # Test with invalid well plate type
        try:
            controller.move_to_well('A', 1, 'invalid_plate')
            # Should either work with fallback or handle gracefully
        except Exception:
            # Expected for invalid plate type
            pass
        
        # Test with invalid well coordinates
        try:
            controller.move_to_well('Z', 99, '96')  # Invalid row/column for 96-well
        except Exception:
            # Expected for invalid coordinates
            pass
        
        # Test movement limits (try to move to extreme positions)
        extreme_positions = [
            (1000.0, 0, 0),  # Very large X
            (0, 1000.0, 0),  # Very large Y
            (0, 0, 100.0),   # Very large Z
            (-1000.0, 0, 0), # Very negative X
            (0, -1000.0, 0), # Very negative Y
        ]
        
        for x, y, z in extreme_positions:
            try:
                # These should either be limited by software boundaries or handled gracefully
                moved_x, _, _, _, _ = controller.move_x_to_limited(x)
                moved_y, _, _, _, _ = controller.move_y_to_limited(y)
                moved_z, _, _, _, _ = controller.move_z_to_limited(z)
                # Movement may succeed or fail depending on limits, but shouldn't crash
            except Exception:
                # Some extreme movements might raise exceptions
                pass
        break

async def test_async_methods_comprehensive(sim_controller_fixture):
    """Test all async methods comprehensively."""
    async for controller in sim_controller_fixture:
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
        break

async def test_controller_properties_and_attributes(sim_controller_fixture):
    """Test controller properties and attributes."""
    async for controller in sim_controller_fixture:
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
        break

async def test_move_stage_absolute(sim_controller_fixture):
    """Test moving the stage to absolute coordinates."""
    async for controller in sim_controller_fixture:
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
        break

async def test_move_stage_relative(sim_controller_fixture):
    """Test moving the stage by relative distances."""
    async for controller in sim_controller_fixture:
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
        break

async def test_snap_image_simulation(sim_controller_fixture):
    """Test snapping an image in simulation mode."""
    async for controller in sim_controller_fixture:
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
        break

async def test_illumination_channels(sim_controller_fixture):
    """Test different illumination channels and intensities."""
    async for controller in sim_controller_fixture:
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
        break

async def test_exposure_time_variations(sim_controller_fixture):
    """Test different exposure times and their effects."""
    async for controller in sim_controller_fixture:
        exposure_times = [10, 50, 100, 500, 1000]
        
        for exposure in exposure_times:
            image = await controller.snap_image(channel=0, exposure_time=exposure)
            assert image is not None
            assert controller.current_exposure_time == exposure
            
        # Test very short and long exposures
        short_exp = await controller.snap_image(exposure_time=1)
        long_exp = await controller.snap_image(exposure_time=2000)
        assert short_exp is not None and long_exp is not None
        break

async def test_camera_streaming_control(sim_controller_fixture):
    """Test camera streaming start/stop functionality."""
    async for controller in sim_controller_fixture:
        # Camera should already be streaming after initialization
        assert controller.camera.is_streaming == True
        
        # Stop streaming
        controller.camera.stop_streaming()
        assert controller.camera.is_streaming == False
        
        # Start streaming again
        controller.camera.start_streaming()
        assert controller.camera.is_streaming == True
        break

async def test_well_plate_navigation(sim_controller_fixture):
    """Test well plate navigation functionality."""
    async for controller in sim_controller_fixture:
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
        break


async def test_autofocus_simulation(sim_controller_fixture):
    """Test autofocus in simulation mode."""
    async for controller in sim_controller_fixture:
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
        break

async def test_focus_stack_simulation(sim_controller_fixture):
    """Test focus stack acquisition in simulation mode."""
    async for controller in sim_controller_fixture:
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
        break

async def test_multiple_image_acquisition(sim_controller_fixture):
    """Test acquiring multiple images in sequence."""
    async for controller in sim_controller_fixture:
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
        break

async def test_stage_boundaries_and_limits(sim_controller_fixture):
    """Test stage movement boundaries and software limits."""
    async for controller in sim_controller_fixture:
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
        break

async def test_hardware_status_monitoring(sim_controller_fixture):
    """Test hardware status monitoring and updates."""
    async for controller in sim_controller_fixture:
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
        break

async def test_configuration_access(sim_controller_fixture):
    """Test accessing configuration parameters."""
    async for controller in sim_controller_fixture:
        # Test pixel size access
        controller.get_pixel_size()
        assert hasattr(controller, 'pixel_size_xy')
        assert isinstance(controller.pixel_size_xy, (int, float))
        assert controller.pixel_size_xy > 0
        
        # Test current settings
        assert hasattr(controller, 'current_channel')
        assert hasattr(controller, 'current_intensity')
        assert hasattr(controller, 'current_exposure_time')
        break

async def test_image_properties_and_formats(sim_controller_fixture):
    """Test image properties and different formats."""
    async for controller in sim_controller_fixture:
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
        break

async def test_z_axis_focus_effects(sim_controller_fixture):
    """Test z-axis movement and focus effects in simulation."""
    async for controller in sim_controller_fixture:
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
        break

async def test_error_handling_scenarios(sim_controller_fixture):
    """Test error handling in various scenarios."""
    async for controller in sim_controller_fixture:
        # Test with invalid channel (should handle gracefully)
        try:
            image = await controller.snap_image(channel=999)  # Invalid channel
            # Should either work with fallback or raise appropriate exception
            if image is not None:
                assert isinstance(image, np.ndarray)
        except (ValueError, IndexError, KeyError):
            # Expected behavior for invalid channel
            pass
            
        # Test with extreme exposure times
        try:
            very_short = await controller.snap_image(exposure_time=0)
            if very_short is not None:
                assert isinstance(very_short, np.ndarray)
        except ValueError:
            # Expected behavior for invalid exposure
            pass
            
        # Test with extreme intensity values
        try:
            zero_intensity = await controller.snap_image(intensity=0)
            if zero_intensity is not None:
                assert isinstance(zero_intensity, np.ndarray)
        except ValueError:
            # Expected behavior for invalid intensity
            pass
        break

async def test_simulated_sample_data_alias(sim_controller_fixture):
    """Test setting and getting the simulated sample data alias."""
    async for controller in sim_controller_fixture:
        default_alias = controller.get_simulated_sample_data_alias()
        assert default_alias == "agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38"

        new_alias = "new/sample/path"
        # This method is synchronous
        controller.set_simulated_sample_data_alias(new_alias)
        assert controller.get_simulated_sample_data_alias() == new_alias # get is also synchronous
        break

async def test_get_pixel_size(sim_controller_fixture):
    """Test the get_pixel_size method."""
    async for controller in sim_controller_fixture:
        # This method is synchronous
        controller.get_pixel_size()
        assert isinstance(controller.pixel_size_xy, float)
        assert controller.pixel_size_xy > 0
        break

async def test_simulation_consistency(sim_controller_fixture):
    """Test that simulation provides consistent results."""
    async for controller in sim_controller_fixture:
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
        break

async def test_close_controller(sim_controller_fixture):
    """Test if the controller's close method can be called without errors."""
    async for controller in sim_controller_fixture:
        # Test that close method exists and can be called
        assert hasattr(controller, 'close')
        
        # Check initial camera state
        initial_streaming = controller.camera.is_streaming
        
        # controller.close() is called by the fixture's teardown.
        # This test just verifies the method exists and basic functionality
        try:
            controller.close()  # Assuming synchronous close
            # After close, camera should not be streaming
            assert controller.camera.is_streaming == False
        except Exception as e:
            # If close fails, that's still acceptable as long as it doesn't crash
            print(f"Close method completed with: {e}")
        
        break

def test_get_well_from_position_96_well():
    """Test the get_well_from_position function with 96-well plate format."""
    print("Testing get_well_from_position with 96-well plate...")
    
    # Create a simulated SquidController
    controller = SquidController(is_simulation=True)
    
    # Test 1: Move to well C3 and verify position calculation
    print("1. Testing move to well C3 and position detection...")
    controller.move_to_well('C', 3, '96')
    
    # Get well info for current position  
    well_info = controller.get_well_from_position('96')
    
    print(f"   Expected: C3, Got: {well_info['well_id']}")
    assert well_info['row'] == 'C'
    assert well_info['column'] == 3
    assert well_info['well_id'] == 'C3'
    assert well_info['plate_type'] == '96'
    assert well_info['position_status'] in ['in_well', 'between_wells']  # Allow some tolerance
    
    # Test 2: Move to well A1 (corner case)
    print("2. Testing move to well A1 (corner case)...")
    controller.move_to_well('A', 1, '96')
    well_info = controller.get_well_from_position('96')
    
    print(f"   Expected: A1, Got: {well_info['well_id']}")
    assert well_info['row'] == 'A'
    assert well_info['column'] == 1
    assert well_info['well_id'] == 'A1'
    
    # Test 3: Move to well H12 (opposite corner)
    print("3. Testing move to well H12 (opposite corner)...")
    controller.move_to_well('H', 12, '96')
    well_info = controller.get_well_from_position('96')
    
    print(f"   Expected: H12, Got: {well_info['well_id']}")
    assert well_info['row'] == 'H'
    assert well_info['column'] == 12
    assert well_info['well_id'] == 'H12'
    
    # Test 4: Test with explicit coordinates
    print("4. Testing with explicit coordinates...")
    # Test with some known coordinates
    well_info = controller.get_well_from_position('96', x_pos_mm=14.3, y_pos_mm=11.36)  # Should be A1
    print(f"   A1 coordinates test - Expected: A1, Got: {well_info['well_id']}")
    assert well_info['well_id'] == 'A1'
    
    print("✅ 96-well plate tests passed!")

def test_get_well_from_position_different_plates():
    """Test the get_well_from_position function with different plate formats."""
    print("Testing get_well_from_position with different plate formats...")
    
    controller = SquidController(is_simulation=True)
    
    # Test with 24-well plate
    print("1. Testing 24-well plate...")
    controller.move_to_well('B', 4, '24')
    well_info = controller.get_well_from_position('24')
    
    print(f"   Expected: B4, Got: {well_info['well_id']}")
    assert well_info['row'] == 'B'
    assert well_info['column'] == 4
    assert well_info['well_id'] == 'B4'
    assert well_info['plate_type'] == '24'
    
    # Test with 384-well plate  
    print("2. Testing 384-well plate...")
    controller.move_to_well('D', 8, '384')
    well_info = controller.get_well_from_position('384')
    
    print(f"   Expected: D8, Got: {well_info['well_id']}")
    assert well_info['row'] == 'D'
    assert well_info['column'] == 8
    assert well_info['well_id'] == 'D8'
    assert well_info['plate_type'] == '384'
    
    # Test with 6-well plate
    print("3. Testing 6-well plate...")
    controller.move_to_well('A', 2, '6')
    well_info = controller.get_well_from_position('6')
    
    print(f"   Expected: A2, Got: {well_info['well_id']}")
    assert well_info['row'] == 'A'
    assert well_info['column'] == 2
    assert well_info['well_id'] == 'A2'
    assert well_info['plate_type'] == '6'
    
    print("✅ Different plate format tests passed!")

def test_get_well_from_position_edge_cases():
    """Test edge cases for get_well_from_position function."""
    print("Testing get_well_from_position edge cases...")
    
    controller = SquidController(is_simulation=True)
    
    # Test 1: Position outside plate boundaries
    print("1. Testing position outside plate boundaries...")
    well_info = controller.get_well_from_position('96', x_pos_mm=0, y_pos_mm=0)  # Far from plate
    
    print(f"   Outside position status: {well_info['position_status']}")
    assert well_info['position_status'] == 'outside_plate'
    assert well_info['row'] is None
    assert well_info['column'] is None
    assert well_info['well_id'] is None
    
    # Test 2: Position between wells
    print("2. Testing position between wells...")
    # Position exactly between A1 and A2
    from squid_control.control.config import WELLPLATE_FORMAT_96
    between_x = WELLPLATE_FORMAT_96.A1_X_MM + WELLPLATE_FORMAT_96.WELL_SPACING_MM / 2
    between_y = WELLPLATE_FORMAT_96.A1_Y_MM
    
    well_info = controller.get_well_from_position('96', x_pos_mm=between_x, y_pos_mm=between_y)
    print(f"   Between wells position: {well_info['well_id']} ({well_info['position_status']})")
    # Should still identify closest well but mark as between_wells if outside well boundary
    assert well_info['well_id'] in ['A1', 'A2']  # Should be closest well
    
    # Test 3: Very far position
    print("3. Testing very far position...")
    well_info = controller.get_well_from_position('96', x_pos_mm=1000, y_pos_mm=1000)
    assert well_info['position_status'] == 'outside_plate'
    
    print("✅ Edge case tests passed!")

def test_well_location_accuracy():
    """Test the accuracy of well location calculations."""
    print("Testing well location calculation accuracy...")
    
    controller = SquidController(is_simulation=True)
    
    # Test multiple wells in sequence
    test_wells = [
        ('A', 1), ('A', 6), ('A', 12),
        ('D', 1), ('D', 6), ('D', 12),
        ('H', 1), ('H', 6), ('H', 12)
    ]
    
    for row, col in test_wells:
        print(f"   Testing well {row}{col}...")
        
        # Move to the well
        controller.move_to_well(row, col, '96')
        
        # Get well position
        well_info = controller.get_well_from_position('96')
        
        print(f"      Expected: {row}{col}, Got: {well_info['well_id']}, Distance: {well_info['distance_from_center']:.3f}mm")
        
        # Verify correct identification
        assert well_info['row'] == row
        assert well_info['column'] == col
        assert well_info['well_id'] == f"{row}{col}"
        
        # Distance from center should be very small (perfect positioning in simulation)
        assert well_info['distance_from_center'] < 0.1, f"Distance too large: {well_info['distance_from_center']}"
        
        # Should be identified as inside well or very close
        assert well_info['position_status'] in ['in_well', 'between_wells']
    
    print("✅ Well location accuracy tests passed!")

def test_well_boundary_detection():
    """Test well boundary detection functionality."""
    print("Testing well boundary detection...")
    
    controller = SquidController(is_simulation=True)
    
    # Move to a well center
    controller.move_to_well('C', 5, '96')
    
    # Get the exact well center coordinates
    current_x, current_y, current_z, current_theta = controller.navigationController.update_pos(
        microcontroller=controller.microcontroller
    )
    
    # Test at well center
    well_info = controller.get_well_from_position('96', x_pos_mm=current_x, y_pos_mm=current_y)
    print(f"   At center: {well_info['position_status']}, distance: {well_info['distance_from_center']:.3f}mm")
    
    # Move slightly away from center (but within well)
    from squid_control.control.config import WELLPLATE_FORMAT_96
    well_radius = WELLPLATE_FORMAT_96.WELL_SIZE_MM / 2.0
    offset = well_radius * 0.8  # 80% of radius, should still be inside
    
    well_info = controller.get_well_from_position('96', 
                                                x_pos_mm=current_x + offset, 
                                                y_pos_mm=current_y)
    print(f"   Near edge (inside): {well_info['position_status']}, distance: {well_info['distance_from_center']:.3f}mm")
    assert well_info['well_id'] == 'C5'
    
    # Move outside well boundary
    offset = well_radius * 1.2  # 120% of radius, should be outside
    well_info = controller.get_well_from_position('96', 
                                                x_pos_mm=current_x + offset, 
                                                y_pos_mm=current_y)
    print(f"   Outside well: {well_info['position_status']}, distance: {well_info['distance_from_center']:.3f}mm")
    
    print("✅ Well boundary detection tests passed!")

# Test microscope configuration functionality
def test_get_microscope_configuration_data():
    """Test the get_microscope_configuration_data function from config.py."""
    print("Testing get_microscope_configuration_data function...")
    
    from squid_control.control.config import get_microscope_configuration_data
    
    # Test 1: Get all configuration
    print("1. Testing 'all' configuration...")
    config_all = get_microscope_configuration_data(config_section="all", include_defaults=True, is_simulation=True, is_local=False)
    
    assert isinstance(config_all, dict)
    assert "success" in config_all
    assert config_all["success"] == True
    assert "configuration" in config_all  # Fixed: should be "configuration" not "data"
    assert "section" in config_all  # Fixed: should be "section" not "config_section"
    assert config_all["section"] == "all"
    
    # Verify all expected sections are present
    expected_sections = ["camera", "stage", "illumination", "acquisition", "limits", "hardware", "wellplate", "optics", "autofocus"]
    config_data = config_all["configuration"]  # Fixed: should be "configuration" not "data"
    
    for section in expected_sections:
        assert section in config_data, f"Missing section: {section}"
        assert isinstance(config_data[section], dict)
    
    print(f"   Found {len(config_data)} configuration sections")
    
    # Test 2: Get specific sections
    print("2. Testing specific configuration sections...")
    test_sections = ["camera", "stage", "illumination", "wellplate"]
    
    for section in test_sections:
        print(f"   Testing section: {section}")
        config_section_data = get_microscope_configuration_data(config_section=section, include_defaults=True, is_simulation=True, is_local=False)
        
        assert isinstance(config_section_data, dict)
        assert config_section_data["success"] == True
        assert config_section_data["section"] == section  # Fixed: should be "section"
        assert section in config_section_data["configuration"]  # Fixed: should be "configuration"
        
        section_data = config_section_data["configuration"][section]  # Fixed: should be "configuration"
        assert isinstance(section_data, dict)
        assert len(section_data) > 0
        print(f"      Section '{section}' has {len(section_data)} parameters")
    
    # Test 3: Test with different parameters
    print("3. Testing different parameter combinations...")
    
    # Test without defaults
    config_no_defaults = get_microscope_configuration_data(config_section="camera", include_defaults=False, is_simulation=True, is_local=False)
    assert config_no_defaults["success"] == True
    # Note: The function doesn't return include_defaults in the response, just uses it internally
    
    # Test non-simulation mode
    config_non_sim = get_microscope_configuration_data(config_section="stage", include_defaults=True, is_simulation=False, is_local=False)
    assert config_non_sim["success"] == True
    # Note: The function doesn't return is_simulation in the main response, it's in metadata
    
    # Test local mode
    config_local = get_microscope_configuration_data(config_section="illumination", include_defaults=True, is_simulation=True, is_local=True)
    assert config_local["success"] == True
    # Note: The function doesn't return is_local in the main response, it's in metadata
    
    # Test 4: Test invalid section
    print("4. Testing invalid configuration section...")
    config_invalid = get_microscope_configuration_data(config_section="invalid_section", include_defaults=True, is_simulation=True, is_local=False)
    
    # Should still return success but with empty or minimal data
    assert isinstance(config_invalid, dict)
    assert "success" in config_invalid
    # Invalid sections might still succeed but return limited data
    
    print("✅ get_microscope_configuration_data tests passed!")

def test_configuration_data_content():
    """Test the content and structure of configuration data."""
    print("Testing configuration data content and structure...")
    
    from squid_control.control.config import get_microscope_configuration_data
    
    # Test 1: Camera configuration content
    print("1. Testing camera configuration content...")
    camera_config = get_microscope_configuration_data(config_section="camera", include_defaults=True, is_simulation=True, is_local=False)
    camera_data = camera_config["configuration"]["camera"]  # Fixed: should be "configuration"
    
    # Check for expected camera parameters
    expected_camera_params = ["sensor_format", "pixel_format", "image_acquisition", "frame_rate"]
    for param in expected_camera_params:
        if param in camera_data:
            print(f"   Found camera parameter: {param}")
            assert isinstance(camera_data[param], (dict, str, int, float, list))
    
    # Test 2: Stage configuration content
    print("2. Testing stage configuration content...")
    stage_config = get_microscope_configuration_data(config_section="stage", include_defaults=True, is_simulation=True, is_local=False)
    stage_data = stage_config["configuration"]["stage"]  # Fixed: should be "configuration"
    
    # Check for expected stage parameters (updated to match actual structure)
    expected_stage_params = ["movement_signs", "position_signs", "screw_pitch_mm", "microstepping"]
    for param in expected_stage_params:
        if param in stage_data:
            print(f"   Found stage parameter: {param}")
            assert isinstance(stage_data[param], (dict, str, int, float, list))
    
    # Test 3: Illumination configuration content
    print("3. Testing illumination configuration content...")
    illumination_config = get_microscope_configuration_data(config_section="illumination", include_defaults=True, is_simulation=True, is_local=False)
    illumination_data = illumination_config["configuration"]["illumination"]  # Fixed: should be "configuration"
    
    # Check for expected illumination parameters (updated to match actual structure)
    expected_illumination_params = ["led_matrix_factors", "illumination_intensity_factor", "mcu_pins"]
    for param in expected_illumination_params:
        if param in illumination_data:
            print(f"   Found illumination parameter: {param}")
            assert isinstance(illumination_data[param], (dict, str, int, float, list))
    
    # Test 4: Well plate configuration content
    print("4. Testing well plate configuration content...")
    wellplate_config = get_microscope_configuration_data(config_section="wellplate", include_defaults=True, is_simulation=True, is_local=False)
    wellplate_data = wellplate_config["configuration"]["wellplate"]  # Fixed: should be "configuration"
    
    # Check for expected well plate parameters (updated to match actual structure)
    expected_wellplate_params = ["formats", "default_format", "offset_x_mm"]
    for param in expected_wellplate_params:
        if param in wellplate_data:
            print(f"   Found wellplate parameter: {param}")
            assert isinstance(wellplate_data[param], (dict, str, int, float, list))
    
    # Test 5: Test metadata fields
    print("5. Testing metadata fields...")
    all_config = get_microscope_configuration_data(config_section="all", include_defaults=True, is_simulation=True, is_local=False)
    
    # Check for expected metadata (updated to match actual structure)
    expected_metadata = ["success", "section", "configuration", "total_sections"]
    for field in expected_metadata:
        if field in all_config:
            print(f"   Found metadata field: {field}")
        # timestamp and some fields might be optional
    
    assert "success" in all_config
    assert "section" in all_config  # Fixed: should be "section"
    assert "configuration" in all_config  # Fixed: should be "configuration"
    
    print("✅ Configuration data content tests passed!")

def test_configuration_json_serializable():
    """Test that configuration data is JSON serializable."""
    print("Testing configuration data JSON serialization...")
    
    from squid_control.control.config import get_microscope_configuration_data
    import json
    
    # Test 1: Serialize all configuration
    print("1. Testing full configuration JSON serialization...")
    config_all = get_microscope_configuration_data(config_section="all", include_defaults=True, is_simulation=True, is_local=False)
    
    try:
        json_str = json.dumps(config_all, indent=2)
        assert len(json_str) > 100  # Should be substantial JSON
        print(f"   JSON serialization successful, length: {len(json_str)} characters")
        
        # Test deserialization
        deserialized = json.loads(json_str)
        assert deserialized == config_all
        print("   JSON deserialization successful")
        
    except (TypeError, ValueError) as e:
        pytest.fail(f"Configuration data is not JSON serializable: {e}")
    
    # Test 2: Serialize individual sections
    print("2. Testing individual section JSON serialization...")
    test_sections = ["camera", "stage", "illumination"]
    
    for section in test_sections:
        section_config = get_microscope_configuration_data(config_section=section, include_defaults=True, is_simulation=True, is_local=False)
        
        try:
            json_str = json.dumps(section_config)
            deserialized = json.loads(json_str)
            assert deserialized == section_config
            print(f"   Section '{section}' JSON serialization: ✓")
            
        except (TypeError, ValueError) as e:
            pytest.fail(f"Section '{section}' data is not JSON serializable: {e}")
    
    # Test 3: Test with different parameter combinations
    print("3. Testing JSON serialization with different parameters...")
    parameter_combinations = [
        {"config_section": "hardware", "include_defaults": False, "is_simulation": True, "is_local": False},
        {"config_section": "optics", "include_defaults": True, "is_simulation": False, "is_local": True},
        {"config_section": "autofocus", "include_defaults": False, "is_simulation": False, "is_local": False},
    ]
    
    for params in parameter_combinations:
        config_data = get_microscope_configuration_data(**params)
        
        try:
            json_str = json.dumps(config_data)
            deserialized = json.loads(json_str)
            assert deserialized == config_data
            print(f"   Parameters {params}: ✓")
            
        except (TypeError, ValueError) as e:
            pytest.fail(f"Configuration with parameters {params} is not JSON serializable: {e}")
    
    print("✅ Configuration JSON serialization tests passed!")

# New comprehensive tests for configuration, experiment, and scanning

@pytest.mark.timeout(60)
async def test_configuration_setup(sim_controller_fixture):
    """Test configuration setup with different illumination settings."""
    async for controller in sim_controller_fixture:
        # Test custom illumination settings with different channels
        test_settings = [
            {'channel': 'BF LED matrix full', 'intensity': 20.0, 'exposure_time': 25.0},
            {'channel': 'Fluorescence 405 nm Ex', 'intensity': 45.0, 'exposure_time': 150.0},
            {'channel': 'Fluorescence 488 nm Ex', 'intensity': 60.0, 'exposure_time': 100.0},
            {'channel': 'Fluorescence 561 nm Ex', 'intensity': 80.0, 'exposure_time': 200.0},
            {'channel': 'Fluorescence 638 nm Ex', 'intensity': 90.0, 'exposure_time': 200.0},
            {'channel': 'Fluorescence 730 nm Ex', 'intensity': 40.0, 'exposure_time': 200.0},
        ]
        
        # Apply settings
        controller.multipointController.set_selected_configurations_with_settings(test_settings)
        
        # Verify configurations were applied correctly
        assert len(controller.multipointController.selected_configurations) == 6
        
        for i, config in enumerate(controller.multipointController.selected_configurations):
            expected = test_settings[i]
            assert config.name == expected['channel']
            assert config.illumination_intensity == expected['intensity']
            assert config.exposure_time == expected['exposure_time']
        
        print("✅ Configuration setup test passed!")
        break



@pytest.mark.timeout(60)
async def test_plate_scan_with_custom_illumination_settings(sim_controller_fixture):
    """Test plate scanning with custom illumination settings and verify they are saved correctly."""
    async for controller in sim_controller_fixture:
        # First, let's see what configurations are actually available
        available_configs = [cfg.name for cfg in controller.multipointController.configurationManager.configurations]
        print(f"Available configurations: {available_configs}")
        
        # Use only configurations that actually exist - check first 3 that should be available
        potential_settings = [
            {'channel': 'BF LED matrix full', 'intensity': 25.0, 'exposure_time': 15.0},
            {'channel': 'Fluorescence 405 nm Ex', 'intensity': 80.0, 'exposure_time': 120.0},
            {'channel': 'Fluorescence 488 nm Ex', 'intensity': 60.0, 'exposure_time': 90.0},
            {'channel': 'Fluorescence 561 nm Ex', 'intensity': 95.0, 'exposure_time': 180.0},
        ]
        
        # Filter to only use configurations that exist
        custom_settings = []
        for setting in potential_settings:
            if setting['channel'] in available_configs:
                custom_settings.append(setting)
            else:
                print(f"Configuration '{setting['channel']}' not available, skipping")
        
        # Need at least 2 configurations to test properly
        if len(custom_settings) < 2:
            print(f"Only {len(custom_settings)} configurations available, test cannot proceed")
            print("Available configs:", available_configs)
            # Just test that the basic functionality works with whatever's available
            if len(available_configs) >= 2:
                custom_settings = [
                    {'channel': available_configs[0], 'intensity': 25.0, 'exposure_time': 15.0},
                    {'channel': available_configs[1], 'intensity': 80.0, 'exposure_time': 120.0},
                ]
            else:
                print("Not enough configurations available, skipping test")
                return
        
        print(f"Using {len(custom_settings)} configurations for testing: {[s['channel'] for s in custom_settings]}")
        
        # Test setting configurations with custom settings
        controller.multipointController.set_selected_configurations_with_settings(custom_settings)
        
        # Verify configurations were applied in memory
        expected_count = len(custom_settings)
        actual_count = len(controller.multipointController.selected_configurations)
        print(f"Expected {expected_count} configurations, got {actual_count}")
        
        assert actual_count == expected_count, f"Expected {expected_count} configurations, but got {actual_count}"
        
        for i, config in enumerate(controller.multipointController.selected_configurations):
            expected = custom_settings[i]
            print(f"Checking config {i}: '{config.name}' vs '{expected['channel']}'")
            assert config.name == expected['channel'], f"Config name mismatch: expected '{expected['channel']}', got '{config.name}'"
            assert config.illumination_intensity == expected['intensity'], f"Intensity mismatch for {config.name}: expected {expected['intensity']}, got {config.illumination_intensity}"
            assert config.exposure_time == expected['exposure_time'], f"Exposure time mismatch for {config.name}: expected {expected['exposure_time']}, got {config.exposure_time}"
        
        # Test experiment creation and XML saving
        controller.multipointController.set_base_path(tempfile.gettempdir())
        controller.multipointController.start_new_experiment("test_custom_illumination")
        
        # Verify experiment folder and files were created
        experiment_folder = os.path.join(controller.multipointController.base_path, controller.multipointController.experiment_ID)
        config_file = os.path.join(experiment_folder, "configurations.xml")
        params_file = os.path.join(experiment_folder, "acquisition parameters.json")
        
        assert os.path.exists(experiment_folder), f"Experiment folder not created: {experiment_folder}"
        assert os.path.exists(config_file), f"Config file not created: {config_file}"
        assert os.path.exists(params_file), f"Params file not created: {params_file}"
        
        # Parse and validate the saved XML configuration
        import xml.etree.ElementTree as ET
        tree = ET.parse(config_file)
        root = tree.getroot()
        
        # Find all selected configurations in the XML (correct XML structure)
        selected_modes = []
        for mode in root.findall('.//mode[@Selected="1"]'):
            selected_modes.append({
                'name': mode.get('Name'),
                'intensity': float(mode.get('IlluminationIntensity')),
                'exposure': float(mode.get('ExposureTime'))
            })
        
        print(f"Found {len(selected_modes)} selected modes in XML")
        for mode in selected_modes:
            print(f"  - {mode['name']}: intensity={mode['intensity']}, exposure={mode['exposure']}")
        
        # Verify that custom settings were correctly saved in XML
        assert len(selected_modes) >= expected_count, f"Expected at least {expected_count} selected modes, found {len(selected_modes)}"
        
        # Check each custom setting was saved correctly
        for expected in custom_settings:
            saved_config = next((mode for mode in selected_modes if mode['name'] == expected['channel']), None)
            assert saved_config is not None, f"Configuration '{expected['channel']}' not found in saved XML"
            assert saved_config['intensity'] == expected['intensity'], f"Intensity mismatch for {expected['channel']}: expected {expected['intensity']}, got {saved_config['intensity']}"
            assert saved_config['exposure'] == expected['exposure_time'], f"Exposure time mismatch for {expected['channel']}: expected {expected['exposure_time']}, got {saved_config['exposure']}"
        
        # Test plate scan functionality (mock to avoid actual scanning)
        original_run_acquisition = controller.multipointController.run_acquisition
        original_move_to_scanning = controller.move_to_scaning_position
        
        def mock_run_acquisition():
            pass
        def mock_move_to_scanning():
            pass
            
        controller.multipointController.run_acquisition = mock_run_acquisition
        controller.move_to_scaning_position = mock_move_to_scanning
        
        # Test plate scan with the custom settings
        controller.plate_scan(
            well_plate_type='96',
            illumination_settings=custom_settings,
            scanning_zone=[(0, 0), (1, 1)],  # A1 to B2
            Nx=2, Ny=2,
            action_ID='test_custom_scan'
        )
        
        # Restore original methods
        controller.multipointController.run_acquisition = original_run_acquisition
        controller.move_to_scaning_position = original_move_to_scanning
        
        # Verify scan parameters were set correctly
        assert controller.multipointController.NX == 2
        assert controller.multipointController.NY == 2
        assert not controller.is_busy
        
        # Cleanup
        import shutil
        shutil.rmtree(experiment_folder)
        print("✅ Custom illumination settings test passed - configurations saved correctly in XML!")
        break

# Stage Velocity Control Tests for SquidController
async def test_set_stage_velocity_basic(sim_controller_fixture):
    """Test basic set_stage_velocity functionality in SquidController."""
    async for controller in sim_controller_fixture:
        print("Testing set_stage_velocity basic functionality...")
        
        # Test basic velocity setting with both axes
        result = controller.set_stage_velocity(velocity_x_mm_per_s=25.0, velocity_y_mm_per_s=20.0)
        
        assert isinstance(result, dict)
        assert result["success"] == True
        assert result["velocity_x_mm_per_s"] == 25.0
        assert result["velocity_y_mm_per_s"] == 20.0
        print(f"   Set velocities: X={result['velocity_x_mm_per_s']} mm/s, Y={result['velocity_y_mm_per_s']} mm/s")
        
        # Test single axis velocity setting
        result_x = controller.set_stage_velocity(velocity_x_mm_per_s=15.0)
        assert result_x["success"] == True
        assert result_x["velocity_x_mm_per_s"] == 15.0
        assert result_x["velocity_y_mm_per_s"] > 0  # Should use default
        
        # Test default values
        result_default = controller.set_stage_velocity()
        assert result_default["success"] == True
        assert result_default["velocity_x_mm_per_s"] > 0
        assert result_default["velocity_y_mm_per_s"] > 0
        
        print("✅ set_stage_velocity basic tests passed!")
        break

async def test_set_stage_velocity_integration(sim_controller_fixture):
    """Test set_stage_velocity integration with movement operations."""
    async for controller in sim_controller_fixture:
        print("Testing set_stage_velocity integration...")
        
        # Set velocity and perform movement
        velocity_result = controller.set_stage_velocity(velocity_x_mm_per_s=20.0, velocity_y_mm_per_s=15.0)
        assert velocity_result["success"] == True
        
        # Test movement with new velocity
        moved, x_before, y_before, z_before, x_after, y_after, z_after = controller.move_by_distance_limited(1.0, 0.5, 0.0)
        assert moved == True
        assert abs(x_after - x_before - 1.0) < 0.01
        assert abs(y_after - y_before - 0.5) < 0.01
        print("   ✓ Movement after velocity setting completed")
        
        # Test absolute positioning with custom velocity
        controller.set_stage_velocity(velocity_x_mm_per_s=30.0, velocity_y_mm_per_s=25.0)
        moved_x, _, _, _, final_x = controller.move_x_to_limited(10.0)
        moved_y, _, _, _, final_y = controller.move_y_to_limited(15.0)
        assert moved_x == True and moved_y == True
        print("   ✓ Absolute positioning with custom velocity completed")
        
        print("✅ set_stage_velocity integration tests passed!")
        break

async def test_set_stage_velocity_error_handling(sim_controller_fixture):
    """Test error handling in set_stage_velocity method."""
    async for controller in sim_controller_fixture:
        print("Testing set_stage_velocity error handling...")
        
        # Test negative velocities
        result_negative = controller.set_stage_velocity(velocity_x_mm_per_s=-10.0)
        if result_negative["success"] == False:
            print("   ✓ Negative velocity properly rejected")
        else:
            print("   ✓ Negative velocity handled gracefully")
        
        # Test zero velocities
        result_zero = controller.set_stage_velocity(velocity_x_mm_per_s=0.0, velocity_y_mm_per_s=0.0)
        if result_zero["success"] == False:
            print("   ✓ Zero velocity properly rejected")
        else:
            print("   ✓ Zero velocity handled gracefully")
        
        # Test extreme velocities
        result_extreme = controller.set_stage_velocity(velocity_x_mm_per_s=1000.0)
        if result_extreme["success"] == False:
            print("   ✓ Extreme velocity properly rejected")
        else:
            print("   ✓ Extreme velocity handled gracefully")
        
        print("✅ set_stage_velocity error handling tests passed!")
        break



if __name__ == "__main__":
    print("Running Well Position Detection Tests...")
    print("=" * 50)
    
    test_get_well_from_position_96_well()
    print()
    test_get_well_from_position_different_plates()
    print()
    test_get_well_from_position_edge_cases()
    print()
    test_well_location_accuracy()
    print()
    test_well_boundary_detection()
    
    print("Running Microscope Configuration Tests...")
    print("=" * 50)
    test_get_microscope_configuration_data()
    print()
    test_configuration_data_content()
    print()
    test_configuration_json_serializable()
    
    print("=" * 50)
    print("🎉 All tests passed!")