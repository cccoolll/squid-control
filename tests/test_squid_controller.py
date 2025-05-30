import pytest
import asyncio
from squid_control.squid_controller import SquidController
from squid_control.control.config import CONFIG, SIMULATED_CAMERA # Import necessary config

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture
async def sim_controller_fixture():
    """Fixture to provide a SquidController instance in simulation mode."""
    controller = SquidController(is_simulation=True)
    yield controller
    # Teardown: close controller resources
    # Assuming controller.close() is synchronous as per typical Python object cleanup
    # If it were async, it should be `await controller.aclose()` or similar.
    controller.close()

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
        assert controller.current_expousre_time == test_exposure
        break

async def test_autofocus_simulation(sim_controller_fixture):
    """Test autofocus in simulation mode."""
    async for controller in sim_controller_fixture:
        initial_x, initial_y, initial_z, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
        
        # These methods are synchronous
        controller.do_autofocus_simulation()
        
        x_after, y_after, z_after, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
        
        assert x_after == pytest.approx(initial_x)
        assert y_after == pytest.approx(initial_y)
        assert z_after != pytest.approx(initial_z)
        assert z_after == pytest.approx(SIMULATED_CAMERA.ORIN_Z, abs=0.01)

        controller.do_autofocus()
        x_final, y_final, z_final, *_ = controller.navigationController.update_pos(microcontroller=controller.microcontroller)
        assert z_final == pytest.approx(SIMULATED_CAMERA.ORIN_Z, abs=0.01)
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


async def test_close_controller(sim_controller_fixture):
    """Test if the controller's close method can be called without errors."""
    async for controller in sim_controller_fixture:
        # controller.close() is called by the fixture's teardown.
        # This test ensures explicit call is also fine and checks camera state.
        controller.close() # Assuming synchronous close
        assert True
        # Check camera state after close
        assert controller.camera.is_streaming == False
        break
