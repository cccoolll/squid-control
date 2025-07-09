import os 
import shutil
# app specific libraries
import squid_control.control.core_reef as core
import squid_control.control.microcontroller as microcontroller
from squid_control.control.config import *
from squid_control.control.camera import get_camera
from squid_control.control.utils import rotate_and_flip_image
from squid_control.control.config import ChannelMapper
import cv2
import logging
import logging.handlers
import matplotlib.path as mpath
# Import serial_peripherals conditionally based on simulation mode
import sys
import numpy as np
from squid_control.stitching.zarr_canvas import ZarrCanvas
from pathlib import Path

_is_simulation_mode = (
    "--simulation" in sys.argv or 
    os.environ.get("SQUID_SIMULATION_MODE", "").lower() in ["true", "1", "yes"] or
    os.environ.get("PYTEST_CURRENT_TEST") is not None  # Running in pytest
)

if _is_simulation_mode:
    print("Simulation mode detected - skipping hardware peripheral imports")
    SERIAL_PERIPHERALS_AVAILABLE = False
    serial_peripherals = None
else:
    try:
        import squid_control.control.serial_peripherals as serial_peripherals
        SERIAL_PERIPHERALS_AVAILABLE = True
    except ImportError as e:
        print(f"serial_peripherals import error - hardware peripheral functionality not available: {e}")
        SERIAL_PERIPHERALS_AVAILABLE = False
        serial_peripherals = None
if CONFIG.SUPPORT_LASER_AUTOFOCUS:
    import squid_control.control.core_displacement_measurement as core_displacement_measurement

import time
import asyncio
#using os to set current working directory
#find the current path
path=os.path.abspath(__file__)
# Get the directory where config.py is located
config_dir = os.path.join(os.path.dirname(os.path.dirname(path)), 'squid_control/control')
cache_file_path = os.path.join(config_dir, 'cache_config_file_path.txt')

# Try to read the cached config path
config_path = None
if os.path.exists(cache_file_path):
    try:
        with open(cache_file_path, 'r') as f:
            config_path = f.readline().strip()
    except:
        pass

def setup_logging(log_file="squid_controller.log", max_bytes=100000, backup_count=3):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

class SquidController:
    fps_software_trigger= 10

    def __init__(self,is_simulation, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.data_channel = None
        self.is_simulation = is_simulation
        self.is_busy = False
        self.scan_stop_requested = False  # Flag to stop ongoing scans
        if is_simulation:
            config_path = os.path.join(os.path.dirname(path), 'configuration_HCS_v2_example.ini')
        else:
            config_path = os.path.join(os.path.dirname(path), 'configuration_HCS_v2.ini')

        print(f"Loading configuration from: {config_path}")
        load_config(config_path, False)
        
        # Create objects after configuration is loaded to use updated CONFIG values
        self.objectiveStore = core.ObjectiveStore()
        print(f"ObjectiveStore initialized with default objective: {self.objectiveStore.current_objective}")
        camera, camera_fc = get_camera(CONFIG.CAMERA_TYPE)
        # load objects
        if self.is_simulation:
            if CONFIG.ENABLE_SPINNING_DISK_CONFOCAL and SERIAL_PERIPHERALS_AVAILABLE:
                self.xlight = serial_peripherals.XLight_Simulation()
            if CONFIG.SUPPORT_LASER_AUTOFOCUS:
                self.camera = camera.Camera_Simulation(
                    rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                    flip_image=CONFIG.FLIP_IMAGE,
                )
                self.camera_focus = camera_fc.Camera_Simulation()
            else:
                self.camera = camera.Camera_Simulation(
                    rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                    flip_image=CONFIG.FLIP_IMAGE,
                )
            self.microcontroller = microcontroller.Microcontroller_Simulation()
        else:
            if CONFIG.ENABLE_SPINNING_DISK_CONFOCAL and SERIAL_PERIPHERALS_AVAILABLE:
                self.xlight = serial_peripherals.XLight()
            try:
                if CONFIG.SUPPORT_LASER_AUTOFOCUS:
                    sn_camera_main = camera.get_sn_by_model(CONFIG.MAIN_CAMERA_MODEL)
                    sn_camera_focus = camera_fc.get_sn_by_model(
                        CONFIG.FOCUS_CAMERA_MODEL
                    )
                    self.camera = camera.Camera(
                        sn=sn_camera_main,
                        rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                        flip_image=CONFIG.FLIP_IMAGE,
                    )
                    self.camera.open()
                    self.camera_focus = camera_fc.Camera(sn=sn_camera_focus)
                    self.camera_focus.open()
                else:
                    self.camera = camera.Camera(
                        rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                        flip_image=CONFIG.FLIP_IMAGE,
                    )
                    self.camera.open()
            except:
                if CONFIG.SUPPORT_LASER_AUTOFOCUS:
                    self.camera = camera.Camera_Simulation(
                        rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                        flip_image=CONFIG.FLIP_IMAGE,
                    )
                    self.camera.open()
                    self.camera_focus = camera.Camera_Simulation()
                    self.camera_focus.open()
                else:
                    self.camera = camera.Camera_Simulation(
                        rotate_image_angle=CONFIG.ROTATE_IMAGE_ANGLE,
                        flip_image=CONFIG.FLIP_IMAGE,
                    )
                    self.camera.open()
                print("! camera not detected, using simulated camera !")
            self.microcontroller = microcontroller.Microcontroller(
                version=CONFIG.CONTROLLER_VERSION
            )

        # reset the MCU
        self.microcontroller.reset()
        time.sleep(0.5)

        # reinitialize motor drivers and DAC (in particular for V2.1 driver board where PG is not functional)
        self.microcontroller.initialize_drivers()
        time.sleep(0.5)

        # configure the actuators
        self.microcontroller.configure_actuators()

        self.configurationManager = core.ConfigurationManager(
            filename="./u2os_fucci_illumination_configurations.xml"
        )

        self.streamHandler = core.StreamHandler(
            display_resolution_scaling=CONFIG.DEFAULT_DISPLAY_CROP / 100
        )
        self.liveController = core.LiveController(
            self.camera, self.microcontroller, self.configurationManager
        )
        self.navigationController = core.NavigationController(
            self.microcontroller, parent=self
        )
        self.slidePositionController = core.SlidePositionController(
            self.navigationController, self.liveController, is_for_wellplate=True
        )
        self.autofocusController = core.AutoFocusController(
            self.camera, self.navigationController, self.liveController
        )
        self.scanCoordinates = core.ScanCoordinates()
        self.multipointController = core.MultiPointController(
            self.camera,
            self.navigationController,
            self.liveController,
            self.autofocusController,
            self.configurationManager,
            scanCoordinates=self.scanCoordinates,
            parent=self,
        )
        if CONFIG.ENABLE_TRACKING:
            self.trackingController = core.TrackingController(
                self.camera,
                self.microcontroller,
                self.navigationController,
                self.configurationManager,
                self.liveController,
                self.autofocusController,
            )

        # retract the objective
        self.navigationController.home_z()
        # wait for the operation to finish
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print("z homing timeout, the program will exit")
                exit()
        print("objective retracted")

        # set encoder arguments
        # set axis pid control enable
        # only CONFIG.ENABLE_PID_X and CONFIG.HAS_ENCODER_X are both enable, can be enable to PID
        if CONFIG.HAS_ENCODER_X == True:
            self.navigationController.configure_encoder(
                0,
                (CONFIG.SCREW_PITCH_X_MM * 1000) / CONFIG.ENCODER_RESOLUTION_UM_X,
                CONFIG.ENCODER_FLIP_DIR_X,
            )
            self.navigationController.set_pid_control_enable(0, CONFIG.ENABLE_PID_X)
        if CONFIG.HAS_ENCODER_Y == True:
            self.navigationController.configure_encoder(
                1,
                (CONFIG.SCREW_PITCH_Y_MM * 1000) / CONFIG.ENCODER_RESOLUTION_UM_Y,
                CONFIG.ENCODER_FLIP_DIR_Y,
            )
            self.navigationController.set_pid_control_enable(1, CONFIG.ENABLE_PID_Y)
        if CONFIG.HAS_ENCODER_Z == True:
            self.navigationController.configure_encoder(
                2,
                (CONFIG.SCREW_PITCH_Z_MM * 1000) / CONFIG.ENCODER_RESOLUTION_UM_Z,
                CONFIG.ENCODER_FLIP_DIR_Z,
            )
            self.navigationController.set_pid_control_enable(2, CONFIG.ENABLE_PID_Z)
        time.sleep(0.5)

        self.navigationController.set_z_limit_pos_mm(
            CONFIG.SOFTWARE_POS_LIMIT.Z_POSITIVE
        )

        # home XY, set zero and set software limit
        print("home xy")
        timestamp_start = time.time()
        # x needs to be at > + 20 mm when homing y
        self.navigationController.move_x(20)  # to-do: add blocking code
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        # home y
        self.navigationController.home_y()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print("y homing timeout, the program will exit")
                exit()
        self.navigationController.zero_y()
        # home x
        self.navigationController.home_x()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print("y homing timeout, the program will exit")
                exit()
        self.navigationController.zero_x()
        self.slidePositionController.homing_done = True
        print("home xy done")

        # move to scanning position
        self.navigationController.move_x(32.3)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y(29.35)
        while self.microcontroller.is_busy():
            time.sleep(0.005)

        # move z
        self.navigationController.move_z_to(CONFIG.DEFAULT_Z_POS_MM)
        # wait for the operation to finish
        
        # FIXME: This is failing right now, z return timeout
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 5:
                print("z return timeout, the program will exit")
                exit()

        # set output's gains
        div = 1 if CONFIG.OUTPUT_GAINS.REFDIV is True else 0
        gains = CONFIG.OUTPUT_GAINS.CHANNEL0_GAIN << 0
        gains += CONFIG.OUTPUT_GAINS.CHANNEL1_GAIN << 1
        gains += CONFIG.OUTPUT_GAINS.CHANNEL2_GAIN << 2
        gains += CONFIG.OUTPUT_GAINS.CHANNEL3_GAIN << 3
        gains += CONFIG.OUTPUT_GAINS.CHANNEL4_GAIN << 4
        gains += CONFIG.OUTPUT_GAINS.CHANNEL5_GAIN << 5
        gains += CONFIG.OUTPUT_GAINS.CHANNEL6_GAIN << 6
        gains += CONFIG.OUTPUT_GAINS.CHANNEL7_GAIN << 7
        self.microcontroller.configure_dac80508_refdiv_and_gain(div, gains)

        # set illumination intensity factor
        self.microcontroller.set_dac80508_scaling_factor_for_illumination(
            CONFIG.ILLUMINATION_INTENSITY_FACTOR
        )

        # open the camera
        # camera start streaming
        # self.camera.set_reverse_x(CAMERA_REVERSE_X) # these are not implemented for the cameras in use
        # self.camera.set_reverse_y(CAMERA_REVERSE_Y) # these are not implemented for the cameras in use
        self.camera.set_software_triggered_acquisition()  # self.camera.set_continuous_acquisition()
        self.camera.set_callback(self.streamHandler.on_new_frame)
        #self.camera.enable_callback()
        self.camera.start_streaming()     

        if CONFIG.SUPPORT_LASER_AUTOFOCUS:

            # controllers
            self.configurationManager_focus_camera = core.ConfigurationManager(filename='./focus_camera_configurations.xml')
            self.streamHandler_focus_camera = core.StreamHandler()
            self.liveController_focus_camera = core.LiveController(self.camera_focus,self.microcontroller,self.configurationManager_focus_camera,control_illumination=False,for_displacement_measurement=True)
            self.multipointController = core.MultiPointController(self.camera,self.navigationController,self.liveController,self.autofocusController,self.configurationManager,scanCoordinates=self.scanCoordinates,parent=self)
            self.displacementMeasurementController = core_displacement_measurement.DisplacementMeasurementController()
            self.laserAutofocusController = core.LaserAutofocusController(self.microcontroller,self.camera_focus,self.liveController_focus_camera,self.navigationController,has_two_interfaces=CONFIG.HAS_TWO_INTERFACES,use_glass_top=CONFIG.USE_GLASS_TOP)

            # camera
            self.camera_focus.set_software_triggered_acquisition() #self.camera.set_continuous_acquisition()
            self.camera_focus.set_callback(self.streamHandler_focus_camera.on_new_frame)
            #self.camera_focus.enable_callback()
            self.camera_focus.start_streaming()


        # set software limits        
        self.navigationController.set_x_limit_pos_mm(CONFIG.SOFTWARE_POS_LIMIT.X_POSITIVE)
        self.navigationController.set_x_limit_neg_mm(CONFIG.SOFTWARE_POS_LIMIT.X_NEGATIVE)
        self.navigationController.set_y_limit_pos_mm(CONFIG.SOFTWARE_POS_LIMIT.Y_POSITIVE)
        self.navigationController.set_y_limit_neg_mm(CONFIG.SOFTWARE_POS_LIMIT.Y_NEGATIVE)

        # set the default infomation, this will be used for the simulated camera
        self.dz = 0
        self.current_channel = 0
        self.current_exposure_time = 100
        self.current_intensity = 100
        self.pixel_size_xy = 0.333
        # drift correction for image map
        self.drift_correction_x = -1.6
        self.drift_correction_y = -2.1
        # simulated sample data alias
        self.sample_data_alias = "agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38"
        self.get_pixel_size()

        # Initialize multi-fileset tracking
        self.zarr_canvases = {}  # Dictionary to track multiple zarr canvases by name
        self.active_canvas_name = None  # Track the currently active canvas
        self.zarr_canvas = None  # Keep for backward compatibility - will point to active canvas
        
        # Clean up ZARR_PATH directory on startup
        self._cleanup_zarr_directory()

    def get_pixel_size(self):
        """Calculate pixel size based on imaging parameters."""
        try:
            tube_lens_mm = float(CONFIG.TUBE_LENS_MM)
            pixel_size_um = float(CONFIG.CAMERA_PIXEL_SIZE_UM[CONFIG.CAMERA_SENSOR])
            
            object_dict_key = self.objectiveStore.current_objective
            objective = self.objectiveStore.objectives_dict[object_dict_key]
            magnification =  float(objective['magnification'])
            objective_tube_lens_mm = float(objective['tube_lens_f_mm'])
            print(f"Using objective: {object_dict_key}")
            print(f"CONFIG.DEFAULT_OBJECTIVE: {CONFIG.DEFAULT_OBJECTIVE}")
            print(f"Tube lens: {tube_lens_mm} mm, Objective tube lens: {objective_tube_lens_mm} mm, Pixel size: {pixel_size_um} µm, Magnification: {magnification}")
        except Exception as e:
            logging.error(f"Missing required parameters for pixel size calculation: {e}")
            return

        self.pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
        self.pixel_size_xy = self.pixel_size_xy * CONFIG.PIXEL_SIZE_ADJUSTMENT_FACTOR
        print(f"Pixel size: {self.pixel_size_xy} µm (adjustment factor: {CONFIG.PIXEL_SIZE_ADJUSTMENT_FACTOR})")
        
                
    def move_to_scaning_position(self):
        # move to scanning position
        self.navigationController.move_z_to(0.4)
        self.navigationController.move_x(20)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y(20)
        while self.microcontroller.is_busy():
            time.sleep(0.005)

        # move z
        self.navigationController.move_z_to(CONFIG.DEFAULT_Z_POS_MM)
        # wait for the operation to finish
        t0 = time.time() 
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 5:
                print('z return timeout, the program will exit')
                exit()

    def plate_scan(self, well_plate_type='96', illumination_settings=None, do_contrast_autofocus=False, do_reflection_af=True, scanning_zone=[(0,0),(2,2)], Nx=3, Ny=3, action_ID='testPlateScanNew'):
        """
        New well plate scanning function with custom illumination settings.
        
        Args:
            well_plate_type (str): Type of well plate ('96', '384', etc.)
            illumination_settings (list): List of dictionaries with illumination settings
                Each dict should contain:
                {
                    'channel': 'BF LED matrix full',  # Channel name
                    'intensity': 50.0,               # Illumination intensity (0-100)
                    'exposure_time': 25.0            # Exposure time in ms
                }
            do_contrast_autofocus (bool): Whether to perform contrast-based autofocus
            do_reflection_af (bool): Whether to perform reflection-based autofocus
            scanning_zone (list): List of two tuples [(start_row, start_col), (end_row, end_col)]
            Nx (int): Number of X positions per well
            Ny (int): Number of Y positions per well
            action_ID (str): Identifier for this scan
        """
        if illumination_settings is None:
            logging.warning("No illumination settings provided, using default settings")
            # Default settings if none provided
            illumination_settings = [
                {'channel': 'BF LED matrix full', 'intensity': 18, 'exposure_time': 37},
                {'channel': 'Fluorescence 405 nm Ex', 'intensity': 45, 'exposure_time': 30},
                {'channel': 'Fluorescence 488 nm Ex', 'intensity': 30, 'exposure_time': 100},
                {'channel': 'Fluorescence 561 nm Ex', 'intensity': 100, 'exposure_time': 200},
                {'channel': 'Fluorescence 638 nm Ex', 'intensity': 100, 'exposure_time': 200},
                {'channel': 'Fluorescence 730 nm Ex', 'intensity': 100, 'exposure_time': 200},
            ]
        
        # Update configurations with custom settings
        self.multipointController.set_selected_configurations_with_settings(illumination_settings)
        
        # Move to scanning position
        self.move_to_scaning_position()
        
        # Set up scan coordinates
        self.scanCoordinates.well_selector.set_selected_wells(scanning_zone[0], scanning_zone[1])
        self.scanCoordinates.get_selected_wells_to_coordinates(wellplate_type=well_plate_type, is_simulation=self.is_simulation)
        
        # Configure multipoint controller
        self.multipointController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.multipointController.do_autofocus = do_contrast_autofocus
        self.multipointController.do_reflection_af = do_reflection_af
        self.multipointController.set_NX(Nx)
        self.multipointController.set_NY(Ny)
        self.multipointController.start_new_experiment(action_ID)
        
        # Start scanning
        self.is_busy = True
        print('Starting new plate scan with custom illumination settings')
        self.multipointController.run_acquisition()
        print('New plate scan completed')
        self.is_busy = False
    
    def stop_plate_scan(self):
        self.multipointController.abort_acqusition_requested = True
        self.is_busy = False
        print('Plate scan stopped')
        
    def stop_scan_well_plate_new(self):
        """Stop the new well plate scan - alias for stop_plate_scan"""
        self.stop_plate_scan()
        
    async def send_trigger_simulation(self, channel=0, intensity=100, exposure_time=100):
        print('Getting simulated image')
        current_x, current_y, current_z, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)
        self.dz = current_z - SIMULATED_CAMERA.ORIN_Z
        self.current_channel = channel
        magnification_factor = SIMULATED_CAMERA.MAGNIFICATION_FACTOR
        self.current_exposure_time = exposure_time
        self.current_intensity = intensity
        corrected_x = current_x + self.drift_correction_x
        corrected_y = current_y + self.drift_correction_y
        await self.camera.send_trigger(corrected_x, corrected_y, self.dz, self.pixel_size_xy, channel, intensity, exposure_time, magnification_factor, sample_data_alias=self.sample_data_alias)
        print(f'For simulated camera, exposure_time={exposure_time}, intensity={intensity}, magnification_factor={magnification_factor}, current position: {current_x},{current_y},{current_z}')
   
    def set_simulated_sample_data_alias(self, sample_data_alias):
        self.sample_data_alias = sample_data_alias

    def get_simulated_sample_data_alias(self):
        return self.sample_data_alias

    async def do_autofocus(self):
        
        if self.is_simulation:
            await self.do_autofocus_simulation()
        else:
            self.autofocusController.set_deltaZ(1.524)
            self.autofocusController.set_N(15)
            self.autofocusController.autofocus()
            self.autofocusController.wait_till_autofocus_has_completed()

    async def do_autofocus_simulation(self):
        
        random_z = SIMULATED_CAMERA.ORIN_Z + np.random.normal(0,0.001)
        self.navigationController.move_z_to(random_z)
        await self.send_trigger_simulation(self.current_channel, self.current_intensity, self.current_exposure_time)
        
    def init_laser_autofocus(self):
        self.laserAutofocusController.initialize_auto()

    async def do_laser_autofocus(self):
        if self.is_simulation:
            await self.do_autofocus_simulation()
        else:
            self.laserAutofocusController.move_to_target(0)
    
    def measure_displacement(self):
        self.laserAutofocusController.measure_displacement()     
        
    def move_to_well(self,row,column, wellplate_type='96'):
        if wellplate_type == '6':
            wellplate_format = WELLPLATE_FORMAT_6
        elif wellplate_type == '12':
            wellplate_format = WELLPLATE_FORMAT_12
        elif wellplate_type == '24':
            wellplate_format = WELLPLATE_FORMAT_24
        elif wellplate_type == '96':
            wellplate_format = WELLPLATE_FORMAT_96
        elif wellplate_type == '384':
            wellplate_format = WELLPLATE_FORMAT_384
        else:
            # Default to 96-well plate if unsupported type is provided
            wellplate_format = WELLPLATE_FORMAT_96
        
        if column != 0 and column != None:
            mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM/(self.navigationController.x_microstepping*CONFIG.FULLSTEPS_PER_REV_X)
            if self.is_simulation:
                x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM
            else:
                x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_X_MM
            x_usteps = CONFIG.STAGE_MOVEMENT_SIGN_X*round(x_mm/mm_per_ustep_X)
            self.microcontroller.move_x_to_usteps(x_usteps)
        if row != 0 and row != None:
            mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM/(self.navigationController.y_microstepping*CONFIG.FULLSTEPS_PER_REV_Y)
            if self.is_simulation:
                y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM
            else:
                y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_Y_MM
            y_usteps = CONFIG.STAGE_MOVEMENT_SIGN_Y*round(y_mm/mm_per_ustep_Y)
            self.microcontroller.move_y_to_usteps(y_usteps)
            while self.microcontroller.is_busy():
                time.sleep(0.005)

    async def move_to_well_async(self, row, column, wellplate_type='96'):
        """
        Async version of move_to_well that doesn't block the event loop.
        
        Args:
            row: Row letter (e.g., 'A', 'B', 'C')
            column: Column number (e.g., 1, 2, 3)
            wellplate_type: Type of well plate ('6', '12', '24', '96', '384')
        """
        if wellplate_type == '6':
            wellplate_format = WELLPLATE_FORMAT_6
        elif wellplate_type == '12':
            wellplate_format = WELLPLATE_FORMAT_12
        elif wellplate_type == '24':
            wellplate_format = WELLPLATE_FORMAT_24
        elif wellplate_type == '96':
            wellplate_format = WELLPLATE_FORMAT_96
        elif wellplate_type == '384':
            wellplate_format = WELLPLATE_FORMAT_384
        else:
            # Default to 96-well plate if unsupported type is provided
            wellplate_format = WELLPLATE_FORMAT_96
        
        if column != 0 and column != None:
            mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM/(self.navigationController.x_microstepping*CONFIG.FULLSTEPS_PER_REV_X)
            if self.is_simulation:
                x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM
            else:
                x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_X_MM
            x_usteps = CONFIG.STAGE_MOVEMENT_SIGN_X*round(x_mm/mm_per_ustep_X)
            self.microcontroller.move_x_to_usteps(x_usteps)
        if row != 0 and row != None:
            mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM/(self.navigationController.y_microstepping*CONFIG.FULLSTEPS_PER_REV_Y)
            if self.is_simulation:
                y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM
            else:
                y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM + CONFIG.WELLPLATE_OFFSET_Y_MM
            y_usteps = CONFIG.STAGE_MOVEMENT_SIGN_Y*round(y_mm/mm_per_ustep_Y)
            self.microcontroller.move_y_to_usteps(y_usteps)
            # Use async sleep to avoid blocking the event loop
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)

    async def move_to_well_center_for_autofocus(self, row, column, wellplate_type='96', velocity_mm_per_s=30.0):
        """
        Optimized method to move to well center for autofocus operations.
        Sets velocity, moves to well center, and waits for completion.
        
        Args:
            row: Row letter (e.g., 'A', 'B', 'C')
            column: Column number (e.g., 1, 2, 3)
            wellplate_type: Type of well plate ('6', '12', '24', '96', '384')
            velocity_mm_per_s: Velocity for movement (default 30.0 mm/s)
        """
        # Set high speed velocity for moving to well center
        velocity_result = self.set_stage_velocity(velocity_mm_per_s, velocity_mm_per_s)
        if not velocity_result['success']:
            logging.warning(f"Failed to set high-speed velocity for autofocus: {velocity_result['message']}")
        
        # Move to well center using async method
        await self.move_to_well_async(row, column, wellplate_type)
        
        logging.info(f'Moved to well {row}{column} center for autofocus')

    def get_well_from_position(self, wellplate_type='96', x_pos_mm=None, y_pos_mm=None):
        """
        Calculate which well position corresponds to the given X,Y coordinates.
        
        Args:
            wellplate_type (str): Type of well plate ('6', '12', '24', '96', '384')
            x_pos_mm (float, optional): X position in mm. If None, uses current position.
            y_pos_mm (float, optional): Y position in mm. If None, uses current position.
            
        Returns:
            dict: {
                'row': str or None,  # Well row letter (e.g., 'A', 'B', 'C')
                'column': int or None,  # Well column number (e.g., 1, 2, 3)
                'well_id': str or None,  # Combined well ID (e.g., 'A1', 'C3')
                'is_inside_well': bool,  # True if position is inside the well boundary
                'distance_from_center': float,  # Distance from well center in mm
                'position_status': str,  # 'in_well', 'between_wells', or 'outside_plate'
                'x_mm': float,  # Actual X position used
                'y_mm': float,  # Actual Y position used
                'plate_type': str  # Well plate type used
            }
        """
        # Get well plate format configuration
        if wellplate_type == '6':
            wellplate_format = WELLPLATE_FORMAT_6
            max_rows = 2  # A-B
            max_cols = 3  # 1-3
        elif wellplate_type == '12':
            wellplate_format = WELLPLATE_FORMAT_12
            max_rows = 3  # A-C
            max_cols = 4  # 1-4
        elif wellplate_type == '24':
            wellplate_format = WELLPLATE_FORMAT_24
            max_rows = 4  # A-D
            max_cols = 6  # 1-6
        elif wellplate_type == '96':
            wellplate_format = WELLPLATE_FORMAT_96
            max_rows = 8  # A-H
            max_cols = 12  # 1-12
        elif wellplate_type == '384':
            wellplate_format = WELLPLATE_FORMAT_384
            max_rows = 16  # A-P
            max_cols = 24  # 1-24
        else:
            # Default to 96-well plate if unsupported type is provided
            wellplate_format = WELLPLATE_FORMAT_96
            max_rows = 8
            max_cols = 12
            wellplate_type = '96'
        
        # Get current position if not provided
        if x_pos_mm is None or y_pos_mm is None:
            current_x, current_y, current_z, current_theta = self.navigationController.update_pos(
                microcontroller=self.microcontroller
            )
            if x_pos_mm is None:
                x_pos_mm = current_x
            if y_pos_mm is None:
                y_pos_mm = current_y
        
        # Apply well plate offset for hardware mode
        if self.is_simulation:
            x_offset = 0
            y_offset = 0
        else:
            x_offset = CONFIG.WELLPLATE_OFFSET_X_MM
            y_offset = CONFIG.WELLPLATE_OFFSET_Y_MM
        
        # Calculate which well this position corresponds to
        # Reverse of the move_to_well calculation
        x_relative = x_pos_mm - (wellplate_format.A1_X_MM + x_offset)
        y_relative = y_pos_mm - (wellplate_format.A1_Y_MM + y_offset)
        
        # Calculate well indices (0-based initially)
        col_index = round(x_relative / wellplate_format.WELL_SPACING_MM)
        row_index = round(y_relative / wellplate_format.WELL_SPACING_MM)
        
        # Initialize result dictionary
        result = {
            'row': None,
            'column': None,
            'well_id': None,
            'is_inside_well': False,
            'distance_from_center': float('inf'),
            'position_status': 'outside_plate',
            'x_mm': x_pos_mm,
            'y_mm': y_pos_mm,
            'plate_type': wellplate_type
        }
        
        # Check if the calculated well indices are valid
        if 0 <= col_index < max_cols and 0 <= row_index < max_rows:
            # Convert to 1-based column and letter-based row
            column = col_index + 1
            row = chr(ord('A') + row_index)
            
            result['row'] = row
            result['column'] = column
            result['well_id'] = f"{row}{column}"
            
            # Calculate the exact center position of this well
            well_center_x = wellplate_format.A1_X_MM + x_offset + col_index * wellplate_format.WELL_SPACING_MM
            well_center_y = wellplate_format.A1_Y_MM + y_offset + row_index * wellplate_format.WELL_SPACING_MM
            
            # Calculate distance from well center
            dx = x_pos_mm - well_center_x
            dy = y_pos_mm - well_center_y
            distance_from_center = np.sqrt(dx**2 + dy**2)
            result['distance_from_center'] = distance_from_center
            
            # Check if position is inside the well boundary
            well_radius = wellplate_format.WELL_SIZE_MM / 2.0
            if distance_from_center <= well_radius:
                result['is_inside_well'] = True
                result['position_status'] = 'in_well'
            else:
                result['is_inside_well'] = False
                result['position_status'] = 'between_wells'
        else:
            # Position is outside the valid well range
            result['position_status'] = 'outside_plate'
            
            # Find the closest valid well for reference
            closest_col = max(0, min(max_cols - 1, col_index))
            closest_row = max(0, min(max_rows - 1, row_index))
            
            closest_well_center_x = wellplate_format.A1_X_MM + x_offset + closest_col * wellplate_format.WELL_SPACING_MM
            closest_well_center_y = wellplate_format.A1_Y_MM + y_offset + closest_row * wellplate_format.WELL_SPACING_MM
            
            dx = x_pos_mm - closest_well_center_x
            dy = y_pos_mm - closest_well_center_y
            result['distance_from_center'] = np.sqrt(dx**2 + dy**2)
        
        return result

    def move_x_to_limited(self, x):
        
        x_pos_before,y_pos_before, z_pos_before, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        self.navigationController.move_x_to_limited(x)
        while self.microcontroller.is_busy():
            time.sleep(0.005) 
        
        x_pos,y_pos, z_pos, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        if abs(x_pos - x) < CONFIG.STAGE_MOVED_THRESHOLD:
            return True, x_pos_before, y_pos_before, z_pos_before, x

        return False, x_pos_before, y_pos_before, z_pos_before, x
    
    def move_y_to_limited(self, y):
        x_pos_before,y_pos_before, z_pos_before, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)
        self.navigationController.move_y_to_limited(y)

        while self.microcontroller.is_busy():
            time.sleep(0.005)
        x_pos,y_pos, z_pos, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        if abs(y_pos - y) < CONFIG.STAGE_MOVED_THRESHOLD:
            return True, x_pos_before, y_pos_before, z_pos_before, y
    
        return False, x_pos_before, y_pos_before, z_pos_before, y
    
    def move_z_to_limited(self, z):
        x_pos_before,y_pos_before, z_pos_before, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)
        self.navigationController.move_z_to_limited(z)

        while self.microcontroller.is_busy():
            time.sleep(0.005)
        x_pos,y_pos, z_pos, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        if abs(z_pos - z) < CONFIG.STAGE_MOVED_THRESHOLD:
            return True, x_pos_before, y_pos_before, z_pos_before, z

        return False, x_pos_before, y_pos_before, z_pos_before, z
    

    def move_by_distance_limited(self, dx, dy, dz):
        x_pos_before,y_pos_before, z_pos_before, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        self.navigationController.move_x_limited(dx)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y_limited(dy)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_z_limited(dz)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        
        x_pos,y_pos, z_pos, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)

        if abs(x_pos-x_pos_before)<CONFIG.STAGE_MOVED_THRESHOLD and dx!=0:
            return False, x_pos_before, y_pos_before, z_pos_before, x_pos_before+dx, y_pos_before+dy, z_pos_before+dz
        if abs(y_pos-y_pos_before)<CONFIG.STAGE_MOVED_THRESHOLD and dy!=0:
            return False, x_pos_before, y_pos_before, z_pos_before, x_pos_before+dx, y_pos_before+dy, z_pos_before+dz
        if abs(z_pos-z_pos_before)<CONFIG.STAGE_MOVED_THRESHOLD and dz!=0:
            return False, x_pos_before, y_pos_before, z_pos_before, x_pos_before+dx, y_pos_before+dy, z_pos_before+dz


        return True, x_pos_before, y_pos_before, z_pos_before, x_pos, y_pos, z_pos
    
    def home_stage(self):
        # retract the object
        self.navigationController.home_z()
        # wait for the operation to finish
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print('z homing timeout, the program will exit')
                exit()
        print('objective retracted')
        self.navigationController.set_z_limit_pos_mm(CONFIG.SOFTWARE_POS_LIMIT.Z_POSITIVE)

        # home XY, set zero and set software limit
        print('home xy')
        timestamp_start = time.time()
        # x needs to be at > + 20 mm when homing y
        self.navigationController.move_x(20) # to-do: add blocking code
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        # home y
        self.navigationController.home_y()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print('y homing timeout, the program will exit')
                exit()
        self.navigationController.zero_y()
        # home x
        self.navigationController.home_x()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 10:
                print('y homing timeout, the program will exit')
                exit()
        self.navigationController.zero_x()
        self.slidePositionController.homing_done = True
        print('home xy done')
        
    def return_stage(self):
        # move to scanning position
        self.navigationController.move_x(30.26)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y(29.1)
        while self.microcontroller.is_busy():
            time.sleep(0.005)

        # move z
        self.navigationController.move_z_to(CONFIG.DEFAULT_Z_POS_MM)
        # wait for the operation to finish
        t0 = time.time() 
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 5:
                print('z return timeout, the program will exit')

    async def snap_image(self, channel=0, intensity=100, exposure_time=100, full_frame=False):
        # turn off the illumination if it is on
        need_to_turn_illumination_back = False
        if self.liveController.illumination_on:
            need_to_turn_illumination_back = True
            self.liveController.turn_off_illumination()
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)
        self.camera.set_exposure_time(exposure_time)
        self.liveController.set_illumination(channel, intensity)
        self.liveController.turn_on_illumination()
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)

        if self.is_simulation:
            await self.send_trigger_simulation(channel, intensity, exposure_time)
        else:
            self.camera.send_trigger()
        await asyncio.sleep(0.005)

        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)

        gray_img = self.camera.read_frame()
        # Apply rotation and flip first
        gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)
        
        # In simulation mode, resize small images to expected camera resolution
        if self.is_simulation:
            height, width = gray_img.shape[:2]
            # If image is too small, resize it to expected camera dimensions
            expected_width = 3000  # Expected camera width
            expected_height = 3000  # Expected camera height
            if width < expected_width or height < expected_height:
                gray_img = cv2.resize(gray_img, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
        
        # Return full frame if requested, otherwise crop using configuration settings
        if full_frame:
            result_img = gray_img
        else:
            # Crop using configuration-based dimensions with proper bounds checking
            crop_height = CONFIG.Acquisition.CROP_HEIGHT
            crop_width = CONFIG.Acquisition.CROP_WIDTH
            height, width = gray_img.shape[:2]
            start_x = width // 2 - crop_width // 2
            start_y = height // 2 - crop_height // 2
            
            # Add bounds checking
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, start_x + crop_width)
            end_y = min(height, start_y + crop_height)
            
            result_img = gray_img[start_y:end_y, start_x:end_x]

        if not need_to_turn_illumination_back:
            self.liveController.turn_off_illumination()
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)

        return result_img
    
    async def get_camera_frame_simulation(self, channel=0, intensity=100, exposure_time=100):
        self.camera.set_exposure_time(exposure_time)
        self.liveController.set_illumination(channel, intensity)
        await self.send_trigger_simulation(channel, intensity, exposure_time)
        gray_img = self.camera.read_frame() 
        gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)
        
        # In simulation mode, resize small images to expected camera resolution
        height, width = gray_img.shape[:2]
        # If image is too small, resize it to expected camera dimensions
        expected_width = 3000  # Expected camera width
        expected_height = 3000  # Expected camera height
        if width < expected_width or height < expected_height:
            gray_img = cv2.resize(gray_img, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
        
        return gray_img

    def get_camera_frame(self, channel=0, intensity=100, exposure_time=100):
        try:
            self.camera.send_trigger()
            gray_img = self.camera.read_frame()
            if gray_img is None:
                print(f"Warning: read_frame() returned None for channel {channel}")
                # Return a placeholder image instead of None to prevent crashes
                return np.zeros((self.camera.Height, self.camera.Width), dtype=np.uint8)
            gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)
            return gray_img
        except Exception as e:
            print(f"Error in get_camera_frame: {e}")
            # Return a placeholder image on error
            return np.zeros((self.camera.Height, self.camera.Width), dtype=np.uint8)
    

    def close(self):
        # In simulation mode, skip stage movements to avoid delays
        if self.is_simulation:
            print("Simulation mode: Skipping stage close operations")
            # Close only essential components that don't cause delays
            if hasattr(self, 'liveController'):
                # LiveController doesn't have close method, just stop live
                if hasattr(self.liveController, 'stop_live'):
                    self.liveController.stop_live()
            if hasattr(self, 'camera'):
                self.camera.close()
            return
        
        # Normal close operations for real hardware
        print("closing the system")
        if hasattr(self, 'liveController'):
            # LiveController doesn't have close method, just stop live
            if hasattr(self.liveController, 'stop_live'):
                self.liveController.stop_live()
        if hasattr(self, 'camera'):
            self.camera.close()
        
        # Move to safe position synchronously (no threading)
        if hasattr(self, 'navigationController') and hasattr(self, 'microcontroller'):
            try:
                self.navigationController.move_x_to(30)
                while self.microcontroller.is_busy():
                    time.sleep(0.005)
                self.navigationController.move_y_to(30)
                while self.microcontroller.is_busy():
                    time.sleep(0.005)
            except Exception as e:
                print(f"Error moving to safe position during close: {e}")
        
        if hasattr(self, 'camera_focus'):
            self.camera_focus.close()
        
        if hasattr(self, 'microcontroller'):
            self.microcontroller.close()

    def set_stage_velocity(self, velocity_x_mm_per_s=None, velocity_y_mm_per_s=None):
        """
        Set the maximum velocity for X and Y stage axes.
        
        Args:
            velocity_x_mm_per_s (float, optional): Maximum velocity for X axis in mm/s. 
                                                 If None, uses default from configuration.
            velocity_y_mm_per_s (float, optional): Maximum velocity for Y axis in mm/s.
                                                 If None, uses default from configuration.
        
        Returns:
            dict: Status and current velocity settings
        """
        # Use default values from configuration if not specified
        if velocity_x_mm_per_s is None:
            velocity_x_mm_per_s = CONFIG.MAX_VELOCITY_X_MM
        if velocity_y_mm_per_s is None:
            velocity_y_mm_per_s = CONFIG.MAX_VELOCITY_Y_MM
            
        # Validate velocity ranges (microcontroller limit is 65535/100 = 655.35 mm/s)
        max_velocity_limit = 655.35
        if velocity_x_mm_per_s > max_velocity_limit or velocity_x_mm_per_s <= 0:
            return {
                "success": False,
                "message": f"X velocity must be between 0 and {max_velocity_limit} mm/s (exclusive of 0)",
                "velocity_x_mm_per_s": velocity_x_mm_per_s,
                "velocity_y_mm_per_s": velocity_y_mm_per_s
            }
        if velocity_y_mm_per_s > max_velocity_limit or velocity_y_mm_per_s <= 0:
            return {
                "success": False,
                "message": f"Y velocity must be between 0 and {max_velocity_limit} mm/s (exclusive of 0)",
                "velocity_x_mm_per_s": velocity_x_mm_per_s,
                "velocity_y_mm_per_s": velocity_y_mm_per_s
            }
        
        try:
            # Set X axis velocity (keeping default acceleration)
            self.microcontroller.set_max_velocity_acceleration(
                microcontroller.AXIS.X, velocity_x_mm_per_s, CONFIG.MAX_ACCELERATION_X_MM
            )
            self.microcontroller.wait_till_operation_is_completed()
            
            # Set Y axis velocity (keeping default acceleration)  
            self.microcontroller.set_max_velocity_acceleration(
                microcontroller.AXIS.Y, velocity_y_mm_per_s, CONFIG.MAX_ACCELERATION_Y_MM
            )
            self.microcontroller.wait_till_operation_is_completed()
            
            return {
                "success": True,
                "message": "Stage velocity updated successfully",
                "velocity_x_mm_per_s": velocity_x_mm_per_s,
                "velocity_y_mm_per_s": velocity_y_mm_per_s,
                "acceleration_x_mm_per_s2": CONFIG.MAX_ACCELERATION_X_MM,
                "acceleration_y_mm_per_s2": CONFIG.MAX_ACCELERATION_Y_MM
            }
            
        except Exception as e:
            return {
                "success": False, 
                "message": f"Failed to set stage velocity: {str(e)}",
                "velocity_x_mm_per_s": velocity_x_mm_per_s,
                "velocity_y_mm_per_s": velocity_y_mm_per_s
            }

    async def normal_scan_with_stitching(self, start_x_mm, start_y_mm, Nx, Ny, dx_mm, dy_mm, 
                                        illumination_settings=None, do_contrast_autofocus=False, 
                                        do_reflection_af=False, action_ID='normal_scan_stitching',
                                        timepoint=0, fileset_name=None):
        """
        Normal scan with live stitching to OME-Zarr canvas.
        
        Args:
            start_x_mm (float): Starting X position in mm
            start_y_mm (float): Starting Y position in mm
            Nx (int): Number of positions in X
            Ny (int): Number of positions in Y
            dx_mm (float): Interval between positions in X (mm)
            dy_mm (float): Interval between positions in Y (mm)
            illumination_settings (list): List of channel settings
            do_contrast_autofocus (bool): Whether to perform contrast-based autofocus
            do_reflection_af (bool): Whether to perform reflection-based autofocus
            action_ID (str): Identifier for this scan
            timepoint (int): Timepoint index for the scan (default 0)
            fileset_name (str, optional): Name of the zarr fileset to use. If None, uses active fileset
        """
        if illumination_settings is None:
            illumination_settings = [
                {'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100}
            ]
        
        
        # Switch to specified fileset if provided
        if fileset_name:
            self.set_active_zarr_fileset(fileset_name)
        
        # Ensure we have an active canvas
        canvas = self.get_active_canvas()
        if canvas is None:
            raise RuntimeError("No zarr canvas available for stitching")
        
        # Validate that all requested channels are available in the zarr canvas
        channel_map = ChannelMapper.get_human_to_id_map()
        for settings in illumination_settings:
            channel_name = settings['channel']
            if channel_name not in canvas.channel_to_zarr_index:
                logging.error(f"Requested channel '{channel_name}' not found in zarr canvas!")
                logging.error(f"Available channels: {list(canvas.channel_to_zarr_index.keys())}")
                raise ValueError(f"Channel '{channel_name}' not available in zarr canvas")
        
        # Start the background stitching task
        await canvas.start_stitching()
        
        try:
            self.is_busy = True
            self.scan_stop_requested = False  # Reset stop flag at start of scan
            logging.info(f'Starting normal scan with stitching: {Nx}x{Ny} positions, dx={dx_mm}mm, dy={dy_mm}mm, timepoint={timepoint}')
            
            # Map channel names to indices
            channel_map = ChannelMapper.get_human_to_id_map()
            
            # Scan pattern: snake pattern for efficiency
            for i in range(Ny):
                # Check for stop request before each row
                if self.scan_stop_requested:
                    logging.info("Scan stopped by user request")
                    self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                    break
                    
                for j in range(Nx):
                    # Check for stop request before each position
                    if self.scan_stop_requested:
                        logging.info("Scan stopped by user request")
                        self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                        break
                    # Calculate position (snake pattern - reverse X on odd rows)
                    if i % 2 == 0:
                        x_idx = j
                    else:
                        x_idx = Nx - 1 - j
                    
                    x_mm = start_x_mm + x_idx * dx_mm
                    y_mm = start_y_mm + i * dy_mm
                    
                    # Move to position
                    self.navigationController.move_x_to(x_mm)
                    self.navigationController.move_y_to(y_mm)
                    while self.microcontroller.is_busy():
                        await asyncio.sleep(0.005)
                    
                    # Let stage settle
                    await asyncio.sleep(CONFIG.SCAN_STABILIZATION_TIME_MS_X / 1000)
                    
                    # Update position from microcontroller to get actual stage position
                    actual_x_mm, actual_y_mm, actual_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
                    
                    # Autofocus if requested (first position or periodically)
                    if do_reflection_af and (i == 0 and j == 0):
                        if hasattr(self, 'laserAutofocusController'):
                            await self.do_laser_autofocus()
                            # Update position again after autofocus
                            actual_x_mm, actual_y_mm, actual_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
                    elif do_contrast_autofocus and ((i * Nx + j) % CONFIG.Acquisition.NUMBER_OF_FOVS_PER_AF == 0):
                        await self.do_autofocus()
                        # Update position again after autofocus
                        actual_x_mm, actual_y_mm, actual_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
                    
                    # Acquire images for each channel
                    for idx, settings in enumerate(illumination_settings):
                        channel_name = settings['channel']
                        intensity = settings['intensity']
                        exposure_time = settings['exposure_time']
                        
                        # Get global channel index for snap_image (uses global channel IDs)
                        global_channel_idx = channel_map.get(channel_name, 0)
                        
                        # Get local zarr channel index (0, 1, 2, etc.)
                        try:
                            zarr_channel_idx = canvas.get_zarr_channel_index(channel_name)
                        except ValueError as e:
                            logging.error(f"Channel mapping error: {e}")
                            continue
                        
                        # Snap image using global channel ID with full frame for stitching
                        image = await self.snap_image(global_channel_idx, intensity, exposure_time, full_frame=True)
                        
                        # Convert to 8-bit if needed
                        if image.dtype != np.uint8:
                            # Scale to 8-bit
                            if image.dtype == np.uint16:
                                image = (image / 256).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)
                        
                        # Add to stitching queue using actual stage position and local zarr index
                        await canvas.add_image_async(
                            image, actual_x_mm, actual_y_mm, 
                            channel_idx=zarr_channel_idx,  # Use local zarr index
                            z_idx=0,
                            timepoint=timepoint  # Add timepoint parameter
                        )
                        
                        logging.info(f'Added image at actual position ({actual_x_mm:.2f}, {actual_y_mm:.2f}) for channel {channel_name} (global_id: {global_channel_idx}, zarr_idx: {zarr_channel_idx}), timepoint={timepoint} (intended: {x_mm:.2f}, {y_mm:.2f})')
            
            logging.info('Normal scan with stitching completed')
            
            # Give the stitching queue a moment to process any final images
            logging.info('Allowing time for final images to be queued for stitching...')
            await asyncio.sleep(0.3)  # Wait 500ms to ensure all images are queued
            
            # Save a preview image from the lowest resolution scale
            # try:
            #     preview_path = self.zarr_canvas.save_preview(action_ID)
            #     if preview_path:
            #         logging.info(f'Canvas preview saved at: {preview_path}')
            # except Exception as e:
            #     logging.error(f'Failed to save canvas preview: {e}')
            
        finally:
            self.is_busy = False
            # Stop the stitching task (this will now process all remaining images)
            await canvas.stop_stitching()
    
    async def _initialize_zarr_canvas(self):
        """Initialize the Zarr canvas for stitching with all available channels."""
        # Get ALL channel names from ChannelMapper (not just from current illumination_settings)
        all_channels = ChannelMapper.get_all_human_names()
        
        # Check if we already have a canvas initialized with all channels
        if hasattr(self, 'zarr_canvas') and self.zarr_canvas is not None:
            # Check if the existing canvas has all channels
            if set(self.zarr_canvas.channels) == set(all_channels):
                logging.info('Using existing Zarr canvas with all channels')
                return
            else:
                # Close existing canvas and create new one with all channels
                logging.info('Closing existing canvas and creating new one with all channels')
                self.zarr_canvas.close()
        
        # Get the base path from environment variable
        zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
        
        # Define stage limits (from CONFIG)
        stage_limits = {
            'x_positive': 120,  # mm
            'x_negative': 0,    # mm
            'y_positive': 86,   # mm
            'y_negative': 0,    # mm
            'z_positive': 6     # mm
        }
        
        # Create the canvas with ALL available channels and optimized timepoint allocation
        self.zarr_canvas = ZarrCanvas(
            base_path=zarr_path,
            pixel_size_xy_um=self.pixel_size_xy,
            stage_limits=stage_limits,
            channels=all_channels,  # Use all channels from ChannelMapper
            rotation_angle_deg=CONFIG.STITCHING_ROTATION_ANGLE_DEG,
            initial_timepoints=20,  # Pre-allocate 20 timepoints to avoid resize delays
            timepoint_expansion_chunk=10  # Expand by 10 timepoints when needed
        )
        
        # Initialize the OME-Zarr structure
        self.zarr_canvas.initialize_canvas()
        logging.info(f'Initialized Zarr canvas at {zarr_path} with ALL channels: {len(all_channels)} channels')
        logging.info(f'Available channels: {all_channels}')
        logging.info(f'Channel to zarr index mapping: {self.zarr_canvas.channel_to_zarr_index}')
    
    def get_stitched_region(self, center_x_mm, center_y_mm, width_mm, height_mm, 
                           scale_level=0, channel_name='BF LED matrix full', timepoint=0):
        """
        Get a region from the stitched canvas.
        
        Args:
            center_x_mm (float): Center X position in mm
            center_y_mm (float): Center Y position in mm
            width_mm (float): Width of region in mm
            height_mm (float): Height of region in mm
            scale_level (int): Scale level (0=full res, 1=1/4, 2=1/16, etc)
            channel_name (str): Name of channel to retrieve
            timepoint (int): Timepoint index (default 0)
            
        Returns:
            np.ndarray: The requested region
        """
        if not hasattr(self, 'zarr_canvas') or self.zarr_canvas is None:
            raise RuntimeError("Zarr canvas not initialized. Run a scan first.")
        
        # Get the region using the new channel name method
        region = self.zarr_canvas.get_canvas_region_by_channel_name(
            center_x_mm, center_y_mm, width_mm, height_mm,
            channel_name, scale=scale_level, timepoint=timepoint
        )
        
        if region is None:
            logging.warning(f"Failed to get region for channel {channel_name}, timepoint {timepoint}")
            return None
        
        return region
    
    async def initialize_zarr_canvas_if_needed(self):
        """
        Initialize the zarr canvas with all channels if not already initialized.
        This can be called early in the service lifecycle to ensure the canvas is ready.
        """
        if not hasattr(self, 'zarr_canvas') or self.zarr_canvas is None:
            # Initialize with all channels from ChannelMapper
            await self._initialize_zarr_canvas()
            logging.info("Zarr canvas pre-initialized with all available channels")

    def _initialize_empty_canvas(self, fileset_name="default"):
        """Initialize an empty zarr canvas with a specific name.
        
        Args:
            fileset_name: Name for the zarr fileset (default: "default")
        """
        try:
            # Get the base path from environment variable
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            logging.info(f'ZARR_PATH environment variable: {os.getenv("ZARR_PATH", "not set (using default)")}')
            logging.info(f'Initializing empty zarr canvas at: {zarr_path}')
            
            # Check if the directory is writable
            zarr_dir = Path(zarr_path)
            
            # The actual zarr file will be at zarr_path/{fileset_name}.zarr
            actual_zarr_path = zarr_dir / f"{fileset_name}.zarr"
            logging.info(f'Zarr file will be created at: {actual_zarr_path}')
            
            if not zarr_dir.exists():
                logging.info(f'Creating zarr base directory: {zarr_dir}')
                try:
                    zarr_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError as e:
                    raise RuntimeError(f"Cannot create zarr directory {zarr_dir}: Permission denied. Please check directory permissions or set ZARR_PATH to a writable location.")
            
            # Test write permissions
            test_file = zarr_dir / 'test_write_permission.tmp'
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                test_file.unlink()  # Remove test file
                logging.info(f'Write permission test successful for {zarr_dir}')
            except (PermissionError, OSError) as e:
                raise RuntimeError(f"No write permission for zarr directory {zarr_dir}: {e}. Please check directory permissions or set ZARR_PATH to a writable location.")
            
            # Define stage limits (from README)
            stage_limits = {
                'x_positive': 120,  # mm
                'x_negative': 0,    # mm
                'y_positive': 86,   # mm
                'y_negative': 0,    # mm
                'z_positive': 6     # mm
            }
            
            # Use ALL available channels from ChannelMapper for initialization
            default_channels = ChannelMapper.get_all_human_names()
            logging.info(f'Initializing zarr canvas with channels from ChannelMapper: {len(default_channels)} channels')
            
            # Create the canvas with optimized timepoint allocation
            canvas = ZarrCanvas(
                base_path=zarr_path,
                pixel_size_xy_um=self.pixel_size_xy,
                stage_limits=stage_limits,
                channels=default_channels,
                rotation_angle_deg=CONFIG.STITCHING_ROTATION_ANGLE_DEG,
                initial_timepoints=20,  # Pre-allocate 20 timepoints to avoid resize delays
                timepoint_expansion_chunk=10,  # Expand by 10 timepoints when needed
                fileset_name=fileset_name  # Use the provided fileset name
            )
            
            # Initialize the OME-Zarr structure
            canvas.initialize_canvas()
            
            # Add to tracking dictionary and set as active
            self.zarr_canvases[fileset_name] = canvas
            self.active_canvas_name = fileset_name
            self.zarr_canvas = canvas  # For backward compatibility
            
            logging.info(f'Successfully initialized Zarr canvas "{fileset_name}" at {canvas.zarr_path} with ALL channels: {len(default_channels)} channels')
            logging.info(f'Available channels: {default_channels}')
            logging.info(f'Pixel size: {self.pixel_size_xy:.3f} µm')
            
        except Exception as e:
            logging.error(f'Failed to initialize zarr canvas "{fileset_name}": {e}')
            logging.info('To fix this, either:')
            logging.info('  1. Set ZARR_PATH environment variable to a writable directory (e.g., export ZARR_PATH=/tmp/zarr_canvas)')
            logging.info('  2. Create the directory and set proper permissions')
            logging.info('  3. Run the application with appropriate permissions')

    async def quick_scan_with_stitching(self, wellplate_type='96', exposure_time=5, intensity=50, 
                                      fps_target=10, action_ID='quick_scan_stitching',
                                      n_stripes=4, stripe_width_mm=4.0, dy_mm=0.9, velocity_scan_mm_per_s=7.0,
                                      do_contrast_autofocus=False, do_reflection_af=False, timepoint=0, fileset_name=None):
        """
        Quick scan with live stitching to OME-Zarr canvas - brightfield only.
        Uses 4-stripe × 4 mm scanning pattern with serpentine motion per well.
        
        Args:
            wellplate_type (str): Well plate type ('6', '12', '24', '96', '384')
            exposure_time (float): Camera exposure time in ms (max 30ms)
            intensity (float): Brightfield LED intensity (0-100)
            fps_target (int): Target frame rate for acquisition (default 10fps)
            action_ID (str): Identifier for this scan
            n_stripes (int): Number of stripes per well (default 4)
            stripe_width_mm (float): Length of each stripe inside a well in mm (default 4.0)
            dy_mm (float): Y increment between stripes in mm (default 0.9)
            velocity_scan_mm_per_s (float): Stage velocity during stripe scanning in mm/s (default 7.0)
            do_contrast_autofocus (bool): Whether to perform contrast-based autofocus
            do_reflection_af (bool): Whether to perform reflection-based autofocus
            timepoint (int): Timepoint index for the scan (default 0)
            fileset_name (str, optional): Name of the zarr fileset to use. If None, uses active fileset
        """
        
        # Validate exposure time
        if exposure_time > 30:
            raise ValueError("Quick scan exposure time must not exceed 30ms")
        
        # Get well plate format configuration
        if wellplate_type == '6':
            wellplate_format = WELLPLATE_FORMAT_6
            max_rows = 2  # A-B
            max_cols = 3  # 1-3
        elif wellplate_type == '12':
            wellplate_format = WELLPLATE_FORMAT_12
            max_rows = 3  # A-C
            max_cols = 4  # 1-4
        elif wellplate_type == '24':
            wellplate_format = WELLPLATE_FORMAT_24
            max_rows = 4  # A-D
            max_cols = 6  # 1-6
        elif wellplate_type == '96':
            wellplate_format = WELLPLATE_FORMAT_96
            max_rows = 8  # A-H
            max_cols = 12  # 1-12
        elif wellplate_type == '384':
            wellplate_format = WELLPLATE_FORMAT_384
            max_rows = 16  # A-P
            max_cols = 24  # 1-24
        else:
            # Default to 96-well plate if unsupported type is provided
            wellplate_format = WELLPLATE_FORMAT_96
            max_rows = 8
            max_cols = 12
            wellplate_type = '96'
        
        
        # Switch to specified fileset if provided
        if fileset_name:
            self.set_active_zarr_fileset(fileset_name)
        
        # Ensure we have an active canvas
        canvas = self.get_active_canvas()
        if canvas is None:
            raise RuntimeError("No zarr canvas available for stitching")
        
        # Validate that brightfield channel is available in the zarr canvas
        channel_name = 'BF LED matrix full'
        if channel_name not in canvas.channel_to_zarr_index:
            logging.error(f"Brightfield channel '{channel_name}' not found in zarr canvas!")
            logging.error(f"Available channels: {list(canvas.channel_to_zarr_index.keys())}")
            raise ValueError(f"Channel '{channel_name}' not available in zarr canvas")
        
        # Get zarr channel index for brightfield
        zarr_channel_idx = canvas.get_zarr_channel_index(channel_name)
        
        # Start the background stitching task
        await canvas.start_stitching()
        
        # Store original velocity settings for restoration
        original_velocity_result = self.set_stage_velocity()
        original_velocity_x = original_velocity_result.get('velocity_x_mm_per_s', CONFIG.MAX_VELOCITY_X_MM)
        original_velocity_y = original_velocity_result.get('velocity_y_mm_per_s', CONFIG.MAX_VELOCITY_Y_MM)
        
        # Define velocity constants
        HIGH_SPEED_VELOCITY_MM_PER_S = 30.0  # For moving between wells
        scan_velocity = velocity_scan_mm_per_s  # For scanning within wells
        
        try:
            self.is_busy = True
            self.scan_stop_requested = False  # Reset stop flag at start of scan
            logging.info(f'Starting quick scan with stitching: {wellplate_type} well plate, {n_stripes} stripes × {stripe_width_mm}mm, dy={dy_mm}mm, scan_velocity={scan_velocity}mm/s, fps={fps_target}, timepoint={timepoint}')
            
            if do_contrast_autofocus:
                logging.info('Contrast autofocus enabled for quick scan')
            if do_reflection_af:
                logging.info('Reflection autofocus enabled for quick scan')
            
            # 1. Before starting scanning, read the position of z axis
            original_x_mm, original_y_mm, original_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
            logging.info(f'Original Z position before autofocus: {original_z_mm:.3f}mm')
            
            # Set camera exposure time
            self.camera.set_exposure_time(exposure_time)
            
            # Calculate well plate parameters
            well_spacing = wellplate_format.WELL_SPACING_MM
            x_offset = CONFIG.WELLPLATE_OFFSET_X_MM
            y_offset = CONFIG.WELLPLATE_OFFSET_Y_MM
            
            # Calculate frame acquisition timing
            frame_interval = 1.0 / fps_target  # seconds between frames
            
            # Get software limits for safety
            limit_x_pos = CONFIG.SOFTWARE_POS_LIMIT.X_POSITIVE
            limit_x_neg = CONFIG.SOFTWARE_POS_LIMIT.X_NEGATIVE
            limit_y_pos = CONFIG.SOFTWARE_POS_LIMIT.Y_POSITIVE
            limit_y_neg = CONFIG.SOFTWARE_POS_LIMIT.Y_NEGATIVE
            
            # Scan each well using snake pattern for rows
            for row_idx in range(max_rows):
                if self.scan_stop_requested:
                    logging.info("Quick scan stopped by user request")
                    self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                    break
                    
                row_letter = chr(ord('A') + row_idx)
                
                # Snake pattern: alternate direction for each row
                if row_idx % 2 == 0:
                    # Even rows (0, 2, 4...): left to right (A1 → A12, C1 → C12, etc.)
                    col_range = range(max_cols)
                    direction = "left-to-right"
                else:
                    # Odd rows (1, 3, 5...): right to left (B12 → B1, D12 → D1, etc.)
                    col_range = range(max_cols - 1, -1, -1)
                    direction = "right-to-left"
                
                logging.info(f'Scanning row {row_letter} ({direction})')
                
                for col_idx in col_range:
                    if self.scan_stop_requested:
                        logging.info("Quick scan stopped by user request")
                        self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                        break
                        
                    col_number = col_idx + 1
                    well_name = f"{row_letter}{col_number}"
                    
                    # Calculate well center position
                    well_center_x = wellplate_format.A1_X_MM + x_offset + col_idx * well_spacing
                    well_center_y = wellplate_format.A1_Y_MM + y_offset + row_idx * well_spacing
                    
                    # Calculate stripe boundaries within the well
                    stripe_half_width = stripe_width_mm / 2
                    stripe_start_x = well_center_x - stripe_half_width
                    stripe_end_x = well_center_x + stripe_half_width
                    
                    # Clamp stripe boundaries to software limits
                    stripe_start_x = max(min(stripe_start_x, limit_x_pos), limit_x_neg)
                    stripe_end_x = max(min(stripe_end_x, limit_x_pos), limit_x_neg)
                    
                    # Calculate starting Y position for stripes (centered around well)
                    stripe_start_y = well_center_y - ((n_stripes - 1) * dy_mm) / 2
                    
                    logging.info(f'Scanning well {well_name}: {n_stripes} stripes × {stripe_width_mm}mm at Y positions starting from {stripe_start_y:.2f}mm')
                    
                    # Autofocus workflow: move to well center first if autofocus is requested
                    if do_contrast_autofocus or do_reflection_af:
                        logging.info(f'Moving to well {well_name} center for autofocus')
                        
                        # Move to well center using optimized async method
                        await self.move_to_well_center_for_autofocus(row_letter, col_number, wellplate_type, HIGH_SPEED_VELOCITY_MM_PER_S)
                        
                        # Perform autofocus
                        if do_reflection_af:
                            logging.info(f'Performing reflection autofocus at well {well_name}')
                            if hasattr(self, 'laserAutofocusController'):
                                await self.do_laser_autofocus()
                            else:
                                logging.warning('Reflection autofocus requested but laserAutofocusController not available')
                        elif do_contrast_autofocus:
                            logging.info(f'Performing contrast autofocus at well {well_name}')
                            await self.do_autofocus()
                        
                        # Update position after autofocus
                        actual_x_mm, actual_y_mm, actual_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
                        logging.info(f'Autofocus completed at well {well_name}, current position: ({actual_x_mm:.2f}, {actual_y_mm:.2f}, {actual_z_mm:.2f})')
                    
                    # Move to well stripe start position at high speed
                    await self._move_to_well_at_high_speed(well_name, stripe_start_x, stripe_start_y, 
                                                            HIGH_SPEED_VELOCITY_MM_PER_S, limit_y_neg, limit_y_pos)
                    
                    # Set scan velocity for stripe scanning
                    velocity_result = self.set_stage_velocity(scan_velocity, scan_velocity)
                    if not velocity_result['success']:
                        logging.warning(f"Failed to set scanning velocity: {velocity_result['message']}")
                    
                    # Scan all stripes within the well with continuous frame acquisition
                    total_frames = await self._scan_well_with_continuous_acquisition(
                        well_name, n_stripes, stripe_start_x, stripe_end_x, 
                        stripe_start_y, dy_mm, intensity, frame_interval, 
                        zarr_channel_idx, limit_y_neg, limit_y_pos, timepoint=timepoint)
                    
                    logging.info(f'Well {well_name} completed with {n_stripes} stripes, total frames: {total_frames}')
                    
                    # 3. After scanning for this well is done, move the z axis back to the remembered position
                    if do_contrast_autofocus or do_reflection_af:
                        logging.info(f'Restoring Z position to original: {original_z_mm:.3f}mm')
                        self.navigationController.move_z_to(original_z_mm)
                        while self.microcontroller.is_busy():
                            await asyncio.sleep(0.005)
                        
            
            logging.info('Quick scan with stitching completed')
            
            # Allow time for final images to be queued for stitching
            logging.info('Allowing time for final images to be queued for stitching...')
            await asyncio.sleep(0.5)
            
        finally:
            self.is_busy = False
            
            # Turn off illumination if still on
            self.liveController.turn_off_illumination()
            
            # Restore original velocity settings
            self._restore_original_velocity(original_velocity_x, original_velocity_y)
            
            # Stop the stitching task
            await self.zarr_canvas.stop_stitching()
    
    async def _move_to_well_at_high_speed(self, well_name, start_x, start_y, high_speed_velocity, limit_y_neg, limit_y_pos):
        """Move to well at high speed (30 mm/s) for efficient inter-well movement."""
        logging.info(f'Moving to well {well_name} at high speed ({high_speed_velocity} mm/s)')
        
        velocity_result = self.set_stage_velocity(high_speed_velocity, high_speed_velocity)
        if not velocity_result['success']:
            logging.warning(f"Failed to set high-speed velocity: {velocity_result['message']}")
        
        # Clamp Y position to limits
        clamped_y = max(min(start_y, limit_y_pos), limit_y_neg)
        
        # Move to first stripe start position
        self.navigationController.move_x_to(start_x)
        self.navigationController.move_y_to(clamped_y)
        
        # Wait for movement to complete
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)
        
        logging.info(f'Moved to well {well_name} start position ({start_x:.2f}, {clamped_y:.2f})')
    
    async def _scan_well_with_continuous_acquisition(self, well_name, n_stripes, stripe_start_x, stripe_end_x, 
                                                   stripe_start_y, dy_mm, intensity, frame_interval, 
                                                   zarr_channel_idx, limit_y_neg, limit_y_pos, timepoint=0):
        """Scan all stripes within a well with continuous frame acquisition."""
        total_frames = 0
        
        # Turn on brightfield illumination once for the entire well
        self.liveController.set_illumination(0, intensity)  # Channel 0 = brightfield
        await asyncio.sleep(0.01)  # Small delay for illumination to stabilize
        self.liveController.turn_on_illumination()
        
        # Start continuous frame acquisition
        last_frame_time = time.time()
        
        try:
            for stripe_idx in range(n_stripes):
                if self.scan_stop_requested:
                    logging.info("Quick scan stopped by user request")
                    self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                    break
                    
                stripe_y = stripe_start_y + stripe_idx * dy_mm
                stripe_y = max(min(stripe_y, limit_y_pos), limit_y_neg)
                
                # Serpentine pattern: alternate direction for each stripe
                if stripe_idx % 2 == 0:
                    # Even stripes: left to right
                    start_x, end_x = stripe_start_x, stripe_end_x
                    direction = "left-to-right"
                else:
                    # Odd stripes: right to left
                    start_x, end_x = stripe_end_x, stripe_start_x
                    direction = "right-to-left"
                
                logging.info(f'Well {well_name}, stripe {stripe_idx + 1}/{n_stripes} ({direction}) from X={start_x:.2f}mm to X={end_x:.2f}mm at Y={stripe_y:.2f}mm')
                
                # Move to stripe start position
                self.navigationController.move_x_to(start_x)
                self.navigationController.move_y_to(stripe_y)
                
                # Wait for positioning to complete
                while self.microcontroller.is_busy():
                    await asyncio.sleep(0.005)
                
                # Let stage settle briefly
                await asyncio.sleep(0.05)
                
                # Start continuous movement to end of stripe
                self.navigationController.move_x_to(end_x)
                
                # Acquire frames while moving along this stripe
                stripe_frames = 0
                while self.microcontroller.is_busy():
                    if self.scan_stop_requested:
                        logging.info("Quick scan stopped during stripe movement")
                        self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                        break
                        
                    current_time = time.time()
                    
                    # Check if it's time for next frame
                    if current_time - last_frame_time >= frame_interval:
                        frame_acquired = await self._acquire_and_process_frame(zarr_channel_idx, timepoint)
                        if frame_acquired:
                            stripe_frames += 1
                            total_frames += 1
                        # Update timing AFTER frame acquisition completes, not before
                        last_frame_time = time.time()
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.001)
                
                logging.info(f'Well {well_name}, stripe {stripe_idx + 1}/{n_stripes} completed, acquired {stripe_frames} frames')
                
                # Continue to next stripe without stopping illumination or frame acquisition
                
        finally:
            # Turn off illumination only when done with the entire well
            self.liveController.turn_off_illumination()
        
        return total_frames
    
    async def _scan_single_stripe(self, start_x, end_x, stripe_y, intensity, frame_interval, zarr_channel_idx):
        """Scan a single stripe and return the number of frames acquired."""
        # Move to stripe start position
        self.navigationController.move_x_to(start_x)
        self.navigationController.move_y_to(stripe_y)
        
        # Wait for positioning to complete
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)
        
        # Let stage settle
        await asyncio.sleep(0.1)
        
        # Turn on brightfield illumination
        self.liveController.set_illumination(0, intensity)  # Channel 0 = brightfield
        await asyncio.sleep(0.01)  # Small delay for illumination to stabilize
        self.liveController.turn_on_illumination()
        
        # Start continuous movement to end of stripe
        self.navigationController.move_x_to(end_x)
        
        # Acquire frames while moving
        frames_acquired = 0
        last_frame_time = time.time()
        
        while self.microcontroller.is_busy():
            if self.scan_stop_requested:
                logging.info("Quick scan stopped during stripe movement")
                self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
                break
                
            current_time = time.time()
            
            # Check if it's time for next frame
            if current_time - last_frame_time >= frame_interval:
                frame_acquired = await self._acquire_and_process_frame(zarr_channel_idx)
                if frame_acquired:
                    frames_acquired += 1
                # Update timing AFTER frame acquisition completes, not before
                last_frame_time = time.time()
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.001)
        
        # Turn off illumination
        self.liveController.turn_off_illumination()
        
        return frames_acquired
    
    async def _acquire_and_process_frame(self, zarr_channel_idx, timepoint=0):
        """Acquire a single frame and add it to the stitching queue."""
        # Get position before frame acquisition
        pos_before_x_mm, pos_before_y_mm, pos_before_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
        
        # Read frame from camera
        self.camera.send_trigger()
        gray_img = self.camera.read_frame()
        
        # Get position after frame acquisition
        pos_after_x_mm, pos_after_y_mm, pos_after_z_mm, _ = self.navigationController.update_pos(self.microcontroller)
        
        # Calculate average position during frame acquisition
        avg_x_mm = (pos_before_x_mm + pos_after_x_mm) / 2.0
        avg_y_mm = (pos_before_y_mm + pos_after_y_mm) / 2.0
        
        if gray_img is not None:
            # Process and add image to stitching queue
            processed_img = self._process_frame_for_stitching(gray_img)
            
            await self._add_image_to_zarr_quick(
                processed_img, avg_x_mm, avg_y_mm,
                channel_idx=zarr_channel_idx, z_idx=0, timepoint=timepoint
            )
            
            logging.debug(f'Acquired frame at average position ({avg_x_mm:.2f}, {avg_y_mm:.2f}), timepoint={timepoint}')
            return True
        
        return False
    
    def _process_frame_for_stitching(self, gray_img):
        """Process a frame for stitching (resize, rotate, flip, convert to 8-bit)."""
        # Immediately rescale to scale1 resolution (1/4 of original)
        original_height, original_width = gray_img.shape[:2]
        scale1_width = original_width // 4
        scale1_height = original_height // 4
        
        # Resize image to scale1 resolution
        scaled_img = cv2.resize(gray_img, (scale1_width, scale1_height), interpolation=cv2.INTER_AREA)
        
        # Apply rotate and flip transformations
        processed_img = rotate_and_flip_image(
            scaled_img,
            rotate_image_angle=self.camera.rotate_image_angle,
            flip_image=self.camera.flip_image
        )
        
        # Convert to 8-bit if needed
        if processed_img.dtype != np.uint8:
            if processed_img.dtype == np.uint16:
                processed_img = (processed_img / 256).astype(np.uint8)
            else:
                processed_img = processed_img.astype(np.uint8)
        
        return processed_img
    
    def _restore_original_velocity(self, original_velocity_x, original_velocity_y):
        """Restore the original stage velocity settings."""
        restore_result = self.set_stage_velocity(original_velocity_x, original_velocity_y)
        if restore_result['success']:
            logging.info(f'Restored original stage velocity: X={original_velocity_x}mm/s, Y={original_velocity_y}mm/s')
        else:
            logging.warning(f'Failed to restore original stage velocity: {restore_result["message"]}')
    
    def stop_scan_and_stitching(self):
        """
        Stop any ongoing scanning and stitching processes.
        This will interrupt normal_scan_with_stitching and quick_scan_with_stitching.
        """
        self.scan_stop_requested = True
        logging.info("Scan stop requested - ongoing scans will be interrupted")
        self._restore_original_velocity(CONFIG.MAX_VELOCITY_X_MM, CONFIG.MAX_VELOCITY_Y_MM)
        return {"success": True, "message": "Scan stop requested"}
    
    async def _add_image_to_zarr_quick(self, image: np.ndarray, x_mm: float, y_mm: float,
                                     channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0):
        """
        Add image to zarr canvas for quick scan - only updates scales 1-5 (skips scale 0).
        The input image should already be at scale1 resolution (1/4 of original).
        Non-blocking queue operation to avoid FPS timing delays.
        """
        try:
            # Use put_nowait to avoid blocking the timing loop
            self.zarr_canvas.stitch_queue.put_nowait({
                'image': image.copy(),
                'x_mm': x_mm,
                'y_mm': y_mm,
                'channel_idx': channel_idx,
                'z_idx': z_idx,
                'timepoint': timepoint,
                'timestamp': time.time(),
                'quick_scan': True  # Flag to indicate this is for quick scan (scales 1-5 only)
            })
        except asyncio.QueueFull:
            # If queue is full, log warning but don't block timing
            logging.warning(f"Zarr stitching queue full, dropping frame at ({x_mm:.2f}, {y_mm:.2f})")

    def _cleanup_zarr_directory(self):
        # Clean up ZARR_PATH directory on startup
        zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
        if os.path.exists(zarr_path):
            try:
                shutil.rmtree(zarr_path)
                logging.info(f'Cleaned up ZARR_PATH directory: {zarr_path}')
            except Exception as e:
                logging.error(f'Failed to clean up ZARR_PATH directory: {e}')

    def create_zarr_fileset(self, fileset_name):
        """Create a new zarr fileset with the given name.
        
        Args:
            fileset_name: Name for the new fileset
            
        Returns:
            dict: Information about the created fileset
            
        Raises:
            ValueError: If fileset already exists
            RuntimeError: If fileset creation fails
        """
        if fileset_name in self.zarr_canvases:
            raise ValueError(f"Fileset '{fileset_name}' already exists")
        
        try:
            # Initialize new canvas
            self._initialize_empty_canvas(fileset_name)
            
            return {
                "fileset_name": fileset_name,
                "is_active": self.active_canvas_name == fileset_name,
                "message": f"Created new zarr fileset '{fileset_name}'"
            }
            
        except Exception as e:
            logging.error(f"Failed to create zarr fileset '{fileset_name}': {e}")
            raise RuntimeError(f"Failed to create fileset '{fileset_name}': {str(e)}") from e
    
    def list_zarr_filesets(self):
        """List all available zarr filesets.
        
        Returns:
            dict: List of filesets and their status
            
        Raises:
            RuntimeError: If listing filesets fails
        """
        try:
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            zarr_dir = Path(zarr_path)
            
            filesets = []
            
            # List from memory (already loaded)
            for name, canvas in self.zarr_canvases.items():
                filesets.append({
                    "name": name,
                    "is_active": name == self.active_canvas_name,
                    "loaded": True,
                    "path": str(canvas.zarr_path),
                    "channels": len(canvas.channels),
                    "timepoints": len(canvas.available_timepoints)
                })
            
            # Also check disk for any zarr directories not in memory
            if zarr_dir.exists():
                for item in zarr_dir.iterdir():
                    if item.is_dir() and item.suffix == '.zarr':
                        name = item.stem
                        if name not in self.zarr_canvases:
                            filesets.append({
                                "name": name,
                                "is_active": False,
                                "loaded": False,
                                "path": str(item)
                            })
            
            return {
                "filesets": filesets,
                "active_fileset": self.active_canvas_name,
                "total_count": len(filesets)
            }
            
        except Exception as e:
            logging.error(f"Failed to list zarr filesets: {e}")
            raise RuntimeError(f"Failed to list filesets: {str(e)}") from e
    
    def set_active_zarr_fileset(self, fileset_name):
        """Set the active zarr fileset for operations.
        
        Args:
            fileset_name: Name of the fileset to activate
            
        Returns:
            dict: Information about the activated fileset
            
        Raises:
            ValueError: If fileset not found
            RuntimeError: If activation fails
        """
        try:
            # Check if already active
            if self.active_canvas_name == fileset_name:
                return {
                    "message": f"Fileset '{fileset_name}' is already active",
                    "fileset_name": fileset_name,
                    "was_already_active": True
                }
            
            # Check if fileset exists in memory
            if fileset_name in self.zarr_canvases:
                self.active_canvas_name = fileset_name
                self.zarr_canvas = self.zarr_canvases[fileset_name]
                return {
                    "message": f"Activated fileset '{fileset_name}'",
                    "fileset_name": fileset_name,
                    "was_already_active": False
                }
            
            # Try to load from disk
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            zarr_dir = Path(zarr_path)
            fileset_path = zarr_dir / f"{fileset_name}.zarr"
            
            if fileset_path.exists():
                # Load the existing zarr canvas
                # Note: This is a simplified loading - you might need to properly restore the canvas
                self._initialize_empty_canvas(fileset_name)
                return {
                    "message": f"Loaded and activated fileset '{fileset_name}' from disk",
                    "fileset_name": fileset_name,
                    "was_already_active": False
                }
            
            raise ValueError(f"Fileset '{fileset_name}' not found")
            
        except Exception as e:
            logging.error(f"Failed to set active zarr fileset '{fileset_name}': {e}")
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"Failed to activate fileset '{fileset_name}': {str(e)}") from e
    
    def remove_zarr_fileset(self, fileset_name):
        """Remove a zarr fileset.
        
        Args:
            fileset_name: Name of the fileset to remove
            
        Returns:
            dict: Information about the removed fileset
            
        Raises:
            ValueError: If trying to remove active fileset
            RuntimeError: If removal fails
        """
        # Can't remove active fileset
        if self.active_canvas_name == fileset_name:
            raise ValueError(f"Cannot remove active fileset '{fileset_name}'. Please switch to another fileset first.")
        
        try:
            removed_from_memory = False
            removed_from_disk = False
            
            # Remove from memory if loaded
            if fileset_name in self.zarr_canvases:
                canvas = self.zarr_canvases[fileset_name]
                canvas.close()
                del self.zarr_canvases[fileset_name]
                removed_from_memory = True
            
            # Remove from disk
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            zarr_dir = Path(zarr_path)
            fileset_path = zarr_dir / f"{fileset_name}.zarr"
            
            if fileset_path.exists():
                shutil.rmtree(fileset_path)
                logging.info(f"Removed zarr fileset '{fileset_name}' from disk")
                removed_from_disk = True
            
            return {
                "message": f"Removed zarr fileset '{fileset_name}'",
                "fileset_name": fileset_name,
                "removed_from_memory": removed_from_memory,
                "removed_from_disk": removed_from_disk
            }
            
        except Exception as e:
            logging.error(f"Failed to remove zarr fileset '{fileset_name}': {e}")
            raise RuntimeError(f"Failed to remove fileset '{fileset_name}': {str(e)}") from e
    
    def get_active_canvas(self):
        """Get the currently active zarr canvas, ensuring one exists.
        
        Returns:
            ZarrCanvas: The active canvas, or None if no canvas is active
        """
        if self.zarr_canvas is None and self.active_canvas_name is None:
            # No canvas exists, create default one
            logging.info("No active zarr canvas, creating default fileset")
            self.create_zarr_fileset("default")
        
        return self.zarr_canvas

async def try_microscope():
    squid_controller = SquidController(is_simulation=False)

    custom_illumination_settings = [
        {'channel': 'BF LED matrix full', 'intensity': 35.0, 'exposure_time': 15.0},
        {'channel': 'Fluorescence 488 nm Ex', 'intensity': 50.0, 'exposure_time': 80.0},
        {'channel': 'Fluorescence 561 nm Ex', 'intensity': 75.0, 'exposure_time': 120.0}
    ]
    
    squid_controller.scan_well_plate_new(
        well_plate_type='96',
        illumination_settings=custom_illumination_settings,
        do_contrast_autofocus=False,
        do_reflection_af=True,
        scanning_zone=[(0,0),(1,1)],  # Scan wells A1 to B2
        Nx=2,
        Ny=2,
        action_ID='customIlluminationTest'
    )
    
    squid_controller.close()


if __name__ == "__main__":
    asyncio.run(try_microscope())

