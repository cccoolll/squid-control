import os 
# app specific libraries
import squid_control.control.core_reef as core
import squid_control.control.microcontroller as microcontroller
from squid_control.control.config import *
from squid_control.control.camera import get_camera
from squid_control.control.utils import rotate_and_flip_image
import cv2
import logging
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


class SquidController:
    fps_software_trigger= 10

    def __init__(self,is_simulation, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.data_channel = None
        self.is_simulation = is_simulation
        self.is_busy = False
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
        self.pixel_size_adjument_factor = 0.936
        # drift correction for image map
        self.drift_correction_x = -1.6
        self.drift_correction_y = -2.1
        # simulated sample data alias
        self.sample_data_alias = "agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38"
        self.get_pixel_size()

        # Initialize empty zarr canvas for stitching
        self._initialize_empty_canvas()

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
        self.pixel_size_xy = self.pixel_size_xy * self.pixel_size_adjument_factor
        print(f"Pixel size: {self.pixel_size_xy} µm")
        
                
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
        self.scanCoordinates.get_selected_wells_to_coordinates()
        
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
            while self.microcontroller.is_busy():
                time.sleep(0.005)
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

    async def snap_image(self, channel=0, intensity=100, exposure_time=100):
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
        gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)
        
        if not need_to_turn_illumination_back:
            self.liveController.turn_off_illumination()
            while self.microcontroller.is_busy():
                await asyncio.sleep(0.005)

        return gray_img
    
    async def get_camera_frame_simulation(self, channel=0, intensity=100, exposure_time=100):
        self.camera.set_exposure_time(exposure_time)
        self.liveController.set_illumination(channel, intensity)
        await self.send_trigger_simulation(channel, intensity, exposure_time)
        gray_img = self.camera.read_frame() 
        gray_img = rotate_and_flip_image(gray_img, self.camera.rotate_image_angle, self.camera.flip_image)
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
                                        do_reflection_af=False, action_ID='normal_scan_stitching'):
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
        """
        if illumination_settings is None:
            illumination_settings = [
                {'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100}
            ]
        
        # Initialize the Zarr canvas if not already done
        if not hasattr(self, 'zarr_canvas') or self.zarr_canvas is None:
            await self._initialize_zarr_canvas(illumination_settings)
        
        # Start the background stitching task
        await self.zarr_canvas.start_stitching()
        
        try:
            self.is_busy = True
            logging.info(f'Starting normal scan with stitching: {Nx}x{Ny} positions, dx={dx_mm}mm, dy={dy_mm}mm')
            
            # Map channel names to indices
            channel_map = {
                'BF LED matrix full': 0,
                'Fluorescence 405 nm Ex': 11,
                'Fluorescence 488 nm Ex': 12,
                'Fluorescence 561 nm Ex': 14,
                'Fluorescence 638 nm Ex': 13,
                'Fluorescence 730 nm Ex': 15
            }
            
            # Scan pattern: snake pattern for efficiency
            for i in range(Ny):
                for j in range(Nx):
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
                    
                    # Autofocus if requested (first position or periodically)
                    if do_reflection_af and (i == 0 and j == 0):
                        if hasattr(self, 'laserAutofocusController'):
                            await self.do_laser_autofocus()
                    elif do_contrast_autofocus and ((i * Nx + j) % CONFIG.Acquisition.NUMBER_OF_FOVS_PER_AF == 0):
                        await self.do_autofocus()
                    
                    # Acquire images for each channel
                    for idx, settings in enumerate(illumination_settings):
                        channel_name = settings['channel']
                        intensity = settings['intensity']
                        exposure_time = settings['exposure_time']
                        
                        # Get channel index
                        channel_idx = channel_map.get(channel_name, 0)
                        
                        # Snap image
                        image = await self.snap_image(channel_idx, intensity, exposure_time)
                        
                        # Convert to 8-bit if needed
                        if image.dtype != np.uint8:
                            # Scale to 8-bit
                            if image.dtype == np.uint16:
                                image = (image / 256).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)
                        
                        # Add to stitching queue
                        await self.zarr_canvas.add_image_async(
                            image, x_mm, y_mm, 
                            channel_idx=idx,  # Use the index in illumination_settings
                            z_idx=0
                        )
                        
                        logging.info(f'Added image at ({x_mm:.2f}, {y_mm:.2f}) for channel {channel_name}')
            
            logging.info('Normal scan with stitching completed')
            
            # Give the stitching queue a moment to process any final images
            logging.info('Allowing time for final images to be queued for stitching...')
            await asyncio.sleep(0.3)  # Wait 500ms to ensure all images are queued
            
            # Save a preview image from the lowest resolution scale
            try:
                preview_path = self.zarr_canvas.save_preview(action_ID)
                if preview_path:
                    logging.info(f'Canvas preview saved at: {preview_path}')
            except Exception as e:
                logging.error(f'Failed to save canvas preview: {e}')
            
        finally:
            self.is_busy = False
            # Stop the stitching task (this will now process all remaining images)
            await self.zarr_canvas.stop_stitching()
    
    async def _initialize_zarr_canvas(self, illumination_settings):
        """Initialize the Zarr canvas for stitching."""
        # Get channel names from illumination settings
        channels = [settings['channel'] for settings in illumination_settings]
        
        # Check if we already have a canvas initialized
        if hasattr(self, 'zarr_canvas') and self.zarr_canvas is not None:
            # Check if the existing canvas has the same channels
            if self.zarr_canvas.channels == channels:
                logging.info('Using existing Zarr canvas with matching channels')
                return
            else:
                # Close existing canvas and create new one with updated channels
                logging.info('Closing existing canvas and creating new one with updated channels')
                self.zarr_canvas.close()
        
        # Get the base path from environment variable
        zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
        
        # Define stage limits (from README)
        stage_limits = {
            'x_positive': 120,  # mm
            'x_negative': 0,    # mm
            'y_positive': 86,   # mm
            'y_negative': 0,    # mm
            'z_positive': 6     # mm
        }
        
        # Create the canvas
        self.zarr_canvas = ZarrCanvas(
            base_path=zarr_path,
            pixel_size_xy_um=self.pixel_size_xy,
            stage_limits=stage_limits,
            channels=channels
        )
        
        # Initialize the OME-Zarr structure
        self.zarr_canvas.initialize_canvas()
        logging.info(f'Initialized Zarr canvas at {zarr_path} with channels: {channels}')
    
    def get_stitched_region(self, center_x_mm, center_y_mm, width_mm, height_mm, 
                           scale_level=0, channel_name='BF LED matrix full'):
        """
        Get a region from the stitched canvas.
        
        Args:
            center_x_mm (float): Center X position in mm
            center_y_mm (float): Center Y position in mm
            width_mm (float): Width of region in mm
            height_mm (float): Height of region in mm
            scale_level (int): Scale level (0=full res, 1=1/4, 2=1/16, etc)
            channel_name (str): Name of channel to retrieve
            
        Returns:
            np.ndarray: The requested region
        """
        if not hasattr(self, 'zarr_canvas') or self.zarr_canvas is None:
            raise RuntimeError("Zarr canvas not initialized. Run a scan first.")
        
        # Get channel index
        try:
            channel_idx = self.zarr_canvas.channels.index(channel_name)
        except ValueError:
            logging.warning(f"Channel {channel_name} not found, using first channel")
            channel_idx = 0
        
        # Get the region
        region = self.zarr_canvas.get_canvas_region(
            center_x_mm, center_y_mm, width_mm, height_mm,
            scale=scale_level, channel_idx=channel_idx
        )
        
        return region

    def _initialize_empty_canvas(self):
        """Initialize an empty zarr canvas when the microscope starts up."""
        try:
            # Get the base path from environment variable
            zarr_path = os.getenv('ZARR_PATH', '/tmp/zarr_canvas')
            logging.info(f'ZARR_PATH environment variable: {os.getenv("ZARR_PATH", "not set (using default)")}')
            logging.info(f'Initializing empty zarr canvas at: {zarr_path}')
            
            # Check if the directory is writable
            zarr_dir = Path(zarr_path)
            
            # The actual zarr file will be at zarr_path/live_stitching.zarr
            actual_zarr_path = zarr_dir / "live_stitching.zarr"
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
            
            # Default channels for initialization
            default_channels = ['BF LED matrix full']
            
            # Create the canvas
            self.zarr_canvas = ZarrCanvas(
                base_path=zarr_path,
                pixel_size_xy_um=self.pixel_size_xy,
                stage_limits=stage_limits,
                channels=default_channels
            )
            
            # Initialize the OME-Zarr structure
            self.zarr_canvas.initialize_canvas()
            logging.info(f'Successfully initialized empty Zarr canvas at {actual_zarr_path} with pixel size {self.pixel_size_xy:.3f} µm')
            
        except Exception as e:
            logging.error(f'Failed to initialize empty zarr canvas: {e}')
            logging.info('Canvas will be initialized later when needed for scanning')
            logging.info('To fix this, either:')
            logging.info('  1. Set ZARR_PATH environment variable to a writable directory (e.g., export ZARR_PATH=/tmp/zarr_canvas)')
            logging.info('  2. Create the directory and set proper permissions')
            logging.info('  3. Run the application with appropriate permissions')
            self.zarr_canvas = None

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
