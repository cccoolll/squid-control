import os 
# app specific libraries
import squid_control.control.core_reef as core
import squid_control.control.microcontroller as microcontroller
from squid_control.control.config import *
from squid_control.control.camera import get_camera
import cv2
import logging
import squid_control.control.serial_peripherals as serial_peripherals
import matplotlib.path as mpath
if CONFIG.SUPPORT_LASER_AUTOFOCUS:
    import squid_control.control.core_displacement_measurement as core_displacement_measurement

import time
import asyncio
#using os to set current working directory
#find the current path
path=os.path.abspath(__file__)
#find the .ini file in the directory
path_ini=os.path.join(os.path.dirname(path),'configuration_HCS_v2.ini')
path_ini=Path(path_ini)



load_config(path_ini, False)
class SquidController:
    fps_software_trigger= 100

    def __init__(self,is_simulation, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.data_channel = None
        #load objects
        self.objectiveStore = core.ObjectiveStore()
        camera, camera_fc = get_camera(CONFIG.CAMERA_TYPE)
        self.is_simulation = is_simulation
        # load objects
        if self.is_simulation:
            if CONFIG.ENABLE_SPINNING_DISK_CONFOCAL:
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
            if CONFIG.ENABLE_SPINNING_DISK_CONFOCAL:
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
            filename="./uc2_fucci_illumination_configurations.xml"
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
        self.navigationController.move_x(25)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y(25)
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

        if CONFIG.SUPPORT_LASER_AUTOFOCUS:

            # controllers
            self.configurationManager_focus_camera = core.ConfigurationManager(filename='./squid_control/focus_camera_configurations.xml')
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
        self.current_expousre_time = 100
        self.current_intensity = 100
        self.pixel_size_xy = 0.1665
        self.get_pixel_size()


    def get_pixel_size(self):
        """Calculate pixel size based on imaging parameters."""
        try:
            tube_lens_mm = float(CONFIG.TUBE_LENS_MM)
            pixel_size_um = float(CONFIG.CAMERA_PIXEL_SIZE_UM[CONFIG.CAMERA_SENSOR])
            
            object_dict_key = self.objectiveStore.current_objective
            objective = self.objectiveStore.objectives_dict[object_dict_key]
            magnification =  float(objective['magnification'])
            objective_tube_lens_mm = float(objective['tube_lens_f_mm'])
            print(f"Tube lens: {tube_lens_mm} mm, Objective tube lens: {objective_tube_lens_mm} mm, Pixel size: {pixel_size_um} µm, Magnification: {magnification}")
        except:
            raise ValueError("Missing required parameters for pixel size calculation.")

        self.pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
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
    
    
    def plate_scan(self,well_plate_type='96', illuminate_channels=['BF LED matrix full'], do_contrast_autofocus=False,do_reflection_af=True, action_ID='testPlateScan'):
        self.move_to_scaning_position()
        self.scanCoordinates.get_selected_wells_to_coordinates()
        location_list = self.scanCoordinates.coordinates_mm
        self.multipointController.set_base_path(CONFIG.DEFAULT_SAVING_PATH)
        self.multipointController.set_selected_configurations(illuminate_channels)
        self.multipointController.do_autofocus = do_contrast_autofocus
        self.multipointController.do_reflection_af = do_reflection_af
        self.multipointController.set_NX(2)
        self.multipointController.set_NY(2)
        self.multipointController.start_new_experiment(action_ID)
        self.multipointController.run_acquisition(location_list=location_list)
        
    async def send_trigger_simulation(self, channel=0, intensity=100, exposure_time=100):
        print('Getting simulated image')
        current_x, current_y, current_z, *_ = self.navigationController.update_pos(microcontroller=self.microcontroller)
        self.dz = current_z - SIMULATED_CAMERA.ORIN_Z
        self.current_channel = channel
        magnification_factor = SIMULATED_CAMERA.MAGNIFICATION_FACTOR
        self.current_expousre_time = exposure_time
        self.current_intensity = intensity
        await self.camera.send_trigger(current_x, current_y, self.dz, self.pixel_size_xy, channel, intensity, exposure_time, magnification_factor)
        print(f'For simulated camera, exposure_time={exposure_time}, intensity={intensity}, magnification_factor={magnification_factor}, current position: {current_x},{current_y},{current_z}')
   
    def do_autofocus(self):
        
        if self.is_simulation:
            self.do_autofocus_simulation()
        else:
            self.autofocusController.set_deltaZ(1.524)
            self.autofocusController.set_N(15)
            self.autofocusController.autofocus()
            self.autofocusController.wait_till_autofocus_has_completed()

    def do_autofocus_simulation(self):
        
        random_z = 3.1+ np.random.normal(0,0.1)
        self.navigationController.move_z_to(random_z)
        self.send_trigger_simulation(self.current_channel, self.current_intensity, self.current_expousre_time)
        
    def init_laser_autofocus(self):
        self.laserAutofocusController.initialize_auto()
    
    def measure_displacement(self):
        self.laserAutofocusController.measure_displacement()     
        
    def platereader_move_to_well(self,row,column, wellplate_type='96'):
        if wellplate_type == '6':
            wellplate_format = WELLPLATE_FORMAT_6
        elif wellplate_type == '24':
            wellplate_format = WELLPLATE_FORMAT_24
        elif wellplate_type == '96':
            wellplate_format = WELLPLATE_FORMAT_96
        elif wellplate_type == '384':
            wellplate_format = WELLPLATE_FORMAT_384 
        
        if column != 0 and column != None:
            mm_per_ustep_X = CONFIG.SCREW_PITCH_X_MM/(self.navigationController.x_microstepping*CONFIG.FULLSTEPS_PER_REV_X)
            x_mm = wellplate_format.A1_X_MM + (int(column)-1)*wellplate_format.WELL_SPACING_MM
            x_usteps = CONFIG.STAGE_MOVEMENT_SIGN_X*round(x_mm/mm_per_ustep_X)
            self.microcontroller.move_x_to_usteps(x_usteps)
        if row != 0 and row != None:
            mm_per_ustep_Y = CONFIG.SCREW_PITCH_Y_MM/(self.navigationController.y_microstepping*CONFIG.FULLSTEPS_PER_REV_Y)
            y_mm = wellplate_format.A1_Y_MM + (ord(row) - ord('A'))*wellplate_format.WELL_SPACING_MM
            y_usteps = CONFIG.STAGE_MOVEMENT_SIGN_Y*round(y_mm/mm_per_ustep_Y)
            self.microcontroller.move_y_to_usteps(y_usteps)
 
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

        # move to scanning position
        self.navigationController.move_x(30)
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

    async def snap_image(self, channel=0, intensity=100, exposure_time=100):
        self.camera.set_exposure_time(exposure_time)
        self.liveController.set_illumination(channel, intensity)
        self.liveController.turn_on_illumination()
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.05)

        if self.is_simulation:
            await self.send_trigger_simulation(channel, intensity, exposure_time)
        else:
            self.camera.send_trigger()
        await asyncio.sleep(0.05)

        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)
        gray_img = self.camera.read_frame()
        self.liveController.turn_off_illumination()
        while self.microcontroller.is_busy():
            await asyncio.sleep(0.005)

        return gray_img
    
    def zoom_scan(self, rectangle, overlap=0.1, velocity_mm_s=9.0):
            """
            Perform a 'zoom scan' over the specified rectangular area.

            Args:
                rectangle: tuple (x_min, y_min, x_max, y_max) in millimeters.
                overlap:   float; fraction of image overlap in [0..1]. E.g. 0.1 = 10% overlap
                velocity_mm_s: stage velocity in mm/s.

            Returns:
                stitched_image (np.ndarray) – final panoramic image stitched from all frames.
            """
            # 1. Unpack the rectangle
            (x_min, y_min, x_max, y_max) = rectangle
            # Basic checks
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("Invalid rectangle coordinates for zoom scan.")

            # 2. Create a ZoomScanWorker (from the snippet in core.py).
            #    If you prefer QThread-based usage, see your ZoomScanController example.
            worker = core.ZoomScanWorker(
                camera=self.camera,
                microcontroller=self.microcontroller,
                navigationController=self.navigationController,
                liveController=self.liveController,
                rectangle=(x_min, y_min, x_max, y_max),
                overlap=overlap,
                velocity_mm_s=velocity_mm_s,
                store_images=False,  # or True if you want to save them
            )

            # 3. Run the worker in a blocking manner (simpler approach).
            #    The run() method does the stage moves, captures images,
            #    and calls _stitch_all() at the end.
            #    We'll override that at the end to actually retrieve the stitched image.

            # Hack: we’ll replace worker.run() with a version that returns the stitched image.
            # A quick approach is to store it in a local variable.

            # Original ZoomScanWorker in the snippet does:
            #   - continuous rows
            #   - stops camera
            #   - at the end does not always call stitch,
            #   so we add a small snippet to do it ourselves.
            worker.run()  # performs the actual scanning
            print(f'Captured {len(worker.captured_frames)} frames')
            # After run() is done, we can manually stitch what was captured if desired:
            if len(worker.captured_frames) > 1:
                stitched_image = worker._stitch_all()
            elif len(worker.captured_frames) == 1:
                # Only one frame, no real stitching needed
                stitched_image = worker.captured_frames[0]
            else:
                # No frames or user aborted
                stitched_image = np.array([])

            return stitched_image

    def close(self):

        # move the objective to a defined position upon exit
        self.navigationController.move_x(0.1) # temporary bug fix - move_x needs to be called before move_x_to if the stage has been moved by the joystick
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_x_to(28)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y(0.1) # temporary bug fix - move_y needs to be called before move_y_to if the stage has been moved by the joystick
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.navigationController.move_y_to(19)
        while self.microcontroller.is_busy():
            time.sleep(0.005)

        self.liveController.stop_live()
        self.camera.close()
        #self.imageSaver.close()
        self.imageDisplay.close()
        if CONFIG.SUPPORT_LASER_AUTOFOCUS:
            self.camera_focus.close()
            #self.imageDisplayWindow_focus.close()
        self.microcontroller.close()

def write_to_file(file_path, data):
    if isinstance(data, np.ndarray):
        np.save(file_path, data)
    else:
        with open(file_path, 'w') as file:
            file.write(data)
        
def try_zoom_scanner():
    squid_controller = SquidController(is_simulation=True)
    print('Squid controller initialized')
    squid_controller.move_x_to_limited(30)
    squid_controller.move_y_to_limited(30)
    zone_image = squid_controller.zoom_scan((30,30,35,35),0.1,9.0)
    write_to_file('zone_image.png',zone_image)
    print(f'Zone image shape: {zone_image.shape}')

if __name__ == "__main__":
    try_zoom_scanner()
