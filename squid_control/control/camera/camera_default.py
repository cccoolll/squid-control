import os
import glob
import time
import numpy as np
from PIL import Image
try:
    import squid_control.control.gxipy as gx
except:
    print("gxipy import error")

from squid_control.control.config import CONFIG
from squid_control.control.camera import TriggerModeSetting
from scipy.ndimage import gaussian_filter
import zarr

script_dir = os.path.dirname(__file__)

def get_sn_by_model(model_name):
    try:
        device_manager = gx.DeviceManager()
        device_num, device_info_list = device_manager.update_device_list()
    except:
        device_num = 0
    if device_num > 0:
        for i in range(device_num):
            if device_info_list[i]["model_name"] == model_name:
                return device_info_list[i]["sn"]
    return None  # return None if no device with the specified model_name is connected


class Camera(object):

    def __init__(
        self, sn=None, is_global_shutter=False, rotate_image_angle=None, flip_image=None
    ):

        # many to be purged
        self.sn = sn
        self.is_global_shutter = is_global_shutter
        self.device_manager = gx.DeviceManager()
        self.device_info_list = None
        self.device_index = 0
        self.camera = None
        self.is_color = None
        self.gamma_lut = None
        self.contrast_lut = None
        self.color_correction_param = None

        self.rotate_image_angle = rotate_image_angle
        self.flip_image = flip_image

        self.exposure_time = 1  # unit: ms
        self.analog_gain = 0
        self.frame_ID = -1
        self.frame_ID_software = -1
        self.frame_ID_offset_hardware_trigger = 0
        self.timestamp = 0

        self.image_locked = False
        self.current_frame = None

        self.callback_is_enabled = False
        self.is_streaming = False

        self.GAIN_MAX = 24
        self.GAIN_MIN = 0
        self.GAIN_STEP = 1
        self.EXPOSURE_TIME_MS_MIN = 0.01
        self.EXPOSURE_TIME_MS_MAX = 4000

        self.trigger_mode = None
        self.pixel_size_byte = 1

        # below are values for IMX226 (MER2-1220-32U3M) - to make configurable
        self.row_period_us = 10
        self.row_numbers = 3036
        self.exposure_delay_us_8bit = 650
        self.exposure_delay_us = self.exposure_delay_us_8bit * self.pixel_size_byte
        self.strobe_delay_us = (
            self.exposure_delay_us
            + self.row_period_us * self.pixel_size_byte * (self.row_numbers - 1)
        )

        self.pixel_format = "MONO8"  # use the default pixel format

        self.is_live = False  # this determines whether a new frame received will be handled in the streamHandler
        # mainly for discarding the last frame received after stop_live() is called, where illumination is being turned off during exposure

    def open(self, index=0):
        (device_num, self.device_info_list) = self.device_manager.update_device_list()
        if device_num == 0:
            raise RuntimeError("Could not find any USB camera devices!")
        if self.sn is None:
            self.device_index = index
            self.camera = self.device_manager.open_device_by_index(index + 1)
        else:
            self.camera = self.device_manager.open_device_by_sn(self.sn)
        self.is_color = self.camera.PixelColorFilter.is_implemented()
        # self._update_image_improvement_params()
        # self.camera.register_capture_callback(self,self._on_frame_callback)
        if self.is_color:
            # self.set_wb_ratios(self.get_awb_ratios())
            print(self.get_awb_ratios())
            # self.set_wb_ratios(1.28125,1.0,2.9453125)
            self.set_wb_ratios(2, 1, 2)

        # temporary
        self.camera.AcquisitionFrameRate.set(1000)
        self.camera.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)

        # turn off device link throughput limit
        self.camera.DeviceLinkThroughputLimitMode.set(gx.GxSwitchEntry.OFF)

        # get sensor parameters
        self.Width = self.camera.Width.get()
        self.Height = self.camera.Height.get()
        self.WidthMax = self.camera.WidthMax.get()
        self.HeightMax = self.camera.HeightMax.get()
        self.OffsetX = self.camera.OffsetX.get()
        self.OffsetY = self.camera.OffsetY.get()

    def set_callback(self, function):
        self.new_image_callback_external = function

    def enable_callback(self):
        if self.callback_is_enabled == False:
            # stop streaming
            if self.is_streaming:
                was_streaming = True
                self.stop_streaming()
            else:
                was_streaming = False
            # enable callback
            user_param = None
            self.camera.register_capture_callback(user_param, self._on_frame_callback)
            self.callback_is_enabled = True
            # resume streaming if it was on
            if was_streaming:
                self.start_streaming()
            self.callback_is_enabled = True
        else:
            pass

    def disable_callback(self):
        if self.callback_is_enabled == True:
            # stop streaming
            if self.is_streaming:
                was_streaming = True
                self.stop_streaming()
            else:
                was_streaming = False
            # disable call back
            self.camera.unregister_capture_callback()
            self.callback_is_enabled = False
            # resume streaming if it was on
            if was_streaming:
                self.start_streaming()
        else:
            pass

    def open_by_sn(self, sn):
        (device_num, self.device_info_list) = self.device_manager.update_device_list()
        if device_num == 0:
            raise RuntimeError("Could not find any USB camera devices!")
        self.camera = self.device_manager.open_device_by_sn(sn)
        self.is_color = self.camera.PixelColorFilter.is_implemented()
        self._update_image_improvement_params()

        """
        if self.is_color is True:
            self.camera.register_capture_callback(_on_color_frame_callback)
        else:
            self.camera.register_capture_callback(_on_frame_callback)
        """

    def close(self):
        self.camera.close_device()
        self.device_info_list = None
        self.camera = None
        self.is_color = None
        self.gamma_lut = None
        self.contrast_lut = None
        self.color_correction_param = None
        self.last_raw_image = None
        self.last_converted_image = None
        self.last_numpy_image = None

    def set_exposure_time(self, exposure_time):
        use_strobe = (
            self.trigger_mode == TriggerModeSetting.HARDWARE
        )  # true if using hardware trigger
        if use_strobe == False or self.is_global_shutter:
            self.exposure_time = exposure_time
            self.camera.ExposureTime.set(exposure_time * 1000)
        else:
            # set the camera exposure time such that the active exposure time (illumination on time) is the desired value
            self.exposure_time = exposure_time
            # add an additional 500 us so that the illumination can fully turn off before rows start to end exposure
            camera_exposure_time = (
                self.exposure_delay_us
                + self.exposure_time * 1000
                + self.row_period_us * self.pixel_size_byte * (self.row_numbers - 1)
                + 500
            )  # add an additional 500 us so that the illumination can fully turn off before rows start to end exposure
            self.camera.ExposureTime.set(camera_exposure_time)

    def update_camera_exposure_time(self):
        use_strobe = (
            self.trigger_mode == TriggerModeSetting.HARDWARE
        )  # true if using hardware trigger
        if use_strobe == False or self.is_global_shutter:
            self.camera.ExposureTime.set(self.exposure_time * 1000)
        else:
            camera_exposure_time = (
                self.exposure_delay_us
                + self.exposure_time * 1000
                + self.row_period_us * self.pixel_size_byte * (self.row_numbers - 1)
                + 500
            )  # add an additional 500 us so that the illumination can fully turn off before rows start to end exposure
            self.camera.ExposureTime.set(camera_exposure_time)

    def set_analog_gain(self, analog_gain):
        self.analog_gain = analog_gain
        self.camera.Gain.set(analog_gain)

    def get_awb_ratios(self):
        self.camera.BalanceWhiteAuto.set(2)
        self.camera.BalanceRatioSelector.set(0)
        awb_r = self.camera.BalanceRatio.get()
        self.camera.BalanceRatioSelector.set(1)
        awb_g = self.camera.BalanceRatio.get()
        self.camera.BalanceRatioSelector.set(2)
        awb_b = self.camera.BalanceRatio.get()
        return (awb_r, awb_g, awb_b)

    def set_wb_ratios(self, wb_r=None, wb_g=None, wb_b=None):
        self.camera.BalanceWhiteAuto.set(0)
        if wb_r is not None:
            self.camera.BalanceRatioSelector.set(0)
            awb_r = self.camera.BalanceRatio.set(wb_r)
        if wb_g is not None:
            self.camera.BalanceRatioSelector.set(1)
            awb_g = self.camera.BalanceRatio.set(wb_g)
        if wb_b is not None:
            self.camera.BalanceRatioSelector.set(2)
            awb_b = self.camera.BalanceRatio.set(wb_b)

    def set_reverse_x(self, value):
        self.camera.ReverseX.set(value)

    def set_reverse_y(self, value):
        self.camera.ReverseY.set(value)

    def start_streaming(self):
        self.camera.stream_on()
        self.is_streaming = True

    def stop_streaming(self):
        self.camera.stream_off()
        self.is_streaming = False

    def set_pixel_format(self, pixel_format):
        if self.is_streaming == True:
            was_streaming = True
            self.stop_streaming()
        else:
            was_streaming = False

        if (
            self.camera.PixelFormat.is_implemented()
            and self.camera.PixelFormat.is_writable()
        ):
            if pixel_format == "MONO8":
                self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
                self.pixel_size_byte = 1
            if pixel_format == "MONO10":
                self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO10)
                self.pixel_size_byte = 1
            if pixel_format == "MONO12":
                self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO12)
                self.pixel_size_byte = 2
            if pixel_format == "MONO14":
                self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO14)
                self.pixel_size_byte = 2
            if pixel_format == "MONO16":
                self.camera.PixelFormat.set(gx.GxPixelFormatEntry.MONO16)
                self.pixel_size_byte = 2
            if pixel_format == "BAYER_RG8":
                self.camera.PixelFormat.set(gx.GxPixelFormatEntry.BAYER_RG8)
                self.pixel_size_byte = 1
            if pixel_format == "BAYER_RG12":
                self.camera.PixelFormat.set(gx.GxPixelFormatEntry.BAYER_RG12)
                self.pixel_size_byte = 2
            self.pixel_format = pixel_format
        else:
            print("pixel format is not implemented or not writable")

        if was_streaming:
            self.start_streaming()

        # update the exposure delay and strobe delay
        self.exposure_delay_us = self.exposure_delay_us_8bit * self.pixel_size_byte
        self.strobe_delay_us = (
            self.exposure_delay_us
            + self.row_period_us * self.pixel_size_byte * (self.row_numbers - 1)
        )

    def set_continuous_acquisition(self):
        self.camera.TriggerMode.set(gx.GxSwitchEntry.OFF)
        self.trigger_mode = TriggerModeSetting.CONTINUOUS
        self.update_camera_exposure_time()

    def set_software_triggered_acquisition(self):
        self.camera.TriggerMode.set(gx.GxSwitchEntry.ON)
        self.camera.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        self.trigger_mode = TriggerModeSetting.SOFTWARE
        self.update_camera_exposure_time()

    def set_hardware_triggered_acquisition(self):
        self.camera.TriggerMode.set(gx.GxSwitchEntry.ON)
        self.camera.TriggerSource.set(
            gx.GxTriggerSourceEntry.LINE2
        )  # LINE0 requires 7 mA min
        # self.camera.TriggerSource.set(gx.GxTriggerActivationEntry.RISING_EDGE)
        self.frame_ID_offset_hardware_trigger = None
        self.trigger_mode = TriggerModeSetting.HARDWARE
        self.update_camera_exposure_time()

    def send_trigger(self):
        if self.is_streaming:
            self.camera.TriggerSoftware.send_command()
        else:
            print("trigger not sent - camera is not streaming")

    def read_frame(self):
        raw_image = self.camera.data_stream[self.device_index].get_image()
        if self.is_color:
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()
            if self.pixel_format == "BAYER_RG12":
                numpy_image = numpy_image << 4
        else:
            numpy_image = raw_image.get_numpy_array()
            if self.pixel_format == "MONO12":
                numpy_image = numpy_image << 4
        # self.current_frame = numpy_image
        return numpy_image

    def _on_frame_callback(self, user_param, raw_image):
        if raw_image is None:
            print("Getting image failed.")
            return
        if raw_image.get_status() != 0:
            print("Got an incomplete frame")
            return
        if self.image_locked:
            print("last image is still being processed, a frame is dropped")
            return
        if self.is_color:
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()
            if self.pixel_format == "BAYER_RG12":
                numpy_image = numpy_image << 4
        else:
            numpy_image = raw_image.get_numpy_array()
            if self.pixel_format == "MONO12":
                numpy_image = numpy_image << 4
        if numpy_image is None:
            return
        self.current_frame = numpy_image
        self.frame_ID_software = self.frame_ID_software + 1
        self.frame_ID = raw_image.get_frame_id()
        if self.trigger_mode == TriggerModeSetting.HARDWARE:
            if self.frame_ID_offset_hardware_trigger == None:
                self.frame_ID_offset_hardware_trigger = self.frame_ID
            self.frame_ID = self.frame_ID - self.frame_ID_offset_hardware_trigger
        self.timestamp = time.time()
        self.new_image_callback_external(self)

        # self.frameID = self.frameID + 1
        # print(self.frameID)

    def set_ROI(self, offset_x=None, offset_y=None, width=None, height=None):

        # stop streaming if streaming is on
        if self.is_streaming == True:
            was_streaming = True
            self.stop_streaming()
        else:
            was_streaming = False

        if width is not None:
            self.Width = width
            # update the camera setting
            if self.camera.Width.is_implemented() and self.camera.Width.is_writable():
                self.camera.Width.set(self.Width)
            else:
                print("Width is not implemented or not writable")

        if height is not None:
            self.Height = height
            # update the camera setting
            if self.camera.Height.is_implemented() and self.camera.Height.is_writable():
                self.camera.Height.set(self.Height)
            else:
                print("Height is not implemented or not writable")

        if offset_x is not None:
            self.OffsetX = offset_x
            # update the camera setting
            if (
                self.camera.OffsetX.is_implemented()
                and self.camera.OffsetX.is_writable()
            ):
                self.camera.OffsetX.set(self.OffsetX)
            else:
                print("OffsetX is not implemented or not writable")

        if offset_y is not None:
            self.OffsetY = offset_y
            # update the camera setting
            if (
                self.camera.OffsetY.is_implemented()
                and self.camera.OffsetY.is_writable()
            ):
                self.camera.OffsetY.set(self.OffsetY)
            else:
                print("OffsetY is not implemented or not writable")

        # restart streaming if it was previously on
        if was_streaming == True:
            self.start_streaming()

    def reset_camera_acquisition_counter(self):
        if (
            self.camera.CounterEventSource.is_implemented()
            and self.camera.CounterEventSource.is_writable()
        ):
            self.camera.CounterEventSource.set(gx.GxCounterEventSourceEntry.LINE2)
        else:
            print("CounterEventSource is not implemented or not writable")

        if self.camera.CounterReset.is_implemented():
            self.camera.CounterReset.send_command()
        else:
            print("CounterReset is not implemented")

    def set_line3_to_strobe(self):
        # self.camera.StrobeSwitch.set(gx.GxSwitchEntry.ON)
        self.camera.LineSelector.set(gx.GxLineSelectorEntry.LINE3)
        self.camera.LineMode.set(gx.GxLineModeEntry.OUTPUT)
        self.camera.LineSource.set(gx.GxLineSourceEntry.STROBE)

    def set_line3_to_exposure_active(self):
        # self.camera.StrobeSwitch.set(gx.GxSwitchEntry.ON)
        self.camera.LineSelector.set(gx.GxLineSelectorEntry.LINE3)
        self.camera.LineMode.set(gx.GxLineModeEntry.OUTPUT)
        self.camera.LineSource.set(gx.GxLineSourceEntry.EXPOSURE_ACTIVE)


class Camera_Simulation(object):



    def __init__(
        self, sn=None, is_global_shutter=False, rotate_image_angle=None, flip_image=None
    ):
        # many to be purged
        self.sn = sn
        self.is_global_shutter = is_global_shutter
        self.device_info_list = None
        self.device_index = 0
        self.camera = None
        self.is_color = None
        self.gamma_lut = None
        self.contrast_lut = None
        self.color_correction_param = None

        self.rotate_image_angle = rotate_image_angle
        self.flip_image = flip_image

        self.exposure_time = 0
        self.analog_gain = 0
        self.frame_ID = 0
        self.frame_ID_software = -1
        self.frame_ID_offset_hardware_trigger = 0
        self.timestamp = 0

        self.image_locked = False
        self.current_frame = None

        self.callback_is_enabled = False
        self.is_streaming = False

        self.GAIN_MAX = 24
        self.GAIN_MIN = 0
        self.GAIN_STEP = 1
        self.EXPOSURE_TIME_MS_MIN = 0.01
        self.EXPOSURE_TIME_MS_MAX = 4000

        self.trigger_mode = None
        self.pixel_size_byte = 1

        # below are values for IMX226 (MER2-1220-32U3M) - to make configurable
        self.row_period_us = 10
        self.row_numbers = 3036
        self.exposure_delay_us_8bit = 650
        self.exposure_delay_us = self.exposure_delay_us_8bit * self.pixel_size_byte
        self.strobe_delay_us = (
            self.exposure_delay_us
            + self.row_period_us * self.pixel_size_byte * (self.row_numbers - 1)
        )

        self.pixel_format = "MONO8"

        self.is_live = False

        self.Width = 3000
        self.Height = 3000
        self.WidthMax = 4000
        self.HeightMax = 3000
        self.OffsetX = 0
        self.OffsetY = 0
        
        # simulated camera values
        self.simulated_focus = 3.3
        self.channels = [0, 11, 12, 14, 13]
        self.image_paths = {
            0: 'LED.bmp',
            11: '405nm.png',
            12: '488nm.png',
            14: '561nm.png',
            13: '638nm.png',
        }
        self.zarr_path = os.getenv('ZARR_PATH')
        
        self.image_folder = os.getenv('IMAGE_DATA_FOR_SIMULATED_MICROSCOPE')
        if not self.image_folder:
            raise EnvironmentError("Please set IMAGE_DATA_FOR_SIMULATED_MICROSCOPE to the folder path.")

        # Initialize fields of view and channels
        self.fields_of_view = {}
        self.current_fov = None

        # Load available fields of view and channels
        for filepath in glob.glob(os.path.join(self.image_folder, '*.bmp')):
            filename = os.path.basename(filepath)
            parts = filename.split('_')
            if len(parts) >= 5:
                fov_id = '_'.join(parts[:4])  # Capture the full FOV ID, e.g., "A3_1_0_0"
                channel_name = '_'.join(parts[4:])  # Capture the full channel name, e.g., "BF_LED_matrix_full"
                
                if fov_id not in self.fields_of_view:
                    self.fields_of_view[fov_id] = {}
                self.fields_of_view[fov_id][channel_name] = filepath


        # Choose a random initial FOV
        self.current_fov = np.random.choice(list(self.fields_of_view.keys()))
        print("Number of fields of view", len(self.fields_of_view))
        print("current_fov", self.current_fov)
        
    def open(self, index=0):
        pass

    def set_callback(self, function):
        self.new_image_callback_external = function

# TODO: Implement the following methods for the simulated camera
    def register_capture_callback_simulated(self, user_param, callback):
        """
        Register a callback function to be called with simulated camera data.

        :param user_param: User parameter to pass to the callback
        :param callback: Callback function to be called with the simulated data
        """
        self.user_param = user_param
        self.capture_callback = callback

    def simulate_capture_event(self):
        """
        Simulate a camera capture event and call the registered callback.
        """
        if self.capture_callback:
            simulated_data = self.generate_simulated_data()
            self.capture_callback(self.user_param, simulated_data)

    def generate_simulated_data(self):
        """
        Generate simulated camera data.

        :return: Simulated data
        """
        # Replace this with actual simulated data generation logic
        return np.random.randint(0, 256, (self.Height, self.Width), dtype=np.uint8)
        
    def enable_callback(self):
        if self.callback_is_enabled == False:
            # stop streaming
            if self.is_streaming:
                was_streaming = True
                self.stop_streaming()
            else:
                was_streaming = False
            # enable callback
            user_param = None
            self.register_capture_callback_simulated(user_param, self._on_frame_callback)
            self.callback_is_enabled = True
            # resume streaming if it was on
            if was_streaming:
                self.start_streaming()
            self.callback_is_enabled = True
        else:
            pass

    def disable_callback(self):
        self.callback_is_enabled = False

    def open_by_sn(self, sn):
        pass

    def close(self):
        pass

    def set_exposure_time(self, exposure_time):
        pass

    def update_camera_exposure_time(self):
        pass

    def set_analog_gain(self, analog_gain):
        pass

    def get_awb_ratios(self):
        pass

    def set_wb_ratios(self, wb_r=None, wb_g=None, wb_b=None):
        pass

    def start_streaming(self):
        self.frame_ID_software = 0

    def stop_streaming(self):
        pass

    def set_pixel_format(self, pixel_format):
        self.pixel_format = pixel_format
        print(pixel_format)
        self.frame_ID = 0

    def set_continuous_acquisition(self):
        pass

    def set_software_triggered_acquisition(self):
        pass

    def set_hardware_triggered_acquisition(self):
        pass

    def send_trigger(self, x=29.81, y= 36.85, dz=0, pixel_size_um=0.1665, channel=0, intensity=100, exposure_time=100, magnification_factor=20):
        self.frame_ID += 1
        self.timestamp = time.time()
        channel_map = {
            0: 'BF_LED_matrix_full',
            11: 'Fluorescence_405_nm_Ex',
            12: 'Fluorescence_488_nm_Ex',
            14: 'Fluorescence_561_nm_Ex',
            13: 'Fluorescence_638_nm_Ex'
        }
        channel_name = channel_map.get(channel, None)  # Get the channel name or None if not found

        if channel_name is None:
            # If the channel is not found, return a random image
            self.image = np.random.randint(0, 256, (self.Height, self.Width), dtype=np.uint8)
            print(f"Channel {channel} not found, returning a random image")
        else:
            # Load the OME-Zarr file
            root = zarr.open(self.zarr_path, mode='r')

            # Access the specified channel and scale0
            if channel_name not in root:
                # If the channel is not found in the Zarr file, return a random image
                self.image = np.random.randint(0, 256, (self.Height, self.Width), dtype=np.uint8)
                print(f"Channel {channel_name} not found in Zarr file, returning a random image")
            else:
                dataset = root[channel_name]['scale0']  # Access scale0

                # Calculate the pixel coordinates in the scale0 dataset
                pixel_x = int(x / pixel_size_um) * 1000
                pixel_y = int(y / pixel_size_um) * 1000

                # Extract the region of interest (ROI) based on self.Width and self.Height
                start_x = max(0, pixel_x - self.Width // 2)
                start_y = max(0, pixel_y - self.Height // 2)
                end_x = start_x + self.Width
                end_y = start_y + self.Height

                # Ensure the ROI is within the bounds of the dataset
                start_x = min(start_x, dataset.shape[1] - self.Width)
                start_y = min(start_y, dataset.shape[0] - self.Height)
                end_x = start_x + self.Width
                end_y = start_y + self.Height

                # Extract the image data
                self.image = dataset[start_y:end_y, start_x:end_x]
                self.image = self.image.astype(np.uint8)
                print(f"Extracted image data from {channel_name} at {x},{y}, ({start_x}, {start_y}) to ({end_x}, {end_y})")
                
            
            # TODO: callback to the _on_frame_callback use 'generate_simulated_data'
            # self.simulate_capture_event()
            
            


                

        # Simulate intensity and exposure
        exposure_factor = exposure_time / 100
        intensity_factor = intensity / 60
        self.image = np.clip(self.image * exposure_factor * intensity_factor, 0, 255).astype(np.uint8)
        
        # Process the image based on pixel format
        if self.pixel_format == "MONO8":
            self.current_frame = self.image
        elif self.pixel_format == "MONO12":
            self.current_frame = (self.image.astype(np.uint16) * 16).astype(np.uint16)
        elif self.pixel_format == "MONO16":
            self.current_frame = (self.image.astype(np.uint16) * 256).astype(np.uint16)

        # Apply focus effect if `dz` is not zero
        if dz != 0:
            sigma = abs(dz) * 6  # Adjust for blur intensity
            self.current_frame = gaussian_filter(self.current_frame, sigma=sigma)
            print(f"The image is blurred with dz={dz}, sigma={sigma}")
        
        if self.new_image_callback_external is not None and self.callback_is_enabled:
            self.new_image_callback_external(self)
                    
    def read_frame(self):
        return self.current_frame

    def _on_frame_callback(self, user_param, raw_image):
        if raw_image is None:
            raw_image = np.random.randint(0, 256, (self.Height, self.Width), dtype=np.uint8)
        if self.image_locked:
            print("last image is still being processed, a frame is dropped")
            return
        if self.is_color:
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()
            if self.pixel_format == "BAYER_RG12":
                numpy_image = numpy_image << 4
        else:
            numpy_image = raw_image.get_numpy_array()
            if self.pixel_format == "MONO12":
                numpy_image = numpy_image << 4
        if numpy_image is None:
            return
        self.current_frame = numpy_image
        self.frame_ID_software = self.frame_ID_software + 1
        self.frame_ID = raw_image.get_frame_id()
        if self.trigger_mode == TriggerModeSetting.HARDWARE:
            if self.frame_ID_offset_hardware_trigger == None:
                self.frame_ID_offset_hardware_trigger = self.frame_ID
            self.frame_ID = self.frame_ID - self.frame_ID_offset_hardware_trigger
        self.timestamp = time.time()
        self.new_image_callback_external(self)  

    def set_ROI(self, offset_x=None, offset_y=None, width=None, height=None):
        pass

    def reset_camera_acquisition_counter(self):
        pass

    def set_line3_to_strobe(self):
        pass

    def set_line3_to_exposure_active(self):
        pass
