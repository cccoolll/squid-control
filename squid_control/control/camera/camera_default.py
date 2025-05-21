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
from hypha_tools.artifact_manager.artifact_manager import SquidArtifactManager, ZarrImageManager
import asyncio
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
        pass


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
        self.image = None
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
        self.simulated_focus = 4
        self.channels = [0, 11, 12, 14, 13]
        self.image_paths = {
            0: 'BF_LED_matrix_full.bmp',
            11: 'Fluorescence_405_nm_Ex.bmp',
            12: 'Fluorescence_488_nm_Ex.bmp',
            14: 'Fluorescence_561_nm_Ex.bmp',
            13: 'Fluorescence_638_nm_Ex.bmp',
        }
        # Configuration for ZarrImageManager
        self.SERVER_URL = "https://hypha.aicell.io"
        self.WORKSPACE_TOKEN = os.getenv("SQUID_WORKSPACE_TOKEN")
        self.DEFAULT_TIMESTAMP = "2025-04-29_16-38-27"  # Default timestamp for the dataset
        
        # Initialize these to None, will be set up lazily when needed
        self.zarr_image_manager = None
        self.artifact_manager = None

        # Use scale1 instead of scale0 for lower resolution
        self.scale_level = 2
        self.scale_factor = 16  # scale1 is 1/4 of scale0

    def open(self, index=0):
        pass

    def set_callback(self, function):
        self.new_image_callback_external = function

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
        self.cleanup_zarr_resources()
        pass
        
    def cleanup_zarr_resources(self):
        """
        Synchronous wrapper for async cleanup method
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop if the current one is already running
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(self._cleanup_zarr_resources_async())
                new_loop.close()
            else:
                loop.run_until_complete(self._cleanup_zarr_resources_async())
        except Exception as e:
            print(f"Error in cleanup_zarr_resources: {e}")
        
    async def _cleanup_zarr_resources_async(self):
        """
        Clean up Zarr-related resources to prevent resource leaks
        """
        try:
            if self.zarr_image_manager:
                print("Closing ZarrImageManager resources...")
                # Clear the cache to free memory
                if hasattr(self.zarr_image_manager, 'zarr_groups_cache'):
                    self.zarr_image_manager.zarr_groups_cache.clear()
                    self.zarr_image_manager.zarr_groups_timestamps.clear()
                    
                await self.zarr_image_manager.close()
                self.zarr_image_manager = None
                print("ZarrImageManager closed successfully")
                
            if self.artifact_manager:
                print("Closing ArtifactManager resources...")
                # Clear the cache to free memory
                if hasattr(self.artifact_manager, 'zarr_groups_cache'):
                    self.artifact_manager.zarr_groups_cache.clear()
                    self.artifact_manager.zarr_groups_timestamps.clear()
                
                # Close the artifact manager if it has a close method
                if hasattr(self.artifact_manager, 'close'):
                    await self.artifact_manager.close()
                self.artifact_manager = None
                print("ArtifactManager closed successfully")
        except Exception as e:
            print(f"Error closing Zarr resources: {e}")
            import traceback
            print(traceback.format_exc())
            
    async def cleanup_zarr_resources_async(self):
        """
        Legacy method for backward compatibility
        """
        await self._cleanup_zarr_resources_async()

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

    async def get_image_from_zarr(self, x, y, pixel_size_um, channel_name, sample_data_alias="squid-control/image-map-20250429-treatment-zip"):
        """
        Get image data from Zarr storage for the specified coordinates and channel.
        
        Args:
            x (float): X coordinate in mm
            y (float): Y coordinate in mm
            pixel_size_um (float): Pixel size in micrometers
            channel_name (str): Name of the channel to retrieve
            sample_data_alias (str): Alias of the sample data
            
        Returns:
            np.ndarray: The image data
        """
        # Lazily initialize ZarrImageManager if needed
        if self.zarr_image_manager is None:
            print("Creating new ZarrImageManager instance...")
            self.zarr_image_manager = ZarrImageManager()
            print("Connecting to ZarrImageManager...")
            await self.zarr_image_manager.connect(workspace_token=self.WORKSPACE_TOKEN, server_url=self.SERVER_URL)
            print("Connected to ZarrImageManager")
            
            # Set the scale_key to scale1 instead of scale0
            self.zarr_image_manager.scale_key = f'scale{self.scale_level}'
        
        # Convert microscope coordinates (mm) to pixel coordinates - fix conversion factor
        # Divide by scale_factor since we're using scale1 (1/4 resolution)
        pixel_x = int((x / pixel_size_um) * 1000 / self.scale_factor)
        pixel_y = int((y / pixel_size_um) * 1000 / self.scale_factor)
        
        # Print pixel coordinates for debugging
        print(f"Converted coords (mm) x={x}, y={y} to pixel coords: x={pixel_x}, y={pixel_y} (scale{self.scale_level})")
        
        # Use the class variables for dataset configuration
        dataset_id = sample_data_alias
        if dataset_id == "squid-control/image-map-20250429-treatment-zip":
            timestamp = self.DEFAULT_TIMESTAMP
        elif dataset_id == "squid-control/image-map-20250506-treatment-zip":
            timestamp = "2025-05-06_16-56-52"
        
        print(f"Using dataset: {dataset_id}, timestamp: {timestamp}, channel: {channel_name}")
        
        try:
            # Calculate region boundaries with reduced dimensions (Width/4, Height/4)
            scaled_width = self.Width // self.scale_factor
            scaled_height = self.Height // self.scale_factor
            
            half_width = scaled_width // 2
            half_height = scaled_height // 2
            
            y_start = max(0, pixel_y - half_height)
            y_end = y_start + scaled_height
            x_start = max(0, pixel_x - half_width)
            x_end = x_start + scaled_width
            
            # Get the region directly using direct_region parameter and passing scaled Width and Height
            region_data = await self.zarr_image_manager.get_region_np_data(
                dataset_id, 
                timestamp, 
                channel_name, 
                self.scale_level,  # Using scale level from class property
                0,  # x coordinate (ignored when using direct_region)
                0,  # y coordinate (ignored when using direct_region)
                direct_region=(y_start, y_end, x_start, x_end),
                width=scaled_width,
                height=scaled_height
            )
            
            return region_data
        except Exception as e:
            print(f"Error getting image from Zarr: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    async def send_trigger(self, x=29.81, y=36.85, dz=0, pixel_size_um=0.333, channel=0, intensity=100, exposure_time=100, magnification_factor=20, performace_mode=False, sample_data_alias="squid-control/image-map-20250429-treatment-zip"):
        print(f"Sending trigger with x={x}, y={y}, dz={dz}, pixel_size_um={pixel_size_um}, channel={channel}, intensity={intensity}, exposure_time={exposure_time}, magnification_factor={magnification_factor}, performace_mode={performace_mode}")
        self.frame_ID += 1
        self.timestamp = time.time()

        channel_map = {
            0: 'BF_LED_matrix_full',
            11: 'Fluorescence_405_nm_Ex',
            12: 'Fluorescence_488_nm_Ex',
            14: 'Fluorescence_561_nm_Ex',
            13: 'Fluorescence_638_nm_Ex'
        }
        channel_name = channel_map.get(channel, None)

        if channel_name is None:
            self.image = np.array(Image.open(os.path.join(script_dir, f"example-data/{self.image_paths[channel]}")))
            print(f"Channel {channel} not found, returning a random image")
        
        elif performace_mode:
            self.image = np.array(Image.open(os.path.join(script_dir, f"example-data/{self.image_paths[channel]}")))
            print(f"Using performance mode, example image for channel {channel}")
        else:
            self.image = await self.get_image_from_zarr(x, y, pixel_size_um, channel_name, sample_data_alias)
            if self.image is None:
                # Fallback to example image if Zarr access fails
                self.image = np.array(Image.open(os.path.join(script_dir, f"example-data/{self.image_paths[channel]}")))
                print(f"Failed to get image from Zarr, using example image for channel {channel}")

        # Apply exposure and intensity scaling
        exposure_factor = max(0.1, exposure_time / 100)  # Ensure minimum factor to prevent black images
        intensity_factor = max(0.1, intensity / 60)      # Ensure minimum factor to prevent black images
        
        # Check if image contains any valid data before scaling
        if np.count_nonzero(self.image) == 0:
            print("WARNING: Image contains all zeros before scaling!")
            self.image = np.ones((self.Height, self.Width), dtype=np.uint8) * 128
        # Convert to float32 for scaling, apply factors, then clip and convert back to uint8
        self.image = np.clip(self.image.astype(np.float32) * exposure_factor * intensity_factor, 0, 255).astype(np.uint8)
        
        # Check if image contains any valid data after scaling
        if np.count_nonzero(self.image) == 0:
            print("WARNING: Image contains all zeros after scaling!")
            # Set to a gray image instead of black
            self.image = np.ones((self.Height, self.Width), dtype=np.uint8) * 128

        if self.pixel_format == "MONO8":
            self.current_frame = self.image
        elif self.pixel_format == "MONO12":
            self.current_frame = (self.image.astype(np.uint16) * 16).astype(np.uint16)
        elif self.pixel_format == "MONO16":
            self.current_frame = (self.image.astype(np.uint16) * 256).astype(np.uint16)
        else:
            # For any other format, default to MONO8
            print(f"Unrecognized pixel format {self.pixel_format}, using MONO8")
            self.current_frame = self.image

        if dz != 0:
            sigma = abs(dz) * 6
            self.current_frame = gaussian_filter(self.current_frame, sigma=sigma)
            print(f"The image is blurred with dz={dz}, sigma={sigma}")
        
        # Final check to ensure we're not sending a completely black image
        if np.count_nonzero(self.current_frame) == 0:
            print("CRITICAL: Final image is completely black, setting to gray")
            if self.pixel_format == "MONO8":
                self.current_frame = np.ones((self.Height, self.Width), dtype=np.uint8) * 128
            elif self.pixel_format == "MONO12":
                self.current_frame = np.ones((self.Height, self.Width), dtype=np.uint16) * 2048
            elif self.pixel_format == "MONO16":
                self.current_frame = np.ones((self.Height, self.Width), dtype=np.uint16) * 32768

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
