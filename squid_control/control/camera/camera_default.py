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
        self.simulated_focus = 3.3
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
        self.WORKSPACE_TOKEN = os.getenv("AGENT_LENS_WORKSPACE_TOKEN")
        self.ARTIFACT_ALIAS = "image-map-20250429-treatment-zip"
        self.DEFAULT_TIMESTAMP = "2025-04-29_16-38-27"  # Default timestamp for the dataset
        
        # Initialize these to None, will be set up lazily when needed
        self.zarr_image_manager = None
        self.artifact_manager = None

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

    async def send_trigger(self, x=29.81, y=36.85, dz=0, pixel_size_um=0.333, channel=0, intensity=100, exposure_time=100, magnification_factor=20, performace_mode=False):
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
            async def get_image_from_zarr():
                # Lazily initialize ZarrImageManager if needed
                if self.zarr_image_manager is None:
                    print("Creating new ZarrImageManager instance...")
                    self.zarr_image_manager = ZarrImageManager()
                    print("Connecting to ZarrImageManager...")
                    await self.zarr_image_manager.connect(workspace_token=self.WORKSPACE_TOKEN, server_url=self.SERVER_URL)
                    print("Connected to ZarrImageManager")
                
                # Convert microscope coordinates (mm) to pixel coordinates - fix conversion factor
                pixel_x = int((x / pixel_size_um) * 1000)  # Fix: Proper scaling with parentheses
                pixel_y = int((y / pixel_size_um) * 1000)  # Fix: Proper scaling with parentheses
                
                # Print pixel coordinates for debugging
                print(f"Converted coords (mm) x={x}, y={y} to pixel coords: x={pixel_x}, y={pixel_y}")
                
                # Calculate the pixel range for the image region
                x_start = max(0, pixel_x - self.Width // 2)
                y_start = max(0, pixel_y - self.Height // 2)
                
                # Use the class variables for dataset configuration
                dataset_id = f"agent-lens/{self.ARTIFACT_ALIAS}"  # Fix: Use correctly formatted dataset_id
                timestamp = self.DEFAULT_TIMESTAMP
                
                print(f"Using dataset: {dataset_id}, timestamp: {timestamp}, channel: {channel_name}")
                
                try:
                    # First attempt: Try to use direct Zarr group access for better performance
                    if self.artifact_manager is None:
                        self.artifact_manager = SquidArtifactManager()
                        await self.artifact_manager.connect_server(self.zarr_image_manager.artifact_manager_server)
                    
                    # Extract workspace and artifact_alias from the dataset_id
                    workspace, artifact_alias = "agent-lens", self.ARTIFACT_ALIAS
                    
                    print(f"Retrieving Zarr group with workspace={workspace}, artifact_alias={artifact_alias}")
                    
                    # Get the Zarr group - will use cache automatically based on our changes to SquidArtifactManager
                    zarr_group = await self.artifact_manager.get_zarr_group(
                        workspace=workspace,
                        artifact_alias=artifact_alias,
                        timestamp=timestamp,
                        channel=channel_name
                    )
                    
                    if zarr_group:
                        print(f"Successfully obtained Zarr group for {channel_name}")
                        
                        # Debug: Print available keys in the zarr group
                        print(f"Available keys in Zarr group: {list(zarr_group.keys())}")
                        
                        # Get the appropriate scale level data array with error handling
                        try:
                            scale_array = zarr_group[f'scale0']  # Assuming scale0 is the highest resolution
                            print(f"Scale array shape: {scale_array.shape}, dtype: {scale_array.dtype}")
                        except KeyError:
                            # Try alternative scale naming conventions if 'scale0' doesn't exist
                            if '0' in zarr_group:
                                scale_array = zarr_group['0']
                                print(f"Using alternative scale key '0'. Shape: {scale_array.shape}")
                            else:
                                # If no recognized scale key exists, try the first available key
                                first_key = list(zarr_group.keys())[0]
                                scale_array = zarr_group[first_key]
                                print(f"Using first available key '{first_key}'. Shape: {scale_array.shape}")
                        
                        # Ensure bounds are valid
                        y_start = min(max(0, y_start), scale_array.shape[0] - 1)
                        x_start = min(max(0, x_start), scale_array.shape[1] - 1)
                        
                        # Extract the region of interest directly with bounds checking
                        region_height = min(self.Height, scale_array.shape[0] - y_start)
                        region_width = min(self.Width, scale_array.shape[1] - x_start)
                        
                        if region_height <= 0 or region_width <= 0:
                            raise ValueError(f"Invalid region dimensions: {region_width}x{region_height}")
                        
                        # Get the region data directly from the Zarr array
                        print(f"Reading region from y={y_start} to y={y_start+region_height}, x={x_start} to x={x_start+region_width}")
                        region_data = scale_array[y_start:y_start+region_height, x_start:x_start+region_width]
                        
                        print(f"Region data shape: {region_data.shape}, dtype: {region_data.dtype}, min: {region_data.min()}, max: {region_data.max()}")
                        
                        # If the region is smaller than the expected size, pad it
                        if region_data.shape != (self.Height, self.Width):
                            print(f"Padding region from {region_data.shape} to {(self.Height, self.Width)}")
                            padded_data = np.zeros((self.Height, self.Width), dtype=region_data.dtype)
                            padded_data[:region_height, :region_width] = region_data
                            region_data = padded_data
                        
                        # Ensure the image data is not all zeros
                        if np.count_nonzero(region_data) == 0:
                            print("WARNING: Retrieved region contains all zeros!")
                            # Falling back to default image
                            return None
                            
                        return region_data
                
                except Exception as e:
                    print(f"Direct Zarr access failed: {str(e)}. Falling back to chunk-based approach.")
                    import traceback
                    print(traceback.format_exc())
                    # Continue with the chunk-based approach if direct access fails
                
                # Initialize a numpy array to hold the region data
                region_data = np.zeros((self.Height, self.Width), dtype=np.uint8)
                
                # Determine how many chunks we need to fetch to fill the region
                chunk_size = self.zarr_image_manager.chunk_size
                chunks_x = (self.Width + chunk_size - 1) // chunk_size
                chunks_y = (self.Height + chunk_size - 1) // chunk_size
                
                print(f"Fetching {chunks_x}x{chunks_y} chunks for region at ({x_start}, {y_start}) with size {self.Width}x{self.Height}")
                
                # Fetch chunks and assemble the region
                successful_chunks = 0
                for ty in range(chunks_y):
                    for tx in range(chunks_x):
                        # Calculate the chunk coordinates
                        chunk_x = x_start // chunk_size + tx
                        chunk_y = y_start // chunk_size + ty
                        
                        # Fetch the chunk data - will use cached Zarr group automatically
                        chunk_data = await self.zarr_image_manager.get_region_np_data(
                            dataset_id, 
                            timestamp, 
                            channel_name, 
                            0,  # scale level - 0 is highest resolution
                            chunk_x, 
                            chunk_y
                        )
                        
                        # Calculate the position of this chunk within the region
                        region_x = tx * chunk_size
                        region_y = ty * chunk_size
                        
                        # Calculate the offset within the first chunk
                        offset_x = x_start % chunk_size if tx == 0 else 0
                        offset_y = y_start % chunk_size if ty == 0 else 0
                        
                        # Calculate how much of the chunk to copy
                        copy_width = min(chunk_size - offset_x, self.Width - region_x)
                        copy_height = min(chunk_size - offset_y, self.Height - region_y)
                        
                        if copy_width <= 0 or copy_height <= 0:
                            continue
                        
                        # Copy the chunk data to the region
                        try:
                            if chunk_data is not None and np.count_nonzero(chunk_data) > 0:
                                print(f"Chunk at ({chunk_x},{chunk_y}) has data. Min: {chunk_data.min()}, Max: {chunk_data.max()}")
                                region_data[region_y:region_y+copy_height, region_x:region_x+copy_width] = \
                                    chunk_data[offset_y:offset_y+copy_height, offset_x:offset_x+copy_width]
                                successful_chunks += 1
                            else:
                                print(f"Chunk at ({chunk_x},{chunk_y}) has no valid data.")
                        except Exception as e:
                            print(f"Error copying chunk data: {e}")
                
                print(f"Successfully retrieved {successful_chunks} out of {chunks_x * chunks_y} chunks")
                
                # If we've got at least some valid chunks, return the data
                if successful_chunks > 0 and np.count_nonzero(region_data) > 0:
                    return region_data
                else:
                    print("No valid chunks retrieved, returning None")
                    return None

            try:
                # Use the zarr-based approach to get the image
                self.image = await get_image_from_zarr()
            except Exception as e:
                print(f"Error getting image from ZarrImageManager: {str(e)}")
                import traceback
                print(traceback.format_exc())
                fallback_path = os.path.join(script_dir, f"example-data/{self.image_paths[channel]}")
                print(f"Loading fallback image from: {fallback_path}")
                self.image = np.array(Image.open(fallback_path))

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
