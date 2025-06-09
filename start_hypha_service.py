import os
import logging
import logging.handlers
import time
import argparse
import asyncio
import fractions
from functools import partial
import traceback
import numpy as np
from hypha_rpc import login, connect_to_server, register_rtc_service
import json
import cv2
import dotenv
import sys
import io
from PIL import Image  
# Now you can import squid_control
from squid_control.squid_controller import SquidController
from squid_control.control.camera import TriggerModeSetting
from pydantic import Field, BaseModel
from typing import List, Optional

from squid_control.hypha_tools.hypha_storage import HyphaDataStore
from squid_control.hypha_tools.chatbot.aask import aask
import base64
from pydantic import Field
from hypha_rpc.utils.schema import schema_function
import signal
from squid_control.hypha_tools.artifact_manager.artifact_manager import SquidArtifactManager

# WebRTC imports
import aiohttp
from av import VideoFrame
from aiortc import MediaStreamTrack

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  

# Set up logging

def setup_logging(log_file="squid_control_service.log", max_bytes=100000, backup_count=3):
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

class MicroscopeVideoTrack(MediaStreamTrack):
    """
    A video stream track that provides real-time microscope images.
    """

    kind = "video"

    def __init__(self, microscope_instance):
        super().__init__()
        self.microscope_instance = microscope_instance
        self.count = 0
        self.running = True
        self.start_time = None
        self.fps = 3 # Target FPS for WebRTC stream
        self.frame_width = 720
        self.frame_height = 720
        logger.info("MicroscopeVideoTrack initialized")

    def draw_crosshair(self, img, center_x, center_y, size=20, color=[255, 255, 255]):
        """Draw a crosshair at the specified position"""
        height, width = img.shape[:2]
        
        # Horizontal line
        if 0 <= center_y < height:
            start_x = max(0, center_x - size)
            end_x = min(width, center_x + size)
            img[center_y, start_x:end_x] = color
        
        # Vertical line
        if 0 <= center_x < width:
            start_y = max(0, center_y - size)
            end_y = min(height, center_y + size)
            img[start_y:end_y, center_x] = color

    async def recv(self):
        if not self.running:
            logger.warning("MicroscopeVideoTrack: recv() called but track is not running")
            raise Exception("Track stopped")
            
        try:
            if self.start_time is None:
                self.start_time = time.time()
            
            next_frame_time = self.start_time + (self.count / self.fps)
            sleep_duration = next_frame_time - time.time()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

            # Get frame using the get_video_frame method with frame dimensions
            processed_frame = await self.microscope_instance.get_video_frame(
                frame_width=self.frame_width,
                frame_height=self.frame_height
            )
            
            new_video_frame = VideoFrame.from_ndarray(processed_frame, format="rgb24")
            new_video_frame.pts = self.count
            new_video_frame.time_base = fractions.Fraction(1, self.fps)
            
            if self.count % (self.fps * 5) == 0:  # Log every 5 seconds
                logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}")
            
            self.count += 1
            return new_video_frame
            
        except Exception as e:
            logger.error(f"MicroscopeVideoTrack: Error in recv(): {e}", exc_info=True)
            self.running = False
            raise

    def stop(self):
        logger.info("MicroscopeVideoTrack stop() called.")
        self.running = False

class Microscope:
    def __init__(self, is_simulation, is_local):
        self.login_required = True
        self.current_x = 0
        self.current_y = 0
        self.current_z = 0
        self.current_theta = 0
        self.current_illumination_channel = None
        self.current_intensity = None
        self.is_illumination_on = False
        self.chatbot_service_url = None
        self.is_simulation = is_simulation
        self.is_local = is_local
        self.squidController = SquidController(is_simulation=is_simulation)
        self.squidController.move_to_well('C',3)
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.BF_intensity_exposure = [50, 100]
        self.F405_intensity_exposure = [50, 100]
        self.F488_intensity_exposure = [50, 100]
        self.F561_intensity_exposure = [50, 100]
        self.F638_intensity_exposure = [50, 100]
        self.F730_intensity_exposure = [50, 100]
        self.channel_param_map = {
            0: 'BF_intensity_exposure',
            11: 'F405_intensity_exposure',
            12: 'F488_intensity_exposure',
            13: 'F638_intensity_exposure',
            14: 'F561_intensity_exposure',
            15: 'F730_intensity_exposure',
        }
        self.parameters = {
            'current_x': self.current_x,
            'current_y': self.current_y,
            'current_z': self.current_z,
            'current_theta': self.current_theta,
            'is_illumination_on': self.is_illumination_on,
            'dx': self.dx,
            'dy': self.dy,
            'dz': self.dz,
            'BF_intensity_exposure': self.BF_intensity_exposure,
            'F405_intensity_exposure': self.F405_intensity_exposure,
            'F488_intensity_exposure': self.F488_intensity_exposure,
            'F561_intensity_exposure': self.F561_intensity_exposure,
            'F638_intensity_exposure': self.F638_intensity_exposure,
            'F730_intensity_exposure': self.F730_intensity_exposure,
        }
        self.authorized_emails = self.load_authorized_emails(self.login_required)
        logger.info(f"Authorized emails: {self.authorized_emails}")
        self.datastore = None
        self.server_url = "http://reef.dyn.scilifelab.se:9527" if is_local else "https://hypha.aicell.io/"
        self.server = None
        self.service_id = os.environ.get("MICROSCOPE_SERVICE_ID")
        self.setup_task = None  # Track the setup task
        
        # WebRTC related attributes
        self.video_track = None
        self.webrtc_service_id = None
        self.is_streaming = False
        self.similarity_search_svc = None
        self.video_contrast_min = 0
        self.video_contrast_max = None

        # Add task status tracking
        self.task_status = {
            "move_by_distance": "not_started",
            "move_to_position": "not_started",
            "get_status": "not_started",
            "update_parameters_from_client": "not_started",
            "one_new_frame": "not_started",
            "snap": "not_started",
            "open_illumination": "not_started",
            "close_illumination": "not_started",
            "scan_well_plate": "not_started",
            "scan_well_plate_simulated": "not_started",
            "set_illumination": "not_started",
            "set_camera_exposure": "not_started",
            "stop_scan": "not_started",
            "home_stage": "not_started",
            "return_stage": "not_started",
            "move_to_loading_position": "not_started",
            "auto_focus": "not_started",
            "do_laser_autofocus": "not_started",
            "set_laser_reference": "not_started",
            "navigate_to_well": "not_started",
            "get_chatbot_url": "not_started",
            "adjust_video_frame": "not_started",
        }

    def load_authorized_emails(self, login_required=True):
        if login_required:
            authorized_users_path = os.environ.get("BIOIMAGEIO_AUTHORIZED_USERS_PATH")
            if authorized_users_path:
                assert os.path.exists(
                    authorized_users_path
                ), f"The authorized users file is not found at {authorized_users_path}"
                with open(authorized_users_path, "r") as f:
                    authorized_users = json.load(f)["users"]
                authorized_emails = [
                    user["email"] for user in authorized_users if "email" in user
                ]
            else:
                authorized_emails = None
        else:
            authorized_emails = None
        return authorized_emails

    def check_permission(self, user):
        if user['is_anonymous']:
            return False
        if self.authorized_emails is None or user["email"] in self.authorized_emails:
            return True
        else:
            return False

    async def ping(self, context=None):
        if self.login_required and context and context.get("user"):
            assert self.check_permission(
                context.get("user")
            ), "You don't have permission to use the chatbot, please sign up and wait for approval"
        return "pong"

    def get_task_status(self, task_name):
        """Get the status of a specific task"""
        return self.task_status.get(task_name, "unknown")
    
    @schema_function(skip_self=True)
    def get_all_task_status(self):
        """Get the status of all tasks"""
        logger.info(f"Task status: {self.task_status}")
        return self.task_status

    def reset_task_status(self, task_name):
        """Reset the status of a specific task"""
        if task_name in self.task_status:
            self.task_status[task_name] = "not_started"
    
    def reset_all_task_status(self):
        """Reset the status of all tasks"""
        for task_name in self.task_status:
            self.task_status[task_name] = "not_started"
    @schema_function(skip_self=True)
    def hello_world(self):
        """Hello world"""
        task_name = "hello_world"
        self.task_status[task_name] = "started"
        self.task_status[task_name] = "finished"
        return "Hello world"
    
    @schema_function(skip_self=True)
    def move_by_distance(self, x: float=Field(1.0, description="disntance through X axis, unit: milimeter"), y: float=Field(1.0, description="disntance through Y axis, unit: milimeter"), z: float=Field(1.0, description="disntance through Z axis, unit: milimeter"), context=None):
        """
        Move the stage by a distances in x, y, z axis
        Returns: Result information
        """
        task_name = "move_by_distance"
        self.task_status[task_name] = "started"
        try:
            is_success, x_pos, y_pos, z_pos, x_des, y_des, z_des = self.squidController.move_by_distance_limited(x, y, z)
            if is_success:
                result = f'The stage moved ({x},{y},{z})mm through x,y,z axis, from ({x_pos},{y_pos},{z_pos})mm to ({x_des},{y_des},{z_des})mm'
                self.task_status[task_name] = "finished"
                return {
                    "success": True,
                    "message": result,
                    "initial_position": {"x": x_pos, "y": y_pos, "z": z_pos},
                    "final_position": {"x": x_des, "y": y_des, "z": z_des}
                }
            else:
                result = f'The stage cannot move ({x},{y},{z})mm through x,y,z axis, from ({x_pos},{y_pos},{z_pos})mm to ({x_des},{y_des},{z_des})mm because out of the range.'
                self.task_status[task_name] = "failed"
                return {
                    "success": False,
                    "message": result,
                    "initial_position": {"x": x_pos, "y": y_pos, "z": z_pos},
                    "attempted_position": {"x": x_des, "y": y_des, "z": z_des}
                }
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to move by distance: {e}")
            raise e

    @schema_function(skip_self=True)
    def move_to_position(self, x:float=Field(1.0,description="Unit: milimeter"), y:float=Field(1.0,description="Unit: milimeter"), z:float=Field(1.0,description="Unit: milimeter"), context=None):
        """
        Move the stage to a position in x, y, z axis
        Returns: The result of the movement
        """
        task_name = "move_to_position"
        self.task_status[task_name] = "started"
        try:
            self.get_status()
            initial_x = self.parameters['current_x']
            initial_y = self.parameters['current_y']
            initial_z = self.parameters['current_z']

            if x != 0:
                is_success, x_pos, y_pos, z_pos, x_des = self.squidController.move_x_to_limited(x)
                if not is_success:
                    self.task_status[task_name] = "failed"
                    return {
                        "success": False,
                        "message": f'The stage cannot move to position ({x},{y},{z})mm from ({initial_x},{initial_y},{initial_z})mm because out of the limit of X axis.',
                        "initial_position": {"x": initial_x, "y": initial_y, "z": initial_z},
                        "final_position": {"x": x_pos, "y": y_pos, "z": z_pos}
                    }

            if y != 0:
                is_success, x_pos, y_pos, z_pos, y_des = self.squidController.move_y_to_limited(y)
                if not is_success:
                    self.task_status[task_name] = "failed"
                    return {
                        "success": False,
                        "message": f'X axis moved successfully, the stage is now at ({x_pos},{y_pos},{z_pos})mm. But aimed position is out of the limit of Y axis and the stage cannot move to position ({x},{y},{z})mm.',
                        "initial_position": {"x": initial_x, "y": initial_y, "z": initial_z},
                        "final_position": {"x": x_pos, "y": y_pos, "z": z_pos}
                    }

            if z != 0:
                is_success, x_pos, y_pos, z_pos, z_des = self.squidController.move_z_to_limited(z)
                if not is_success:
                    self.task_status[task_name] = "failed"
                    return {
                        "success": False,
                        "message": f'X and Y axis moved successfully, the stage is now at ({x_pos},{y_pos},{z_pos})mm. But aimed position is out of the limit of Z axis and the stage cannot move to position ({x},{y},{z})mm.',
                        "initial_position": {"x": initial_x, "y": initial_y, "z": initial_z},
                        "final_position": {"x": x_pos, "y": y_pos, "z": z_pos}
                    }

            self.task_status[task_name] = "finished"
            return {
                "success": True,
                "message": f'The stage moved to position ({x},{y},{z})mm from ({initial_x},{initial_y},{initial_z})mm successfully.',
                "initial_position": {"x": initial_x, "y": initial_y, "z": initial_z},
                "final_position": {"x": x_pos, "y": y_pos, "z": z_pos}
            }
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to move to position: {e}")
            raise e

    @schema_function(skip_self=True)
    def get_status(self, context=None):
        """
        Get the current status of the microscope
        Returns: Status of the microscope
        """
        task_name = "get_status"
        self.task_status[task_name] = "started"
        try:
            current_x, current_y, current_z, current_theta = self.squidController.navigationController.update_pos(microcontroller=self.squidController.microcontroller)
            is_illumination_on = self.squidController.liveController.illumination_on
            scan_channel = self.squidController.multipointController.selected_configurations
            is_busy = self.squidController.is_busy
            self.parameters = {
                'is_busy': is_busy,
                'current_x': current_x,
                'current_y': current_y,
                'current_z': current_z,
                'current_theta': current_theta,
                'is_illumination_on': is_illumination_on,
                'dx': self.dx,
                'dy': self.dy,
                'dz': self.dz,
                'current_channel': self.squidController.current_channel,
                'current_channel_name': self.channel_param_map[self.squidController.current_channel],
                'BF_intensity_exposure': self.BF_intensity_exposure,
                'F405_intensity_exposure': self.F405_intensity_exposure,
                'F488_intensity_exposure': self.F488_intensity_exposure,
                'F561_intensity_exposure': self.F561_intensity_exposure,
                'F638_intensity_exposure': self.F638_intensity_exposure,
                'F730_intensity_exposure': self.F730_intensity_exposure,
            }
            self.task_status[task_name] = "finished"
            return self.parameters
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to get status: {e}")
            raise e

    @schema_function(skip_self=True)
    def update_parameters_from_client(self, new_parameters: dict=Field(description="the dictionary parameters user want to update"), context=None):
        """
        Update the parameters from the client side
        Returns: Updated parameters in the microscope
        """
        task_name = "update_parameters_from_client"
        self.task_status[task_name] = "started"
        try:
            if self.parameters is None:
                self.parameters = {}

            # Update only the specified keys
            for key, value in new_parameters.items():
                if key in self.parameters:
                    self.parameters[key] = value
                    logger.info(f"Updated {key} to {value}")

                    # Update the corresponding instance variable if it exists
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        logger.error(f"Attribute {key} does not exist on self, skipping update.")
                else:
                    logger.error(f"Key {key} not found in parameters, skipping update.")

            self.task_status[task_name] = "finished"
            return {"success": True, "message": "Parameters updated successfully.", "updated_parameters": new_parameters}
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to update parameters: {e}")
            raise e

    @schema_function(skip_self=True)
    def set_simulated_sample_data_alias(self, sample_data_alias: str=Field("agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38", description="The alias of the sample data")):
        """
        Set the alias of simulated sample
        """
        self.squidController.set_simulated_sample_data_alias(sample_data_alias)
        return f"The alias of simulated sample is set to {sample_data_alias}"
    
    @schema_function(skip_self=True)
    def get_simulated_sample_data_alias(self):
        """
        Get the alias of simulated sample
        """
        return self.squidController.get_simulated_sample_data_alias()
    
    @schema_function(skip_self=True)
    async def one_new_frame(self, context=None):
        """
        Get an image from the microscope
        Returns: A numpy array with preserved bit depth
        """
        task_name = "one_new_frame"
        self.task_status[task_name] = "started"
        channel = self.squidController.current_channel
        intensity, exposure_time = 50, 100  # Default values
        try:
            #update the current illumination channel and intensity
            param_name = self.channel_param_map.get(channel)
            if param_name:
                stored_params = getattr(self, param_name, None)
                if stored_params and isinstance(stored_params, list) and len(stored_params) == 2:
                    intensity, exposure_time = stored_params
                else:
                    logger.warning(f"Parameter {param_name} for channel {channel} is not properly initialized. Using defaults.")
            else:
                logger.warning(f"Unknown channel {channel} in one_new_frame. Using default intensity/exposure.")
            
            # Get the raw image from the camera with original bit depth preserved
            raw_img = await self.squidController.snap_image(channel, intensity, exposure_time)
            
            # Resize to 3000x3000 while preserving bit depth
            resized_img = cv2.resize(raw_img, (3000, 3000))
            
            self.get_status()
            self.task_status[task_name] = "finished"
            
            # Return the numpy array directly with preserved bit depth
            return resized_img
            
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to get new frame: {e}")
            raise e

    @schema_function(skip_self=True)
    async def get_video_frame(self, frame_width: int=Field(640, description="Width of the video frame"), frame_height: int=Field(480, description="Height of the video frame"), context=None):
        """
        Get the raw frame from the microscope
        Returns: A processed frame ready for video streaming
        """
        try:
            # Get current channel and parameters
            channel = self.squidController.current_channel
            param_name = self.channel_param_map.get(channel)
            intensity, exposure_time = 10, 10  # Default values
            if param_name:
                stored_params = getattr(self, param_name, None)
                if stored_params and isinstance(stored_params, list) and len(stored_params) == 2:
                    intensity, exposure_time = stored_params

            # Get frame directly using snap_image
            if self.is_simulation:
                raw_frame = await self.squidController.get_camera_frame_simulation(channel, intensity, exposure_time)
            else:
                raw_frame = self.squidController.get_camera_frame(channel, intensity, exposure_time)
            
            # Adjust contrast
            min_val = self.video_contrast_min
            max_val = self.video_contrast_max

            if max_val is None:
                if raw_frame.dtype == np.uint16:
                    max_val = 65535
                else:
                    max_val = 255
            
            # Clip and scale to 0-255
            processed_frame = np.clip(raw_frame, min_val, max_val)
            if max_val > min_val:
                processed_frame = ((processed_frame.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                processed_frame = np.zeros_like(raw_frame, dtype=np.uint8)
            
            # Convert to RGB if needed
            if len(processed_frame.shape) == 2:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
            elif processed_frame.shape[2] == 1:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
                
            # Resize to standard dimensions
            processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error getting video frame: {e}", exc_info=True)
            # Return a blank frame with error message
            placeholder_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, f"Error: {str(e)}", (10, frame_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            return placeholder_img

    @schema_function(skip_self=True)
    def adjust_video_frame(self, min_val: int = Field(0, description="Minimum intensity value for contrast stretching"), max_val: Optional[int] = Field(None, description="Maximum intensity value for contrast stretching"), context=None):
        """Adjust the contrast of the video stream by setting min and max intensity values."""
        task_name = "adjust_video_frame"
        self.task_status[task_name] = "started"
        try:
            self.video_contrast_min = min_val
            self.video_contrast_max = max_val
            logger.info(f"Video contrast adjusted: min={min_val}, max={max_val}")
            self.task_status[task_name] = "finished"
            return {"success": True, "message": f"Video contrast adjusted to min={min_val}, max={max_val}."}
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to adjust video frame: {e}")
            raise e

    @schema_function(skip_self=True)
    async def snap(self, exposure_time: int=Field(100, description="Exposure time, in milliseconds"), channel: int=Field(0, description="Light source (0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)"), intensity: int=Field(50, description="Intensity of the illumination source"), context=None):
        """
        Get an image from microscope
        Returns: the URL of the image
        """
        task_name = "snap"
        self.task_status[task_name] = "started"
        try:
            gray_img = await self.squidController.snap_image(channel, intensity, exposure_time)
            logger.info('The image is snapped')
            gray_img = gray_img.astype(np.uint8)
            # Resize the image to a standard size
            resized_img = cv2.resize(gray_img, (2048, 2048))

            # Encode the image directly to PNG without converting to BGR
            _, png_image = cv2.imencode('.png', resized_img)

            # Store the PNG image
            file_id = self.datastore.put('file', png_image.tobytes(), 'snapshot.png', "Captured microscope image in PNG format")
            data_url = self.datastore.get_url(file_id)
            logger.info(f'The image is snapped and saved as {data_url}')
            
            #update the current illumination channel and intensity
            self.squidController.current_channel = channel
            param_name = self.channel_param_map.get(channel)
            if param_name:
                setattr(self, param_name, [intensity, exposure_time])
            else:
                logger.warning(f"Unknown channel {channel} in snap, parameters not updated for intensity/exposure attributes.")
            
            self.get_status()
            self.task_status[task_name] = "finished"
            return data_url
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to snap image: {e}")
            raise e

    @schema_function(skip_self=True)
    def open_illumination(self, context=None):
        """
        Turn on the illumination
        Returns: The message of the action
        """
        task_name = "open_illumination"
        self.task_status[task_name] = "started"
        try:
            self.squidController.liveController.turn_on_illumination()
            logger.info('Bright field illumination turned on.')
            self.task_status[task_name] = "finished"
            return 'Bright field illumination turned on.'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to open illumination: {e}")
            raise e

    @schema_function(skip_self=True)
    def close_illumination(self, context=None):
        """
        Turn off the illumination
        Returns: The message of the action
        """
        task_name = "close_illumination"
        self.task_status[task_name] = "started"
        try:
            self.squidController.liveController.turn_off_illumination()
            logger.info('Illumination turned off.')
            self.task_status[task_name] = "finished"
            return 'Illumination turned off.'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to close illumination: {e}")
            raise e

    @schema_function(skip_self=True)
    def scan_well_plate(self, well_plate_type: str=Field("96", description="Type of the well plate (e.g., '6', '12', '24', '96', '384')"), illumination_settings: List[dict]=Field(default_factory=lambda: [{'channel': 'BF LED matrix full', 'intensity': 28.0, 'exposure_time': 20.0}, {'channel': 'Fluorescence 488 nm Ex', 'intensity': 27.0, 'exposure_time': 60.0}, {'channel': 'Fluorescence 561 nm Ex', 'intensity': 98.0, 'exposure_time': 100.0}], description="Illumination settings with channel name, intensity (0-100), and exposure time (ms) for each channel"), do_contrast_autofocus: bool=Field(False, description="Whether to do contrast based autofocus"), do_reflection_af: bool=Field(True, description="Whether to do reflection based autofocus"), scanning_zone: List[tuple]=Field(default_factory=lambda: [(0,0),(0,0)], description="The scanning zone of the well plate, for 96 well plate, it should be[(0,0),(7,11)] "), Nx: int=Field(3, description="Number of columns to scan"), Ny: int=Field(3, description="Number of rows to scan"), action_ID: str=Field('testPlateScan', description="The ID of the action"), context=None):
        """
        Scan the well plate according to the pre-defined position list with custom illumination settings
        Returns: The message of the action
        """
        task_name = "scan_well_plate"
        self.task_status[task_name] = "started"
        try:
            if illumination_settings is None:
                illumination_settings = [
                    {'channel': 'BF LED matrix full', 'intensity': 28.0, 'exposure_time': 20.0},
                    {'channel': 'Fluorescence 488 nm Ex', 'intensity': 27.0, 'exposure_time': 60.0},
                    {'channel': 'Fluorescence 561 nm Ex', 'intensity': 98.0, 'exposure_time': 100.0}
                ]
            logger.info("Start scanning well plate with custom illumination settings")
            self.squidController.plate_scan(well_plate_type, illumination_settings, do_contrast_autofocus, do_reflection_af, scanning_zone, Nx, Ny, action_ID)
            logger.info("Well plate scanning completed")
            self.task_status[task_name] = "finished"
            return "Well plate scanning completed"
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to scan well plate: {e}")
            raise e
    
    @schema_function(skip_self=True)
    def scan_well_plate_simulated(self, context=None):
        """
        Scan the well plate according to the pre-defined position list
        Returns: The message of the action
        """
        task_name = "scan_well_plate_simulated"
        self.task_status[task_name] = "started"
        try:
            time.sleep(600)
            self.task_status[task_name] = "finished"
            return "Well plate scanning completed"
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to scan well plate: {e}")
            raise e


    @schema_function(skip_self=True)
    def set_illumination(self, channel: int=Field(0, description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)"), intensity: int=Field(50, description="Intensity of the illumination source"), context=None):
        """
        Set the intensity of light source
        Returns:A string message
        """
        task_name = "set_illumination"
        self.task_status[task_name] = "started"
        try:
            # if light is on, turn it off first
            if self.squidController.liveController.illumination_on:
                self.squidController.liveController.turn_off_illumination()
                time.sleep(0.005)
                self.squidController.liveController.set_illumination(channel, intensity)
                self.squidController.liveController.turn_on_illumination()
                time.sleep(0.005)
            else:
                self.squidController.liveController.set_illumination(channel, intensity)
                time.sleep(0.005)
                
            param_name = self.channel_param_map.get(channel)
            self.squidController.current_channel = channel
            if param_name:
                current_params = getattr(self, param_name, [intensity, 100]) # Default exposure if not found
                if not (isinstance(current_params, list) and len(current_params) == 2):
                    logger.warning(f"Parameter {param_name} for channel {channel} was not a list of two items. Resetting with default exposure.")
                    current_params = [intensity, 100] # Default exposure
                setattr(self, param_name, [intensity, current_params[1]])
            else:
                logger.warning(f"Unknown channel {channel} in set_illumination, parameters not updated for intensity attributes.")
                
            logger.info(f'The intensity of the channel {channel} illumination is set to {intensity}.')
            self.task_status[task_name] = "finished"
            return f'The intensity of the channel {channel} illumination is set to {intensity}.'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to set illumination: {e}")
            raise e
    
    @schema_function(skip_self=True)
    def set_camera_exposure(self,channel: int=Field(..., description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)"), exposure_time: int=Field(..., description="Exposure time in milliseconds"), context=None):
        """
        Set the exposure time of the camera
        Returns: A string message
        """
        task_name = "set_camera_exposure"
        self.task_status[task_name] = "started"
        try:
            self.squidController.camera.set_exposure_time(exposure_time)
            
            param_name = self.channel_param_map.get(channel)
            self.squidController.current_channel = channel
            if param_name:
                current_params = getattr(self, param_name, [50, exposure_time]) # Default intensity if not found
                if not (isinstance(current_params, list) and len(current_params) == 2):
                    logger.warning(f"Parameter {param_name} for channel {channel} was not a list of two items. Resetting with default intensity.")
                    current_params = [50, exposure_time] # Default intensity
                setattr(self, param_name, [current_params[0], exposure_time])
            else:
                logger.warning(f"Unknown channel {channel} in set_camera_exposure, parameters not updated for exposure attributes.")

            logger.info(f'The exposure time of the camera is set to {exposure_time}.')
            self.task_status[task_name] = "finished"
            return f'The exposure time of the camera is set to {exposure_time}.'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to set camera exposure: {e}")
            raise e

    @schema_function(skip_self=True)
    def stop_scan(self, context=None):
        """
        Stop the scanning of the well plate.
        Returns: A string message
        """
        task_name = "stop_scan"
        self.task_status[task_name] = "started"
        try:
            self.squidController.liveController.stop_live()
            self.multipointController.abort_acqusition_requested=True
            logger.info("Stop scanning well plate")
            self.task_status[task_name] = "finished"
            return "Stop scanning well plate"
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to stop scan: {e}")
            raise e

    @schema_function(skip_self=True)
    def home_stage(self, context=None):
        """
        Move the stage to home/zero position
        Returns: A string message
        """
        task_name = "home_stage"
        self.task_status[task_name] = "started"
        try:
            self.squidController.home_stage()
            logger.info('The stage moved to home position in z, y, and x axis')
            self.task_status[task_name] = "finished"
            return 'The stage moved to home position in z, y, and x axis'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to home stage: {e}")
            raise e
    
    @schema_function(skip_self=True)
    def return_stage(self,context=None):
        """
        Move the stage to the initial position for imaging.
        Returns: A string message
        """
        task_name = "return_stage"
        self.task_status[task_name] = "started"
        try:
            self.squidController.return_stage()
            logger.info('The stage moved to the initial position')
            self.task_status[task_name] = "finished"
            return 'The stage moved to the initial position'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to return stage: {e}")
            raise e
    
    @schema_function(skip_self=True)
    def move_to_loading_position(self, context=None):
        """
        Move the stage to the loading position.
        Returns: A  string message
        """
        task_name = "move_to_loading_position"
        self.task_status[task_name] = "started"
        try:
            self.squidController.slidePositionController.move_to_slide_loading_position()
            logger.info('The stage moved to loading position')
            self.task_status[task_name] = "finished"
            return 'The stage moved to loading position'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to move to loading position: {e}")
            raise e

    @schema_function(skip_self=True)
    def auto_focus(self, context=None):
        """
        Do contrast-based autofocus
        Returns: A string message
        """
        task_name = "auto_focus"
        self.task_status[task_name] = "started"
        try:
            self.squidController.do_autofocus()
            logger.info('The camera is auto-focused')
            self.task_status[task_name] = "finished"
            return 'The camera is auto-focused'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to auto focus: {e}")
            raise e
    
    @schema_function(skip_self=True)
    def do_laser_autofocus(self, context=None):
        """
        Do reflection-based autofocus
        Returns: A string message
        """
        task_name = "do_laser_autofocus"
        self.task_status[task_name] = "started"
        try:
            self.squidController.do_laser_autofocus()
            logger.info('The camera is auto-focused')
            self.task_status[task_name] = "finished"
            return 'The camera is auto-focused'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to do laser autofocus: {e}")
            raise e
        
    @schema_function(skip_self=True)
    def set_laser_reference(self, context=None):
        """
        Set the reference of the laser
        Returns: A string message
        """
        task_name = "set_laser_reference"
        self.task_status[task_name] = "started"
        try:
            if self.is_simulation:
                pass
            else:
                self.squidController.laserAutofocusController.set_reference()
            logger.info('The laser reference is set')
            self.task_status[task_name] = "finished"
            return 'The laser reference is set'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to set laser reference: {e}")
            raise e
        
    @schema_function(skip_self=True)
    def navigate_to_well(self, row: str=Field('A', description="Row number of the well position (e.g., 'A')"), col: int=Field(1, description="Column number of the well position"), wellplate_type: str=Field('96', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')"), context=None):
        """
        Navigate to the specified well position in the well plate.
        Returns: A string message
        """
        task_name = "navigate_to_well"
        self.task_status[task_name] = "started"
        try:
            if wellplate_type is None:
                wellplate_type = '96'
            self.squidController.move_to_well(row, col, wellplate_type)
            logger.info(f'The stage moved to well position ({row},{col})')
            self.task_status[task_name] = "finished"
            return f'The stage moved to well position ({row},{col})'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to navigate to well: {e}")
            raise e

    @schema_function(skip_self=True)
    def get_chatbot_url(self, context=None):
        """
        Get the URL of the chatbot service.
        Returns: A URL string
        """
        task_name = "get_chatbot_url"
        self.task_status[task_name] = "started"
        try:
            logger.info(f"chatbot_service_url: {self.chatbot_service_url}")
            self.task_status[task_name] = "finished"
            return self.chatbot_service_url
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to get chatbot URL: {e}")
            raise e
    
    def _format_similarity_results(self, results):
        if not results:
            return "No similar images found."

        response_parts = ["Found similar images:"]
        for res in results:
            img_str = res.get('image_base64')
            if not img_str:
                continue
            
            image_bytes = base64.b64decode(img_str)
            file_path = res.get('file_path', '')
            file_name = os.path.basename(file_path) if file_path else "similar_image.png"
            description = res.get('text_description', 'No description available.')

            file_id = self.datastore.put('file', image_bytes, file_name, f"Similar image found: {description}")
            img_url = self.datastore.get_url(file_id)

            similarity = res.get('similarity', 0.0)
            
            response_parts.append(f"![Found image]({img_url})\nDescription: {description}\nSimilarity: {similarity:.4f}")

        return "\n\n".join(response_parts)

    @schema_function(skip_self=True)
    async def find_similar_image_text(self, query_input: str, top_k: int, context=None):
        """
        Find similar image with text query.
        Returns: A list of image information.
        """
        try:
            results = await self.similarity_search_svc.find_similar_images(query_input, top_k)
            return self._format_similarity_results(results)
        except Exception as e:
            logger.error(f"Failed to find similar images by text: {e}")
            return f"An error occurred while searching for similar images: {e}"

    @schema_function(skip_self=True)
    async def find_similar_image_image(self, query_input: str, top_k: int, context=None):
        """
        Find similar image with image's URL query.
        Returns: A list of image information.
        """
        try:
            # download the image from query_input url
            async with aiohttp.ClientSession() as session:
                async with session.get(query_input) as resp:
                    if resp.status != 200:
                        return f"Failed to download image from {query_input}"
                    image_bytes = await resp.read()
        except Exception as e:
            logger.error(f"Failed to download image from {query_input}: {e}")
            return f"Failed to download image from {query_input}: {e}"
        
        try:
            results = await self.similarity_search_svc.find_similar_images(image_bytes, top_k)
            return self._format_similarity_results(results)
        except Exception as e:
            logger.error(f"Failed to find similar images by image: {e}")
            return f"An error occurred while searching for similar images: {e}"

    async def fetch_ice_servers(self):
        """Fetch ICE servers from the coturn service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://ai.imjoy.io/public/services/coturn/get_rtc_ice_servers') as response:
                    if response.status == 200:
                        ice_servers = await response.json()
                        logger.info("Successfully fetched ICE servers")
                        return ice_servers
                    else:
                        logger.warning(f"Failed to fetch ICE servers, status: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching ICE servers: {e}")
            return None
    
    class MoveByDistanceInput(BaseModel):
        """Move the stage by a distance in x, y, z axis."""
        x: float = Field(0, description="Move the stage along X axis")
        y: float = Field(0, description="Move the stage along Y axis")
        z: float = Field(0, description="Move the stage along Z axis")

    class MoveToPositionInput(BaseModel):
        """Move the stage to a position in x, y, z axis."""
        x: Optional[float] = Field(None, description="Move the stage to the X coordinate")
        y: Optional[float] = Field(None, description="Move the stage to the Y coordinate")
        z: float = Field(3.35, description="Move the stage to the Z coordinate")

    class SetSimulatedSampleDataAliasInput(BaseModel):
        """Set the alias of simulated sample"""
        sample_data_alias: str = Field("agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38", description="The alias of the sample data")

    class AutoFocusInput(BaseModel):
        """Reflection based autofocus."""
        N: int = Field(10, description="Number of discrete focus positions")
        delta_Z: float = Field(1.524, description="Step size in the Z-axis in micrometers")

    class SnapImageInput(BaseModel):
        """Snap an image from the camera, and display it in the chatbot."""
        exposure: int = Field(..., description="Exposure time in milliseconds")
        channel: int = Field(..., description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)")
        intensity: int = Field(..., description="Intensity of the illumination source")

    class InspectToolInput(BaseModel):
        """Inspect the images with GPT4-o's vision model."""
        images: List[dict] = Field(..., description="A list of images to be inspected, each with a 'http_url' and 'title'")
        query: str = Field(..., description="User query about the image")
        context_description: str = Field(..., description="Context for the visual inspection task, inspect images taken from the microscope")

    class NavigateToWellInput(BaseModel):
        """Navigate to a well position in the well plate."""
        row: str = Field(..., description="Row number of the well position (e.g., 'A')")
        col: int = Field(..., description="Column number of the well position")
        wellplate_type: str = Field('96', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')")

    class MoveToLoadingPositionInput(BaseModel):
        """Move the stage to the loading position."""

    class SetIlluminationInput(BaseModel):
        """Set the intensity of light source."""
        channel: int = Field(..., description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)")
        intensity: int = Field(..., description="Intensity of the illumination source")

    class SetCameraExposureInput(BaseModel):
        """Set the exposure time of the camera."""
        channel: int = Field(..., description="Light source (e.g., 0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)")
        exposure_time: int = Field(..., description="Exposure time in milliseconds")

    class DoLaserAutofocusInput(BaseModel):
        """Do reflection-based autofocus."""

    class SetLaserReferenceInput(BaseModel):
        """Set the reference of the laser."""

    class GetStatusInput(BaseModel):
        """Get the current status of the microscope."""

    class HomeStageInput(BaseModel):
        """Home the stage in z, y, and x axis."""

    class ReturnStageInput(BaseModel):
        """Return the stage to the initial position."""

    class ImageInfo(BaseModel):
        """Image information."""
        url: str = Field(..., description="The URL of the image.")
        title: Optional[str] = Field(None, description="The title of the image.")
    
    class FindSimilarImageTextInput(BaseModel):
        """Find similar image with text query."""
        query_input: str = Field(..., description="The text of the query input for the similarity search.")
        top_k: int = Field(..., description="The number of similar images to return.")

    class FindSimilarImageImageInput(BaseModel):
        """Find similar image with image's URL query."""
        query_input: str = Field(..., description="The URL of the image of the query input for the similarity search.")
        top_k: int = Field(..., description="The number of similar images to return.")

    async def inspect_tool(self, images: List[dict], query: str, context_description: str) -> str:
        image_infos = [
            self.ImageInfo(url=image_dict['http_url'], title=image_dict.get('title'))
            for image_dict in images
        ]
        for image_info_obj in image_infos:
            assert image_info_obj.url.startswith("http"), "Image URL must start with http."
        response = await aask(image_infos, [context_description, query])
        return response

    def move_by_distance_schema(self, config: MoveByDistanceInput, context=None):
        self.get_status()
        x_pos = self.parameters['current_x']
        y_pos = self.parameters['current_y']
        z_pos = self.parameters['current_z']
        result = self.move_by_distance(config.x, config.y, config.z, context)
        return result['message']

    def move_to_position_schema(self, config: MoveToPositionInput, context=None):
        self.get_status()
        x_pos = self.parameters['current_x']
        y_pos = self.parameters['current_y']
        z_pos = self.parameters['current_z']
        x = config.x if config.x is not None else 0
        y = config.y if config.y is not None else 0
        z = config.z if config.z is not None else 0
        result = self.move_to_position(x, y, z, context)
        return result['message']
    
    def auto_focus_schema(self, config: AutoFocusInput, context=None):
        self.auto_focus(context)
        return "Auto-focus completed."

    async def snap_image_schema(self, config: SnapImageInput, context=None):
        image_url = await self.snap(config.exposure, config.channel, config.intensity, context)
        return f"![Image]({image_url})"

    def navigate_to_well_schema(self, config: NavigateToWellInput, context=None):
        self.navigate_to_well(config.row, config.col, config.wellplate_type, context)
        return f'The stage moved to well position ({config.row},{config.col})'

    async def inspect_tool_schema(self, config: InspectToolInput, context=None):
        response = await self.inspect_tool(config.images, config.query, config.context_description)
        return {"result": response}

    def home_stage_schema(self, context=None):
        response = self.home_stage(context)
        return {"result": response}

    def return_stage_schema(self, context=None):
        response = self.return_stage(context)
        return {"result": response}

    async def find_similar_image_text_schema(self, config: FindSimilarImageTextInput, context=None):
        response = await self.find_similar_image_text(config.query_input, config.top_k, context)
        return {"result": response}

    async def find_similar_image_image_schema(self, config: FindSimilarImageImageInput, context=None):
        response = await self.find_similar_image_image(config.query_input, config.top_k, context)
        return {"result": response}

    def set_illumination_schema(self, config: SetIlluminationInput, context=None):
        response = self.set_illumination(config.channel, config.intensity, context)
        return {"result": response}

    def set_camera_exposure_schema(self, config: SetCameraExposureInput, context=None):
        response = self.set_camera_exposure(config.channel, config.exposure_time, context)
        return {"result": response}

    def do_laser_autofocus_schema(self, context=None):
        response = self.do_laser_autofocus(context)
        return {"result": response}

    def set_laser_reference_schema(self, context=None):
        response = self.set_laser_reference(context)
        return {"result": response}

    def get_status_schema(self, context=None):
        response = self.get_status(context)
        return {"result": response}

    def get_schema(self, context=None):
        return {
            "move_by_distance": self.MoveByDistanceInput.model_json_schema(),
            "move_to_position": self.MoveToPositionInput.model_json_schema(),
            "home_stage": self.HomeStageInput.model_json_schema(),
            "return_stage": self.ReturnStageInput.model_json_schema(),
            "auto_focus": self.AutoFocusInput.model_json_schema(),
            "snap_image": self.SnapImageInput.model_json_schema(),
            "inspect_tool": self.InspectToolInput.model_json_schema(),
            "load_position": self.MoveToLoadingPositionInput.model_json_schema(),
            "navigate_to_well": self.NavigateToWellInput.model_json_schema(),
            "set_illumination": self.SetIlluminationInput.model_json_schema(),
            "set_camera_exposure": self.SetCameraExposureInput.model_json_schema(),
            "do_laser_autofocus": self.DoLaserAutofocusInput.model_json_schema(),
            "set_laser_reference": self.SetLaserReferenceInput.model_json_schema(),
            "get_status": self.GetStatusInput.model_json_schema(),
            "find_similar_image_text": self.FindSimilarImageTextInput.model_json_schema(),
            "find_similar_image_image": self.FindSimilarImageImageInput.model_json_schema()
        }

    async def start_hypha_service(self, server, service_id):
        self.server = server
        self.service_id = service_id
        svc = await server.register_service(
            {
                "name": "Microscope Control Service",
                "id": service_id,
                "config": {
                    "visibility": "public",
                    "run_in_executor": True
                },
                "type": "echo",
                "hello_world": self.hello_world,
                "move_by_distance": self.move_by_distance,
                "snap": self.snap,
                "one_new_frame": self.one_new_frame,
                "get_video_frame": self.get_video_frame,
                "off_illumination": self.close_illumination,
                "on_illumination": self.open_illumination,
                "set_illumination": self.set_illumination,
                "set_camera_exposure": self.set_camera_exposure,
                "scan_well_plate": self.scan_well_plate,
                "scan_well_plate_simulated": self.scan_well_plate_simulated,
                "stop_scan": self.stop_scan,
                "home_stage": self.home_stage,
                "return_stage": self.return_stage,
                "navigate_to_well": self.navigate_to_well,
                "move_to_position": self.move_to_position,
                "move_to_loading_position": self.move_to_loading_position,
                "set_simulated_sample_data_alias": self.set_simulated_sample_data_alias,
                "get_simulated_sample_data_alias": self.get_simulated_sample_data_alias,
                "auto_focus": self.auto_focus,
                "do_laser_autofocus": self.do_laser_autofocus,
                "set_laser_reference": self.set_laser_reference,
                "get_status": self.get_status,
                "update_parameters_from_client": self.update_parameters_from_client,
                "get_chatbot_url": self.get_chatbot_url,
                "get_task_status": self.get_task_status,
                "get_all_task_status": self.get_all_task_status,
                "reset_task_status": self.reset_task_status,
                "reset_all_task_status": self.reset_all_task_status,
                "adjust_video_frame": self.adjust_video_frame,
            },
        )

        logger.info(
            f"Service (service_id={service_id}) started successfully, available at {self.server_url}{server.config.workspace}/services"
        )

        if "local" not in service_id:
            await self.register_service_probes(server)
        
        logger.info(f'You can use this service using the service id: {svc.id}')
        id = svc.id.split(":")[1]

        logger.info(f"You can also test the service via the HTTP proxy: {self.server_url}{server.config.workspace}/services/{id}")

    async def start_chatbot_service(self, server, service_id):
        chatbot_extension = {
            "_rintf": True,
            "id": service_id,
            "type": "bioimageio-chatbot-extension",
            "name": "Squid Microscope Control",
            "description": "You are an AI agent controlling microscope. Automate tasks, adjust imaging parameters, and make decisions based on live visual feedback. Solve all the problems from visual feedback; the user only wants to see good results.",
            "config": {"visibility": "public", "require_context": True},
            "ping": self.ping,
            "get_schema": self.get_schema,
            "tools": {
                "move_by_distance": self.move_by_distance_schema,
                "move_to_position": self.move_to_position_schema,
                "auto_focus": self.auto_focus_schema,
                "snap_image": self.snap_image_schema,
                "home_stage": self.home_stage_schema,
                "return_stage": self.return_stage_schema,
                "load_position": self.move_to_loading_position,
                "navigate_to_well": self.navigate_to_well_schema,
                "inspect_tool": self.inspect_tool_schema,
                "set_illumination": self.set_illumination_schema,
                "set_camera_exposure": self.set_camera_exposure_schema,
                "do_laser_autofocus": self.do_laser_autofocus_schema,
                "set_laser_reference": self.set_laser_reference_schema,
                "get_status": self.get_status_schema,
                "find_similar_image_text": self.find_similar_image_text_schema,
                "find_similar_image_image": self.find_similar_image_image_schema
            }
        }

        svc = await server.register_service(chatbot_extension)
        self.chatbot_service_url = f"https://bioimage.io/chat?server=https://chat.bioimage.io&extension={svc.id}&assistant=Skyler"
        logger.info(f"Extension service registered with id: {svc.id}, you can visit the service at:\n {self.chatbot_service_url}")

    async def start_webrtc_service(self, server, webrtc_service_id_arg):
        self.webrtc_service_id = webrtc_service_id_arg 
        
        async def on_init(peer_connection):
            logger.info("WebRTC peer connection initialized")
            
            @peer_connection.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"WebRTC connection state changed to: {peer_connection.connectionState}")
                if peer_connection.connectionState in ["closed", "failed", "disconnected"]:
                    if self.video_track and self.video_track.running:
                        logger.info(f"Connection state is {peer_connection.connectionState}. Stopping video track.")
                        self.video_track.stop()
            
            @peer_connection.on("track")
            def on_track(track):
                logger.info(f"Track {track.kind} received from client")
                
                if self.video_track and self.video_track.running:
                    self.video_track.stop() 
                
                self.video_track = MicroscopeVideoTrack(self) 
                peer_connection.addTrack(self.video_track)
                logger.info("Added MicroscopeVideoTrack to peer connection")
                
                @track.on("ended")
                def on_ended():
                    logger.info(f"Client track {track.kind} ended")
                    if self.video_track:
                        logger.info("Stopping MicroscopeVideoTrack.")
                        self.video_track.stop()  # Now synchronous
                        self.video_track = None

        ice_servers = await self.fetch_ice_servers()
        if not ice_servers:
            logger.warning("Using fallback ICE servers")
            ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

        try:
            await register_rtc_service(
                server,
                service_id=self.webrtc_service_id,
                config={
                    "visibility": "public",
                    "ice_servers": ice_servers,
                    "on_init": on_init,
                },
            )
            logger.info(f"WebRTC service registered with id: {self.webrtc_service_id}")
        except Exception as e:
            logger.error(f"Failed to register WebRTC service ({self.webrtc_service_id}): {e}")
            if "Service already exists" in str(e):
                logger.info(f"WebRTC service {self.webrtc_service_id} already exists. Attempting to retrieve it.")
                try:
                    _ = await server.get_service(self.webrtc_service_id)
                    logger.info(f"Successfully retrieved existing WebRTC service: {self.webrtc_service_id}")
                except Exception as get_e:
                    logger.error(f"Failed to retrieve existing WebRTC service {self.webrtc_service_id}: {get_e}")
                    raise
            else:
                raise
    
    async def connect_to_similarity_search_service(self):
        if self.is_local:
            similarity_search_server = await connect_to_server(
                {"server_url": "http://192.168.2.1:9527", "token": os.environ.get("REEF_LOCAL_TOKEN"), "workspace": os.environ.get("REEF_LOCAL_WORKSPACE"), "ping_interval": None}
            )
            similarity_search_svc = await similarity_search_server.get_service("image-text-similarity-search")
        else:
            similarity_search_server = await connect_to_server(
                {"server_url": "https://hypha.aicell.io", "token": os.environ.get("AGENT_LENS_WORKSPACE_TOKEN"), "workspace": "agent-lens", "ping_interval": None}
            )
            similarity_search_svc = await similarity_search_server.get_service("image-text-similarity-search")
        return similarity_search_svc

    async def setup(self):

        self.similarity_search_svc = await self.connect_to_similarity_search_service()

        remote_token = os.environ.get("SQUID_WORKSPACE_TOKEN")
        remote_server = await connect_to_server(
                {"server_url": "https://hypha.aicell.io", "token": remote_token, "workspace": "squid-control", "ping_interval": None}
            )
        if not self.service_id:
            raise ValueError("MICROSCOPE_SERVICE_ID is not set in the environment variables.")
        if self.is_local:
            token = os.environ.get("REEF_LOCAL_TOKEN")
            workspace = os.environ.get("REEF_LOCAL_WORKSPACE")
            server = await connect_to_server(
                {"server_url": self.server_url, "token": token, "workspace": workspace, "ping_interval": None}
            )
        else:
            try:  
                token = os.environ.get("SQUID_WORKSPACE_TOKEN")  
            except:  
                token = await login({"server_url": self.server_url})
            
            server = await connect_to_server(
                {"server_url": self.server_url, "token": token, "workspace": "squid-control",  "ping_interval": None}
            )
        
        self.server = server
        
        if self.is_simulation:
            await self.start_hypha_service(self.server, service_id=self.service_id)
            datastore_id = f'data-store-simu-{self.service_id}'
            chatbot_id = f"squid-chatbot-simu-{self.service_id}"
        else:
            await self.start_hypha_service(self.server, service_id=self.service_id)
            datastore_id = f'data-store-real-{self.service_id}'
            chatbot_id = f"squid-chatbot-real-{self.service_id}"
        
        self.datastore = HyphaDataStore()
        try:
            await self.datastore.setup(remote_server, service_id=datastore_id)
        except TypeError as e:
            if "Future" in str(e):
                config = await asyncio.wrap_future(server.config)
                await self.datastore.setup(remote_server, service_id=datastore_id, config=config)
            else:
                raise e
    
        chatbot_server_url = "https://chat.bioimage.io"
        try:
            chatbot_token= os.environ.get("WORKSPACE_TOKEN_CHATBOT")
        except:
            chatbot_token = await login({"server_url": chatbot_server_url})
        chatbot_server = await connect_to_server({"server_url": chatbot_server_url, "token": chatbot_token,  "ping_interval": None})
        await self.start_chatbot_service(chatbot_server, chatbot_id)
        webrtc_id = f"video-track-{self.service_id}"
        if not self.is_local: # only start webrtc service in remote mode
            await self.start_webrtc_service(self.server, webrtc_id)

    async def register_service_probes(self, server):
        async def is_service_healthy():
            try:
                microscope_svc = await server.get_service(self.service_id)
                if microscope_svc is None:
                    raise RuntimeError("Microscope service not found")
                
                result = await microscope_svc.hello_world()
                if result != "Hello world":
                    raise RuntimeError(f"Microscope service returned unexpected response: {result}")
                
                datastore_id = f'data-store-{"simu" if self.is_simulation else "real"}-{self.service_id}'
                datastore_svc = await server.get_service(datastore_id)
                if datastore_svc is None:
                    raise RuntimeError("Datastore service not found")
                
                try:
                    if not self.is_simulation:
                        logger.info("Skipping Zarr access check in non-simulation mode.")
                    else:
                        if not hasattr(self.squidController.camera, 'zarr_image_manager') or self.squidController.camera.zarr_image_manager is None:
                            logger.info("ZarrImageManager not initialized yet, initializing it for health check")
                            
                            try:
                                await asyncio.wait_for(
                                    self.initialize_zarr_manager(self.squidController.camera),
                                    timeout=30
                                )
                            except asyncio.TimeoutError:
                                logger.error("ZarrImageManager initialization timed out")
                                raise RuntimeError("ZarrImageManager initialization timed out")
                        
                        logger.info("Testing existing ZarrImageManager instance from simulated camera.")
                        
                        test_result = await asyncio.wait_for(
                            self.squidController.camera.zarr_image_manager.test_zarr_access(
                                dataset_id="agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38",
                                channel="BF_LED_matrix_full",
                                bypass_cache=True
                            ), 
                            50
                        ) 
                        
                        if not test_result.get("success", False):
                            error_msg = test_result.get("message", "Unknown error")
                            raise RuntimeError(f"Zarr access test failed for existing instance: {error_msg}")
                        else:
                            stats = test_result.get("chunk_stats", {})
                            non_zero = stats.get("non_zero_count", 0)
                            total = stats.get("total_size", 1)
                            if total > 0:
                                logger.info(f"Existing Zarr access test succeeded. Non-zero values: {non_zero}/{total} ({(non_zero/total)*100:.1f}%)")
                            else:
                                logger.info("Existing Zarr access test succeeded, but chunk size was zero.")

                except asyncio.TimeoutError:
                    logger.error("Zarr access health check timed out.")
                    raise RuntimeError("Zarr access health check timed out after 50 seconds.")
                except Exception as artifact_error:
                    logger.error(f"Zarr access health check failed: {str(artifact_error)}")
                    raise RuntimeError(f"Zarr access health check failed: {str(artifact_error)}")
                
                chatbot_id = f"squid-chatbot-{'simu' if self.is_simulation else 'real'}-{self.service_id}"
                
                chatbot_server_url = "https://chat.bioimage.io"
                try:
                    chatbot_token = os.environ.get("WORKSPACE_TOKEN_CHATBOT")
                    if not chatbot_token:
                        logger.warning("Chatbot token not found, skipping chatbot health check")
                    else:
                        chatbot_server = await connect_to_server({
                            "server_url": chatbot_server_url, 
                            "token": chatbot_token,
                            "ping_interval": None
                        })
                        chatbot_svc = await asyncio.wait_for(chatbot_server.get_service(chatbot_id), 10)
                        if chatbot_svc is None:
                            raise RuntimeError("Chatbot service not found")
                except Exception as chatbot_error:
                    raise RuntimeError(f"Chatbot service health check failed: {str(chatbot_error)}")
                
                logger.info("All services are healthy")
                return {"status": "ok", "message": "All services are healthy"}
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Service health check failed: {str(e)}")
        
        logger.info("Registering health probes for Kubernetes")
        await server.register_probes({
            f"readiness-{self.service_id}": is_service_healthy,
            f"liveness-{self.service_id}": is_service_healthy
        })
        logger.info("Health probes registered successfully")

    async def initialize_zarr_manager(self, camera):
        from squid_control.hypha_tools.artifact_manager.artifact_manager import ZarrImageManager
        
        camera.zarr_image_manager = ZarrImageManager()
        
        init_success = await camera.zarr_image_manager.connect(
            server_url=self.server_url
        )
        
        if not init_success:
            raise RuntimeError("Failed to initialize ZarrImageManager")
        
        if hasattr(camera, 'scale_level'):
            camera.zarr_image_manager.scale_key = f'scale{camera.scale_level}'
        
        logger.info("ZarrImageManager initialized successfully for health check")
        return camera.zarr_image_manager

# Define a signal handler for graceful shutdown
def signal_handler(sig, frame):
    logger.info('Signal received, shutting down gracefully...')
    microscope.squidController.close()
    sys.exit(0)

# Register the signal handler for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Squid microscope control services for Hypha."
    )
    parser.add_argument(
        "--simulation",
        dest="simulation",
        action="store_true",
        default=False,
        help="Run in simulation mode (default: True)"
    )
    parser.add_argument(
        "--local",
        dest="local",
        action="store_true",
        default=False,
        help="Run with local server URL (default: False)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    microscope = Microscope(is_simulation=args.simulation, is_local=args.local)

    loop = asyncio.get_event_loop()

    async def main():
        try:
            microscope.setup_task = asyncio.create_task(microscope.setup())
            await microscope.setup_task
        except Exception:
            traceback.print_exc()

    loop.create_task(main())
    loop.run_forever()