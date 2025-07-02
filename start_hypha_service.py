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
from squid_control.control.config import CONFIG
from squid_control.control.config import ChannelMapper
from pydantic import Field, BaseModel
from typing import List, Optional
from collections import deque
import threading

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

class VideoBuffer:
    """
    Video buffer to store and manage compressed microscope frames for smooth video streaming
    """
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.last_frame_data = None  # Store compressed frame data
        self.last_metadata = None  # Store metadata for last frame
        self.frame_timestamp = 0
        
    def put_frame(self, frame_data, metadata=None):
        """Add a compressed frame and its metadata to the buffer
        
        Args:
            frame_data: dict with compressed frame info from _encode_frame_jpeg()
            metadata: dict with frame metadata including stage position and timestamp
        """
        with self.lock:
            self.buffer.append({
                'frame_data': frame_data,
                'metadata': metadata,
                'timestamp': time.time()
            })
            self.last_frame_data = frame_data
            self.last_metadata = metadata
            self.frame_timestamp = time.time()
            
    def get_frame_data(self):
        """Get the most recent compressed frame data and metadata from buffer
        
        Returns:
            tuple: (frame_data, metadata) or (None, None) if no frame available
        """
        with self.lock:
            if self.buffer:
                buffer_entry = self.buffer[-1]
                return buffer_entry['frame_data'], buffer_entry.get('metadata')
            elif self.last_frame_data is not None:
                return self.last_frame_data, self.last_metadata
            else:
                return None, None
    
    def get_frame(self):
        """Get the most recent decompressed frame from buffer (for backward compatibility)"""
        frame_data, _ = self.get_frame_data()  # Ignore metadata for backward compatibility
        if frame_data is None:
            return None
            
        # Decode JPEG back to numpy array
        try:
            if frame_data['format'] == 'jpeg':
                # Decode JPEG data
                nparr = np.frombuffer(frame_data['data'], np.uint8)
                bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr_frame is not None:
                    # Convert BGR back to RGB
                    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            elif frame_data['format'] == 'raw':
                # Raw numpy data
                return np.frombuffer(frame_data['data'], dtype=np.uint8).reshape((-1, 750, 3))
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
        
        return None
                
    def get_frame_age(self):
        """Get the age of the most recent frame in seconds"""
        with self.lock:
            if self.frame_timestamp > 0:
                return time.time() - self.frame_timestamp
            else:
                return float('inf')
                
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.last_frame_data = None
            self.last_metadata = None
            self.frame_timestamp = 0

class MicroscopeVideoTrack(MediaStreamTrack):
    """
    A video stream track that provides real-time microscope images.
    """

    kind = "video"

    def __init__(self, microscope_instance):
        super().__init__()  # Initialize parent MediaStreamTrack
        self.microscope_instance = microscope_instance
        self.running = True
        self.fps = 5  # Default to 5 FPS
        self.count = 0
        self.start_time = None
        self.frame_width = 750
        self.frame_height = 750
        logger.info(f"MicroscopeVideoTrack initialized with FPS: {self.fps}")

    def draw_crosshair(self, img, center_x, center_y, size=20, color=[255, 255, 255]):
        """Draw a crosshair on the image"""
        import cv2
        # Draw horizontal line
        cv2.line(img, (center_x - size, center_y), (center_x + size, center_y), color, 2)
        # Draw vertical line
        cv2.line(img, (center_x, center_y - size), (center_x, center_y + size), color, 2)

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

            # Get compressed frame data WITH METADATA from microscope
            frame_response = await self.microscope_instance.get_video_frame(
                frame_width=self.frame_width,
                frame_height=self.frame_height
            )
            
            # Extract frame data and metadata
            if isinstance(frame_response, dict) and 'data' in frame_response:
                frame_data = frame_response
                frame_metadata = frame_response.get('metadata', {})
            else:
                # Fallback for backward compatibility
                frame_data = frame_response
                frame_metadata = {}
            
            # Decompress JPEG data to numpy array for WebRTC
            processed_frame = self.microscope_instance._decode_frame_jpeg(frame_data)

            current_time = time.time()
            # Use a 90kHz timebase, common for video, to provide accurate frame timing.
            # This prevents video from speeding up if frame acquisition is slow.
            time_base = fractions.Fraction(1, 90000)
            pts = int((current_time - self.start_time) * time_base.denominator)

            # Create VideoFrame
            new_video_frame = VideoFrame.from_ndarray(processed_frame, format="rgb24")
            new_video_frame.pts = pts
            new_video_frame.time_base = time_base

            # SEND METADATA VIA WEBRTC DATA CHANNEL
            # Send metadata through data channel instead of embedding in video frame
            if frame_metadata and hasattr(self.microscope_instance, 'metadata_data_channel'):
                try:
                    # Metadata already includes gray level statistics calculated in background acquisition
                    metadata_json = json.dumps(frame_metadata)
                    # Send metadata via WebRTC data channel
                    asyncio.create_task(self._send_metadata_via_datachannel(metadata_json))
                    logger.debug(f"Sent metadata via data channel: {len(metadata_json)} bytes (with gray level stats)")
                except Exception as e:
                    logger.warning(f"Failed to send metadata via data channel: {e}")
            
            if self.count % (self.fps * 5) == 0:  # Log every 5 seconds
                duration = current_time - self.start_time
                if duration > 0:
                    actual_fps = (self.count + 1) / duration
                    logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}, actual FPS: {actual_fps:.2f}")
                    if frame_metadata:
                        stage_pos = frame_metadata.get('stage_position', {})
                        x_mm = stage_pos.get('x_mm')
                        y_mm = stage_pos.get('y_mm')
                        z_mm = stage_pos.get('z_mm')
                        # Handle None values in position logging
                        x_str = f"{x_mm:.2f}" if x_mm is not None else "None"
                        y_str = f"{y_mm:.2f}" if y_mm is not None else "None"
                        z_str = f"{z_mm:.2f}" if z_mm is not None else "None"
                        logger.info(f"Frame metadata: stage=({x_str}, {y_str}, {z_str}), "
                                   f"channel={frame_metadata.get('channel')}, intensity={frame_metadata.get('intensity')}")
                else:
                    logger.info(f"MicroscopeVideoTrack: Sent frame {self.count}")
            
            self.count += 1
            return new_video_frame
            
        except Exception as e:
            logger.error(f"MicroscopeVideoTrack: Error in recv(): {e}", exc_info=True)
            self.running = False
            raise

    def update_fps(self, new_fps):
        """Update the FPS of the video track"""
        self.fps = new_fps
        logger.info(f"MicroscopeVideoTrack FPS updated to {new_fps}")

    async def _send_metadata_via_datachannel(self, metadata_json):
        """Send metadata via WebRTC data channel"""
        try:
            if hasattr(self.microscope_instance, 'metadata_data_channel') and self.microscope_instance.metadata_data_channel:
                if self.microscope_instance.metadata_data_channel.readyState == 'open':
                    self.microscope_instance.metadata_data_channel.send(metadata_json)
                    logger.debug(f"Metadata sent via data channel: {len(metadata_json)} bytes")
                else:
                    logger.debug(f"Data channel not ready, state: {self.microscope_instance.metadata_data_channel.readyState}")
        except Exception as e:
            logger.warning(f"Error sending metadata via data channel: {e}")

    def stop(self):
        logger.info("MicroscopeVideoTrack stop() called.")
        self.running = False
        # Mark WebRTC as disconnected
        self.microscope_instance.webrtc_connected = False

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
        self.channel_param_map = ChannelMapper.get_id_to_param_map()
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
        self.server_url = "http://192.168.2.1:9527" if is_local else "https://hypha.aicell.io/"
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
        self.metadata_data_channel = None  # WebRTC data channel for metadata

        # Video buffering attributes
        self.video_buffer = VideoBuffer(max_size=5)
        self.frame_acquisition_task = None
        self.frame_acquisition_running = False
        self.buffer_fps = 5  # Background frame acquisition FPS
        self.last_parameters_update = 0
        self.parameters_update_interval = 1.0  # Update parameters every 1 second
        
        # Adjustable frame size attributes - replaces hardcoded 750x750
        self.buffer_frame_width = 750  # Current buffer frame width
        self.buffer_frame_height = 750  # Current buffer frame height
        self.default_frame_width = 750  # Default frame size
        self.default_frame_height = 750
        
        # Auto-stop video buffering attributes
        self.last_video_request_time = None
        self.video_idle_timeout = 1  # Increase to 1 seconds to prevent rapid cycling
        self.video_idle_check_task = None
        self.webrtc_connected = False
        self.buffering_start_time = None
        self.min_buffering_duration = 1.0  # Minimum time to keep buffering active
        
        # Scanning control attributes
        self.scanning_in_progress = False  # Flag to prevent video buffering during scans

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
            "start_video_buffering": "not_started",
            "stop_video_buffering": "not_started",
            "get_current_well_location": "not_started",
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
        return "pong"
    
    async def is_service_healthy(self, context=None):
        """Check if all services are healthy"""
        try:
            microscope_svc = await self.server.get_service(self.service_id)
            if microscope_svc is None:
                raise RuntimeError("Microscope service not found")
            
            result = await microscope_svc.hello_world()
            if result != "Hello world":
                raise RuntimeError(f"Microscope service returned unexpected response: {result}")
            
            datastore_id = f'data-store-{"simu" if self.is_simulation else "real"}-{self.service_id}'
            datastore_svc = await self.server.get_service(datastore_id)
            if datastore_svc is None:
                raise RuntimeError("Datastore service not found")
            
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
            
            try:
                if self.similarity_search_svc is None:
                    raise RuntimeError("Similarity search service not found")
                
                result = await self.similarity_search_svc.hello_world()
                if result != "Hello world":
                    raise RuntimeError(f"Similarity search service returned unexpected response: {result}")
                logger.info("Similarity search service is healthy")
            except Exception as similarity_error:
                logger.error(f"Similarity search service health check failed: {str(similarity_error)}")
                raise RuntimeError(f"Similarity search service health check failed: {str(similarity_error)}")

            logger.info("All services are healthy")
            return {"status": "ok", "message": "All services are healthy"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Service health check failed: {str(e)}")
    
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
            # Get current well location information
            well_info = self.squidController.get_well_from_position('96')  # Default to 96-well plate
            
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
                'video_fps': self.buffer_fps,
                'video_buffering_active': self.frame_acquisition_running,
                'current_well_location': well_info,  # Add well location information
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
        
        # Stop video buffering to prevent camera overload
        if self.frame_acquisition_running:
            logger.info("Stopping video buffering for one_new_frame operation to prevent camera conflicts")
            await self.stop_video_buffering()
            # Wait a moment for the buffering to fully stop
            await asyncio.sleep(0.1)
        
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
            
            # Get the raw image from the camera with original bit depth preserved and full frame
            raw_img = await self.squidController.snap_image(channel, intensity, exposure_time, full_frame=True)
            
            # In simulation mode, resize small images to expected camera resolution
            if self.squidController.is_simulation:
                height, width = raw_img.shape[:2]
                # If image is too small, resize it to expected camera dimensions
                expected_width = 3000  # Expected camera width
                expected_height = 3000  # Expected camera height
                if width < expected_width or height < expected_height:
                    raw_img = cv2.resize(raw_img, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
            
            # Crop the image before resizing, similar to squid_controller.py approach
            crop_height = CONFIG.Acquisition.CROP_HEIGHT
            crop_width = CONFIG.Acquisition.CROP_WIDTH
            height, width = raw_img.shape[:2]  # Support both grayscale and color images
            start_x = width // 2 - crop_width // 2
            start_y = height // 2 - crop_height // 2
            
            # Ensure crop coordinates are within bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, start_x + crop_width)
            end_y = min(height, start_y + crop_height)
            
            cropped_img = raw_img[start_y:end_y, start_x:end_x]
            
            self.get_status()
            self.task_status[task_name] = "finished"
            
            # Return the numpy array directly with preserved bit depth
            return cropped_img
            
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to get new frame: {e}")
            raise e

    @schema_function(skip_self=True)
    async def get_video_frame(self, frame_width: int=Field(750, description="Width of the video frame"), frame_height: int=Field(750, description="Height of the video frame"), context=None):
        """
        Get compressed frame data with metadata from the microscope using video buffering
        Returns: Compressed frame data (JPEG bytes) with associated metadata including stage position and timestamp
        """
        try:
            # If scanning is in progress, return a scanning placeholder immediately
            if self.scanning_in_progress:
                logger.debug("Scanning in progress, returning scanning placeholder frame")
                placeholder = self._create_placeholder_frame(frame_width, frame_height, "Scanning in Progress...")
                placeholder_compressed = self._encode_frame_jpeg(placeholder, quality=85)
                
                # Create metadata for scanning placeholder frame
                scanning_metadata = {
                    'stage_position': {'x_mm': None, 'y_mm': None, 'z_mm': None},
                    'timestamp': time.time(),
                    'channel': None,
                    'intensity': None,
                    'exposure_time_ms': None,
                    'scanning_status': 'in_progress'
                }
                
                return {
                    'format': placeholder_compressed['format'],
                    'data': placeholder_compressed['data'],
                    'width': frame_width,
                    'height': frame_height,
                    'size_bytes': placeholder_compressed['size_bytes'],
                    'compression_ratio': placeholder_compressed.get('compression_ratio', 1.0),
                    'metadata': scanning_metadata
                }
            
            # Update last video request time for auto-stop functionality (only when not scanning)
            self.last_video_request_time = time.time()
            
            # Start video buffering if not already running and not scanning
            if not self.frame_acquisition_running:
                logger.info("Starting video buffering for remote video frame request")
                await self.start_video_buffering()
            
            # Start idle checking task if not running
            if self.video_idle_check_task is None or self.video_idle_check_task.done():
                self.video_idle_check_task = asyncio.create_task(self._monitor_video_idle())
            
            # Get compressed frame data and metadata from buffer
            frame_data, frame_metadata = self.video_buffer.get_frame_data()
            
            if frame_data is not None:
                # Check if we need to resize the frame
                # Use current buffer frame size instead of hardcoded values
                buffered_width = self.buffer_frame_width
                buffered_height = self.buffer_frame_height
                
                if frame_width != buffered_width or frame_height != buffered_height:
                    # Need to resize - decompress, resize, and recompress
                    decompressed_frame = self._decode_frame_jpeg(frame_data)
                    if decompressed_frame is not None:
                        # Resize the frame to requested dimensions
                        resized_frame = cv2.resize(decompressed_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                        # Recompress at requested size
                        resized_compressed = self._encode_frame_jpeg(resized_frame, quality=85)
                        return {
                            'format': resized_compressed['format'],
                            'data': resized_compressed['data'],
                            'width': frame_width,
                            'height': frame_height,
                            'size_bytes': resized_compressed['size_bytes'],
                            'compression_ratio': resized_compressed.get('compression_ratio', 1.0),
                            'metadata': frame_metadata
                        }
                    else:
                        # Fallback to placeholder if decompression fails
                        placeholder = self._create_placeholder_frame(frame_width, frame_height, "Frame decompression failed")
                        placeholder_compressed = self._encode_frame_jpeg(placeholder, quality=85)
                        return {
                            'format': placeholder_compressed['format'],
                            'data': placeholder_compressed['data'],
                            'width': frame_width,
                            'height': frame_height,
                            'size_bytes': placeholder_compressed['size_bytes'],
                            'compression_ratio': placeholder_compressed.get('compression_ratio', 1.0),
                            'metadata': frame_metadata
                        }
                else:
                    # Return buffered frame directly (no resize needed)
                    return {
                        'format': frame_data['format'],
                        'data': frame_data['data'],
                        'width': frame_width,
                        'height': frame_height,
                        'size_bytes': frame_data['size_bytes'],
                        'compression_ratio': frame_data.get('compression_ratio', 1.0),
                        'metadata': frame_metadata
                    }
            else:
                # No buffered frame available, create and compress placeholder
                logger.warning("No buffered frame available")
                placeholder = self._create_placeholder_frame(frame_width, frame_height, "No buffered frame available")
                placeholder_compressed = self._encode_frame_jpeg(placeholder, quality=85)
                
                # Create metadata for placeholder frame
                placeholder_metadata = {
                    'stage_position': {'x_mm': None, 'y_mm': None, 'z_mm': None},
                    'timestamp': time.time(),
                    'channel': None,
                    'intensity': None,
                    'exposure_time_ms': None,
                    'error': 'No buffered frame available'
                }
                
                return {
                    'format': placeholder_compressed['format'],
                    'data': placeholder_compressed['data'],
                    'width': frame_width,
                    'height': frame_height,
                    'size_bytes': placeholder_compressed['size_bytes'],
                    'compression_ratio': placeholder_compressed.get('compression_ratio', 1.0),
                    'metadata': placeholder_metadata
                }
                
        except Exception as e:
            logger.error(f"Error getting video frame: {e}", exc_info=True)
            # Create error placeholder and compress it
            raise e

    @schema_function(skip_self=True)
    def configure_video_buffer(self, buffer_fps: int = Field(5, description="Target FPS for buffer acquisition"), buffer_size: int = Field(5, description="Maximum number of frames to keep in buffer"), context=None):
        """Configure video buffering parameters for optimal streaming performance."""
        try:
            self.buffer_fps_target = max(1, min(30, buffer_fps))  # Clamp between 1-30 FPS
            
            # Update buffer size
            old_size = self.frame_buffer.maxlen
            self.frame_buffer = deque(maxlen=max(1, min(20, buffer_size)))  # Clamp between 1-20 frames
            
            logger.info(f"Video buffer configured: FPS={self.buffer_fps_target}, buffer_size={self.frame_buffer.maxlen} (was {old_size})")
            
            return {
                "success": True,
                "message": f"Video buffer configured with {self.buffer_fps_target} FPS target and {self.frame_buffer.maxlen} frame buffer size",
                "buffer_fps": self.buffer_fps_target,
                "buffer_size": self.frame_buffer.maxlen
            }
        except Exception as e:
            logger.error(f"Failed to configure video buffer: {e}")
            return {
                "success": False,
                "message": f"Failed to configure video buffer: {str(e)}"
            }

    @schema_function(skip_self=True)
    def get_video_buffer_status(self, context=None):
        """Get the current status of the video buffer."""
        try:
            buffer_fill = len(self.video_buffer.frame_buffer)
            buffer_capacity = self.video_buffer.max_size
            
            return {
                "success": True,
                "buffer_running": self.frame_acquisition_running,
                "buffer_fill": buffer_fill,
                "buffer_capacity": buffer_capacity,
                "buffer_fill_percent": (buffer_fill / buffer_capacity * 100) if buffer_capacity > 0 else 0,
                "buffer_fps": self.buffer_fps,
                "frame_dimensions": {
                    "width": self.buffer_frame_width,
                    "height": self.buffer_frame_height
                },
                "video_idle_timeout": self.video_idle_timeout,
                "last_video_request": self.last_video_request_time,
                "webrtc_connected": self.webrtc_connected
            }
        except Exception as e:
            logger.error(f"Failed to get video buffer status: {e}")
            return {
                "success": False,
                "message": f"Failed to get video buffer status: {str(e)}"
            }

    @schema_function(skip_self=True)
    async def start_video_buffering(self, context=None):
        """Manually start video buffering for smooth streaming."""
        try:
            if self.buffer_acquisition_running:
                return {
                    "success": True,
                    "message": "Video buffering is already running",
                    "was_already_running": True
                }
            
            await self.start_frame_buffer_acquisition()
            logger.info("Video buffering started manually")
            
            return {
                "success": True,
                "message": "Video buffering started successfully",
                "buffer_fps": self.buffer_fps_target,
                "buffer_size": self.frame_buffer.maxlen
            }
        except Exception as e:
            logger.error(f"Failed to start video buffering: {e}")
            return {
                "success": False,
                "message": f"Failed to start video buffering: {str(e)}"
            }

    @schema_function(skip_self=True)
    async def stop_video_buffering(self, context=None):
        """Stop the background frame acquisition task"""
        if not self.frame_acquisition_running:
            logger.info("Video buffering not running")
            return
            
        self.frame_acquisition_running = False
        
        # Stop idle monitoring task
        if self.video_idle_check_task and not self.video_idle_check_task.done():
            self.video_idle_check_task.cancel()
            try:
                await self.video_idle_check_task
            except asyncio.CancelledError:
                pass
            self.video_idle_check_task = None
        
        # Stop frame acquisition task
        if self.frame_acquisition_task:
            try:
                await asyncio.wait_for(self.frame_acquisition_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Frame acquisition task did not stop gracefully, cancelling")
                self.frame_acquisition_task.cancel()
                try:
                    await self.frame_acquisition_task
                except asyncio.CancelledError:
                    pass
        
        self.video_buffer.clear()
        self.last_video_request_time = None
        self.buffering_start_time = None
        logger.info("Video buffering stopped")
        
    @schema_function(skip_self=True)
    def configure_video_idle_timeout(self, idle_timeout: float = Field(5.0, description="Idle timeout in seconds (0 to disable automatic stop)"), context=None):
        """Configure how long to wait before automatically stopping video buffering when inactive."""
        try:
            self.video_idle_timeout = max(0, idle_timeout)  # Ensure non-negative
            logger.info(f"Video idle timeout set to {self.video_idle_timeout} seconds")
            
            return {
                "success": True,
                "message": f"Video idle timeout configured to {self.video_idle_timeout} seconds",
                "idle_timeout": self.video_idle_timeout,
                "automatic_stop": self.video_idle_timeout > 0
            }
        except Exception as e:
            logger.error(f"Failed to configure video idle timeout: {e}")
            return {
                "success": False,
                "message": f"Failed to configure video idle timeout: {str(e)}"
            }

    @schema_function(skip_self=True)
    async def set_video_fps(self, fps: int = Field(5, description="Target frames per second for video acquisition (1-30 FPS)"), context=None):
        """
        Set the video acquisition frame rate for smooth streaming.
        This controls how fast the microscope acquires frames for video streaming.
        Higher FPS provides smoother video but uses more resources.
        """
        task_name = "set_video_fps"
        self.task_status[task_name] = "started"
        
        try:
            # Validate FPS range
            if not isinstance(fps, int) or fps < 1 or fps > 30:
                return {
                    "success": False,
                    "message": f"Invalid FPS value: {fps}. Must be an integer between 1 and 30.",
                    "current_fps": self.buffer_fps
                }
            
            # Store old FPS for comparison
            old_fps = self.buffer_fps
            was_running = self.frame_acquisition_running
            
            # Update FPS setting
            self.buffer_fps = fps
            logger.info(f"Video FPS updated from {old_fps} to {fps}")
            
            # Update any active WebRTC video tracks with the new FPS
            if hasattr(self, 'video_track') and self.video_track is not None:
                self.video_track.update_fps(fps)
                logger.info("Updated WebRTC video track FPS")
            
            # If video buffering is currently running, restart it with new FPS
            if was_running:
                logger.info("Restarting video buffering with new FPS settings")
                await self.stop_video_buffering()
                # Brief pause to ensure clean shutdown
                await asyncio.sleep(0.2)
                await self.start_video_buffering()
                logger.info(f"Video buffering restarted with {fps} FPS")
            
            return {
                "success": True,
                "message": f"Video FPS successfully updated from {old_fps} to {fps} FPS",
                "old_fps": old_fps,
                "new_fps": fps,
                "buffering_restarted": was_running
            }
            
        except Exception as e:
            logger.error(f"Failed to set video FPS: {e}")
            return {
                "success": False,
                "message": f"Failed to set video FPS: {str(e)}",
                "current_fps": getattr(self, 'buffer_fps', 5)
            }



    def _reset_video_activity_tracking(self):
        """Reset video activity tracking (internal method)."""
        self.last_video_request_time = None
        logger.info("Video activity tracking reset")

    async def cleanup_for_tests(self):
        """Cleanup method specifically for test environments."""
        try:
            # Stop video buffering if running
            if self.buffer_acquisition_running:
                logger.info("Stopping video buffering for test cleanup")
                await self.stop_frame_buffer_acquisition()
            
            # Close camera resources properly
            if hasattr(self, 'squidController') and self.squidController:
                if hasattr(self.squidController, 'camera') and self.squidController.camera:
                    camera = self.squidController.camera
                    if hasattr(camera, 'cleanup_zarr_resources_async'):
                        try:
                            await asyncio.wait_for(camera.cleanup_zarr_resources_async(), timeout=5.0)
                            logger.info("ZarrImageManager resources cleaned up")
                        except asyncio.TimeoutError:
                            logger.warning("ZarrImageManager cleanup timed out")
                        except Exception as e:
                            logger.warning(f"ZarrImageManager cleanup error: {e}")
        except Exception as e:
            logger.error(f"Error during test cleanup: {e}")

    @schema_function(skip_self=True)
    async def start_video_buffering_api(self, context=None):
        """Start video buffering for smooth video streaming"""
        task_name = "start_video_buffering"
        self.task_status[task_name] = "started"
        try:
            await self.start_video_buffering()
            self.task_status[task_name] = "finished"
            return {"success": True, "message": "Video buffering started successfully"}
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to start video buffering: {e}")
            raise e

    @schema_function(skip_self=True)
    async def stop_video_buffering_api(self, context=None):
        """Manually stop video buffering to save resources."""
        task_name = "stop_video_buffering"
        self.task_status[task_name] = "started"
        try:
            if not self.frame_acquisition_running:
                self.task_status[task_name] = "finished"
                return {
                    "success": True,
                    "message": "Video buffering is already stopped",
                    "was_already_stopped": True
                }
            
            await self.stop_video_buffering()
            logger.info("Video buffering stopped manually")
            
            self.task_status[task_name] = "finished"
            return {
                "success": True,
                "message": "Video buffering stopped successfully"
            }
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to stop video buffering: {e}")
            return {
                "success": False,
                "message": f"Failed to stop video buffering: {str(e)}"
            }

    @schema_function(skip_self=True)
    def get_video_buffering_status(self, context=None):
        """Get the current video buffering status"""
        try:
            buffer_size = len(self.video_buffer.buffer) if self.video_buffer else 0
            frame_age = self.video_buffer.get_frame_age() if self.video_buffer else float('inf')
            
            return {
                "buffering_active": self.frame_acquisition_running,
                "buffer_size": buffer_size,
                "max_buffer_size": self.video_buffer.max_size if self.video_buffer else 0,
                "frame_age_seconds": frame_age if frame_age != float('inf') else None,
                "buffer_fps": self.buffer_fps,
                "has_frames": buffer_size > 0
            }
        except Exception as e:
            logger.error(f"Failed to get video buffering status: {e}")
            return {
                "buffering_active": False,
                "buffer_size": 0,
                "max_buffer_size": 0,
                "frame_age_seconds": None,
                "buffer_fps": 0,
                "has_frames": False,
                "error": str(e)
            }

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
            return {"success": False, "message": f"Failed to adjust video frame: {str(e)}"}

    @schema_function(skip_self=True)
    async def snap(self, exposure_time: int=Field(100, description="Exposure time, in milliseconds"), channel: int=Field(0, description="Light source (0 for Bright Field, Fluorescence channels: 11 for 405 nm, 12 for 488 nm, 13 for 638nm, 14 for 561 nm, 15 for 730 nm)"), intensity: int=Field(50, description="Intensity of the illumination source"), context=None):
        """
        Get an image from microscope
        Returns: the URL of the image
        """
        task_name = "snap"
        self.task_status[task_name] = "started"
        
        # Stop video buffering to prevent camera overload
        if self.frame_acquisition_running:
            logger.info("Stopping video buffering for snap operation to prevent camera conflicts")
            await self.stop_video_buffering()
            # Wait a moment for the buffering to fully stop
            await asyncio.sleep(0.1)
        
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
    async def scan_well_plate(self, well_plate_type: str=Field("96", description="Type of the well plate (e.g., '6', '12', '24', '96', '384')"), illumination_settings: List[dict]=Field(default_factory=lambda: [{'channel': 'BF LED matrix full', 'intensity': 28.0, 'exposure_time': 20.0}, {'channel': 'Fluorescence 488 nm Ex', 'intensity': 27.0, 'exposure_time': 60.0}, {'channel': 'Fluorescence 561 nm Ex', 'intensity': 98.0, 'exposure_time': 100.0}], description="Illumination settings with channel name, intensity (0-100), and exposure time (ms) for each channel"), do_contrast_autofocus: bool=Field(False, description="Whether to do contrast based autofocus"), do_reflection_af: bool=Field(True, description="Whether to do reflection based autofocus"), scanning_zone: List[tuple]=Field(default_factory=lambda: [(0,0),(0,0)], description="The scanning zone of the well plate, for 96 well plate, it should be[(0,0),(7,11)] "), Nx: int=Field(3, description="Number of columns to scan"), Ny: int=Field(3, description="Number of rows to scan"), action_ID: str=Field('testPlateScan', description="The ID of the action"), context=None):
        """
        Scan the well plate according to the pre-defined position list with custom illumination settings
        Returns: The message of the action
        """
        task_name = "scan_well_plate"
        self.task_status[task_name] = "started"
        try:
            if illumination_settings is None:
                logger.warning("No illumination settings provided, using default settings")
                illumination_settings = [
                    {'channel': 'BF LED matrix full', 'intensity': 18, 'exposure_time': 10},
                    {'channel': 'Fluorescence 405 nm Ex', 'intensity': 45, 'exposure_time': 30},
                    {'channel': 'Fluorescence 488 nm Ex', 'intensity': 30, 'exposure_time': 100},
                    {'channel': 'Fluorescence 561 nm Ex', 'intensity': 100, 'exposure_time': 200},
                    {'channel': 'Fluorescence 638 nm Ex', 'intensity': 100, 'exposure_time': 200},
                    {'channel': 'Fluorescence 730 nm Ex', 'intensity': 100, 'exposure_time': 200},
                ]
            
            # Check if video buffering is active and stop it during scanning
            video_buffering_was_active = self.frame_acquisition_running
            if video_buffering_was_active:
                logger.info("Video buffering is active, stopping it temporarily during well plate scanning")
                await self.stop_video_buffering()
                # Wait additional time to ensure camera fully settles after stopping video buffering
                logger.info("Waiting for camera to settle after stopping video buffering...")
                await asyncio.sleep(0.5)
            
            # Set scanning flag to prevent automatic video buffering restart during scan
            self.scanning_in_progress = True
            
            logger.info("Start scanning well plate with custom illumination settings")
            self.squidController.plate_scan(well_plate_type, illumination_settings, do_contrast_autofocus, do_reflection_af, scanning_zone, Nx, Ny, action_ID)
            logger.info("Well plate scanning completed")
            self.task_status[task_name] = "finished"
            return "Well plate scanning completed"
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to scan well plate: {e}")
            raise e
        finally:
            # Always reset the scanning flag, regardless of success or failure
            self.scanning_in_progress = False
            logger.info("Well plate scanning completed, video buffering auto-start is now re-enabled")
    
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
    async def auto_focus(self, context=None):
        """
        Do contrast-based autofocus
        Returns: A string message
        """
        task_name = "auto_focus"
        self.task_status[task_name] = "started"
        try:
            await self.squidController.do_autofocus()
            logger.info('The camera is auto-focused')
            self.task_status[task_name] = "finished"
            return 'The camera is auto-focused'
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to auto focus: {e}")
            raise e
    
    @schema_function(skip_self=True)
    async def do_laser_autofocus(self, context=None):
        """
        Do reflection-based autofocus
        Returns: A string message
        """
        task_name = "do_laser_autofocus"
        self.task_status[task_name] = "started"
        try:
            await self.squidController.do_laser_autofocus()
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

    class GetCurrentWellLocationInput(BaseModel):
        """Get the current well location based on the stage position."""
        wellplate_type: str = Field('96', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')")

    class GetMicroscopeConfigurationInput(BaseModel):
        """Get microscope configuration information in JSON format."""
        config_section: str = Field('all', description="Configuration section to retrieve ('all', 'camera', 'stage', 'illumination', 'acquisition', 'limits', 'hardware', 'wellplate', 'optics', 'autofocus')")
        include_defaults: bool = Field(True, description="Whether to include default values from config.py")

    class SetStageVelocityInput(BaseModel):
        """Set the maximum velocity for X and Y stage axes."""
        velocity_x_mm_per_s: Optional[float] = Field(None, description="Maximum velocity for X axis in mm/s (default: uses configuration value)")
        velocity_y_mm_per_s: Optional[float] = Field(None, description="Maximum velocity for Y axis in mm/s (default: uses configuration value)")

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
    
    async def auto_focus_schema(self, config: AutoFocusInput, context=None):
        await self.auto_focus(context)
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

    async def do_laser_autofocus_schema(self, context=None):
        response = await self.do_laser_autofocus(context)
        return {"result": response}

    def set_laser_reference_schema(self, context=None):
        response = self.set_laser_reference(context)
        return {"result": response}

    def get_status_schema(self, context=None):
        response = self.get_status(context)
        return {"result": response}

    def get_current_well_location_schema(self, config: GetCurrentWellLocationInput, context=None):
        response = self.get_current_well_location(config.wellplate_type, context)
        return {"result": response}

    def get_microscope_configuration_schema(self, config: GetMicroscopeConfigurationInput, context=None):
        response = self.get_microscope_configuration(config.config_section, config.include_defaults, context)
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
            "find_similar_image_image": self.FindSimilarImageImageInput.model_json_schema(),
            "get_current_well_location": self.GetCurrentWellLocationInput.model_json_schema(),
            "get_microscope_configuration": self.GetMicroscopeConfigurationInput.model_json_schema(),
            "set_stage_velocity": self.SetStageVelocityInput.model_json_schema(),
        }

    async def start_hypha_service(self, server, service_id, run_in_executor=None):
        self.server = server
        self.service_id = service_id
        
        # Default to True for production, False for tests (identified by "test" in service_id)
        if run_in_executor is None:
            run_in_executor = "test" not in service_id.lower()
        
        # Build the service configuration
        service_config = {
            "name": "Microscope Control Service",
            "id": service_id,
            "config": {
                "visibility": "public",
                "run_in_executor": run_in_executor
            },
            "type": "echo",
            "hello_world": self.hello_world,
            "is_service_healthy": self.is_service_healthy,
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
            "start_video_buffering": self.start_video_buffering_api,
            "stop_video_buffering": self.stop_video_buffering_api,
            "get_video_buffering_status": self.get_video_buffering_status,
            "set_video_fps": self.set_video_fps,
            "get_current_well_location": self.get_current_well_location,
            "get_microscope_configuration": self.get_microscope_configuration,
            "set_stage_velocity": self.set_stage_velocity,
            # Stitching functions
            "normal_scan_with_stitching": self.normal_scan_with_stitching,
            "quick_scan_with_stitching": self.quick_scan_with_stitching,
            "get_stitched_region": self.get_stitched_region,
            "reset_stitching_canvas": self.reset_stitching_canvas,
        }
        
        # Only register get_canvas_chunk when not in local mode
        if not self.is_local:
            service_config["get_canvas_chunk"] = self.get_canvas_chunk
            logger.info("Registered get_canvas_chunk service (remote mode)")
        else:
            logger.info("Skipped get_canvas_chunk service registration (local mode)")

        svc = await server.register_service(service_config)

        logger.info(
            f"Service (service_id={service_id}) started successfully, available at {self.server_url}{server.config.workspace}/services"
        )

        
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
                "find_similar_image_image": self.find_similar_image_image_schema,
                "get_current_well_location": self.get_current_well_location_schema,
                "get_microscope_configuration": self.get_microscope_configuration_schema,
                "set_stage_velocity": self.set_stage_velocity_schema,
            }
        }

        svc = await server.register_service(chatbot_extension)
        self.chatbot_service_url = f"https://bioimage.io/chat?server=https://chat.bioimage.io&extension={svc.id}&assistant=Skyler"
        logger.info(f"Extension service registered with id: {svc.id}, you can visit the service at:\n {self.chatbot_service_url}")

    async def start_webrtc_service(self, server, webrtc_service_id_arg):
        self.webrtc_service_id = webrtc_service_id_arg 
        
        async def on_init(peer_connection):
            logger.info("WebRTC peer connection initialized")
            # Mark as connected when peer connection starts
            self.webrtc_connected = True
            
            # Create data channel for metadata transmission
            self.metadata_data_channel = peer_connection.createDataChannel("metadata", ordered=True)
            logger.info("Created metadata data channel")
            
            @self.metadata_data_channel.on("open")
            def on_data_channel_open():
                logger.info("Metadata data channel opened")
            
            @self.metadata_data_channel.on("close")
            def on_data_channel_close():
                logger.info("Metadata data channel closed")
            
            @self.metadata_data_channel.on("error")
            def on_data_channel_error(error):
                logger.error(f"Metadata data channel error: {error}")
            
            @peer_connection.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"WebRTC connection state changed to: {peer_connection.connectionState}")
                if peer_connection.connectionState in ["closed", "failed", "disconnected"]:
                    # Mark as disconnected
                    self.webrtc_connected = False
                    self.metadata_data_channel = None
                    if self.video_track and self.video_track.running:
                        logger.info(f"Connection state is {peer_connection.connectionState}. Stopping video track.")
                        self.video_track.stop()
                elif peer_connection.connectionState in ["connected"]:
                    # Mark as connected
                    self.webrtc_connected = True
            
            @peer_connection.on("track")
            def on_track(track):
                logger.info(f"Track {track.kind} received from client")
                
                if self.video_track and self.video_track.running:
                    self.video_track.stop() 
                
                self.video_track = MicroscopeVideoTrack(self) 
                peer_connection.addTrack(self.video_track)
                logger.info("Added MicroscopeVideoTrack to peer connection")
                self.is_streaming = True
                
                # Start video buffering when WebRTC starts
                asyncio.create_task(self.start_video_buffering())
                
                @track.on("ended")
                def on_ended():
                    logger.info(f"Client track {track.kind} ended")
                    if self.video_track:
                        logger.info("Stopping MicroscopeVideoTrack.")
                        self.video_track.stop()  # Now synchronous
                        self.video_track = None
                    self.is_streaming = False
                    self.metadata_data_channel = None
                    
                    # Stop video buffering when WebRTC ends
                    asyncio.create_task(self.stop_video_buffering())

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

        # Determine workspace and token based on simulation mode
        if self.is_simulation and not self.is_local:
            remote_token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
            remote_workspace = "agent-lens"
        else:
            remote_token = os.environ.get("SQUID_WORKSPACE_TOKEN")
            remote_workspace = "squid-control"
            
        remote_server = await connect_to_server(
                {"server_url": "https://hypha.aicell.io", "token": remote_token, "workspace": remote_workspace, "ping_interval": None}
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
            # Determine workspace and token based on simulation mode
            if self.is_simulation:
                try:  
                    token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")  
                except:  
                    token = await login({"server_url": self.server_url})
                workspace = "agent-lens"
            else:
                try:  
                    token = os.environ.get("SQUID_WORKSPACE_TOKEN")  
                except:  
                    token = await login({"server_url": self.server_url})
                workspace = "squid-control"
            
            server = await connect_to_server(
                {"server_url": self.server_url, "token": token, "workspace": workspace,  "ping_interval": None}
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

    async def start_video_buffering(self):
        """Start the background frame acquisition task for video buffering"""
        if self.frame_acquisition_running:
            logger.info("Video buffering already running")
            return
            
        self.frame_acquisition_running = True
        self.buffering_start_time = time.time()
        self.frame_acquisition_task = asyncio.create_task(self._background_frame_acquisition())
        logger.info("Video buffering started")
        
    async def stop_video_buffering(self):
        """Stop the background frame acquisition task"""
        if not self.frame_acquisition_running:
            logger.info("Video buffering not running")
            return
            
        self.frame_acquisition_running = False
        
        # Stop idle monitoring task
        if self.video_idle_check_task and not self.video_idle_check_task.done():
            self.video_idle_check_task.cancel()
            try:
                await self.video_idle_check_task
            except asyncio.CancelledError:
                pass
            self.video_idle_check_task = None
        
        # Stop frame acquisition task
        if self.frame_acquisition_task:
            try:
                await asyncio.wait_for(self.frame_acquisition_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Frame acquisition task did not stop gracefully, cancelling")
                self.frame_acquisition_task.cancel()
                try:
                    await self.frame_acquisition_task
                except asyncio.CancelledError:
                    pass
        
        self.video_buffer.clear()
        self.last_video_request_time = None
        self.buffering_start_time = None
        logger.info("Video buffering stopped")
        
    async def _background_frame_acquisition(self):
        """Background task that continuously acquires frames and stores them in buffer"""
        logger.info("Background frame acquisition started")
        consecutive_failures = 0
        
        while self.frame_acquisition_running:
            try:
                # Control frame acquisition rate with adaptive timing
                start_time = time.time()
                
                # Reduce frequency if camera is struggling
                if consecutive_failures > 3:
                    current_fps = max(1, self.buffer_fps / 2)  # Halve the FPS if struggling
                    logger.warning(f"Camera struggling, reducing acquisition rate to {current_fps} FPS")
                else:
                    current_fps = self.buffer_fps
                
                # Get current parameters
                channel = self.squidController.current_channel
                param_name = self.channel_param_map.get(channel)
                intensity, exposure_time = 10, 10  # Default values
                
                if param_name:
                    stored_params = getattr(self, param_name, None)
                    if stored_params and isinstance(stored_params, list) and len(stored_params) == 2:
                        intensity, exposure_time = stored_params

                # Acquire frame
                try:
                    # LATENCY MEASUREMENT: Start timing background frame acquisition
                    T_cam_start = time.time()
                    
                    if self.is_simulation:
                        # Use existing simulation method for video buffering
                        raw_frame = await self.squidController.get_camera_frame_simulation(
                            channel, intensity, exposure_time
                        )
                    else:
                        # For real hardware, run in executor to avoid blocking
                        raw_frame = await asyncio.get_event_loop().run_in_executor(
                            None, self.squidController.get_camera_frame, channel, intensity, exposure_time
                        )
                    
                    # LATENCY MEASUREMENT: End timing background frame acquisition
                    T_cam_read_complete = time.time()
                    
                    # Calculate frame acquisition time and frame size (only if frame is valid)
                    if raw_frame is not None:
                        frame_acquisition_time_ms = (T_cam_read_complete - T_cam_start) * 1000
                        frame_size_bytes = raw_frame.nbytes
                        frame_size_kb = frame_size_bytes / 1024
                        
                        # Log timing and size information for latency analysis (less frequent to avoid spam)
                        if consecutive_failures == 0:  # Only log on successful acquisitions
                            logger.info(f"LATENCY_MEASUREMENT: Background frame acquisition took {frame_acquisition_time_ms:.2f}ms, "
                                       f"frame size: {frame_size_kb:.2f}KB, exposure_time: {exposure_time}ms, "
                                       f"channel: {channel}, intensity: {intensity}")
                    else:
                        frame_acquisition_time_ms = (T_cam_read_complete - T_cam_start) * 1000
                        logger.info(f"LATENCY_MEASUREMENT: Background frame acquisition failed after {frame_acquisition_time_ms:.2f}ms, "
                                   f"exposure_time: {exposure_time}ms, channel: {channel}, intensity: {intensity}")
                    
                    # Check if frame acquisition was successful
                    if raw_frame is None:
                        consecutive_failures += 1
                        logger.warning(f"Camera frame acquisition returned None - camera may be overloaded (failure #{consecutive_failures})")
                        # Create placeholder frame on None return
                        placeholder_frame = self._create_placeholder_frame(
                            self.buffer_frame_width, self.buffer_frame_height, "Camera Overloaded"
                        )
                        compressed_placeholder = self._encode_frame_jpeg(placeholder_frame, quality=85)
                        
                        # Calculate gray level statistics for placeholder frame
                        placeholder_gray_stats = self._calculate_gray_level_statistics(placeholder_frame)
                        
                        # Create placeholder metadata
                        placeholder_metadata = {
                            'stage_position': {'x_mm': None, 'y_mm': None, 'z_mm': None},
                            'timestamp': time.time(),
                            'channel': channel,
                            'intensity': intensity,
                            'exposure_time_ms': exposure_time,
                            'gray_level_stats': placeholder_gray_stats,
                            'error': 'Camera Overloaded'
                        }
                        self.video_buffer.put_frame(compressed_placeholder, placeholder_metadata)
                        
                        # If too many failures, wait longer before next attempt
                        if consecutive_failures >= 5:
                            await asyncio.sleep(2.0)  # Wait 2 seconds before retry
                            consecutive_failures = max(0, consecutive_failures - 2)  # Gradually recover
                            
                    else:
                        # Process frame normally and reset failure counter
                        consecutive_failures = 0
                        
                        # LATENCY MEASUREMENT: Start timing image processing
                        T_process_start = time.time()
                        
                        processed_frame, gray_level_stats = self._process_raw_frame(
                            raw_frame, frame_width=self.buffer_frame_width, frame_height=self.buffer_frame_height
                        )
                        
                        # LATENCY MEASUREMENT: End timing image processing
                        T_process_complete = time.time()
                        
                        # LATENCY MEASUREMENT: Start timing JPEG compression
                        T_compress_start = time.time()
                        
                        # Compress frame for efficient storage and transmission
                        compressed_frame = self._encode_frame_jpeg(processed_frame, quality=85)
                        
                        # LATENCY MEASUREMENT: End timing JPEG compression
                        T_compress_complete = time.time()
                        
                        # METADATA CAPTURE: Get current stage position and create metadata
                        frame_timestamp = time.time()
                        try:
                            # Update position and get current coordinates
                            self.squidController.navigationController.update_pos(microcontroller=self.squidController.microcontroller)
                            current_x = self.squidController.navigationController.x_pos_mm
                            current_y = self.squidController.navigationController.y_pos_mm
                            current_z = self.squidController.navigationController.z_pos_mm
                            print(f"current_x: {current_x}, current_y: {current_y}, current_z: {current_z}")
                            frame_metadata = {
                                'stage_position': {
                                    'x_mm': current_x,
                                    'y_mm': current_y,
                                    'z_mm': current_z
                                },
                                'timestamp': frame_timestamp,
                                'channel': channel,
                                'intensity': intensity,
                                'exposure_time_ms': exposure_time,
                                'gray_level_stats': gray_level_stats
                            }
                        except Exception as e:
                            logger.warning(f"Failed to capture stage position for metadata: {e}")
                            # Fallback metadata without stage position
                            frame_metadata = {
                                'stage_position': {
                                    'x_mm': None,
                                    'y_mm': None,
                                    'z_mm': None
                                },
                                'timestamp': frame_timestamp,
                                'channel': channel,
                                'intensity': intensity,
                                'exposure_time_ms': exposure_time,
                                'gray_level_stats': gray_level_stats
                            }
                        
                        # Calculate timing statistics
                        processing_time_ms = (T_process_complete - T_process_start) * 1000
                        compression_time_ms = (T_compress_complete - T_compress_start) * 1000
                        total_time_ms = (T_compress_complete - T_cam_start) * 1000
                        
                        # Log comprehensive performance statistics
                        logger.info(f"LATENCY_PROCESSING: Background frame processing took {processing_time_ms:.2f}ms, "
                                   f"compression took {compression_time_ms:.2f}ms, "
                                   f"total_time={total_time_ms:.2f}ms, "
                                   f"compression_ratio={compressed_frame['compression_ratio']:.1f}x, "
                                   f"size: {compressed_frame['original_size']//1024}KB -> {compressed_frame['size_bytes']//1024}KB")
                        
                        # Store compressed frame with metadata in buffer
                        self.video_buffer.put_frame(compressed_frame, frame_metadata)
                    
                except Exception as e:
                    consecutive_failures += 1
                    logger.error(f"Error in background frame acquisition: {e}")
                    # Create placeholder frame on error
                    placeholder_frame = self._create_placeholder_frame(
                        self.buffer_frame_width, self.buffer_frame_height, f"Acquisition Error: {str(e)}"
                    )
                    compressed_placeholder = self._encode_frame_jpeg(placeholder_frame, quality=85)
                    
                    # Calculate gray level statistics for placeholder frame
                    placeholder_gray_stats = self._calculate_gray_level_statistics(placeholder_frame)
                    
                    # Create placeholder metadata for error case
                    error_metadata = {
                        'stage_position': {'x_mm': None, 'y_mm': None, 'z_mm': None},
                        'timestamp': time.time(),
                        'channel': channel if 'channel' in locals() else 0,
                        'intensity': intensity if 'intensity' in locals() else 0,
                        'exposure_time_ms': exposure_time if 'exposure_time' in locals() else 0,
                        'gray_level_stats': placeholder_gray_stats,
                        'error': f"Acquisition Error: {str(e)}"
                    }
                    self.video_buffer.put_frame(compressed_placeholder, error_metadata)
                
                # Control frame rate with adaptive timing
                elapsed = time.time() - start_time
                sleep_time = max(0.1, (1.0 / current_fps) - elapsed)  # Minimum 100ms between attempts
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in background frame acquisition: {e}")
                await asyncio.sleep(1.0)  # Wait 1 second on unexpected error
                
        logger.info("Background frame acquisition stopped")
        
    def _process_raw_frame(self, raw_frame, frame_width=750, frame_height=750):
        """Process raw frame for video streaming - OPTIMIZED"""
        try:
            # OPTIMIZATION 1: Crop FIRST, then resize to reduce data for all subsequent operations
            crop_height = CONFIG.Acquisition.CROP_HEIGHT
            crop_width = CONFIG.Acquisition.CROP_WIDTH
            height, width = raw_frame.shape[:2]  # Support both grayscale and color images
            start_x = width // 2 - crop_width // 2
            start_y = height // 2 - crop_height // 2
            
            # Ensure crop coordinates are within bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, start_x + crop_width)
            end_y = min(height, start_y + crop_height)
            
            cropped_frame = raw_frame[start_y:end_y, start_x:end_x]
            
            # Now resize the cropped frame to target dimensions
            if cropped_frame.shape[:2] != (frame_height, frame_width):
                # Use INTER_AREA for downsampling (faster than INTER_LINEAR)
                processed_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            else:
                processed_frame = cropped_frame.copy()
            
            # Calculate gray level statistics on original frame BEFORE min/max adjustments
            gray_level_stats = self._calculate_gray_level_statistics(processed_frame)
            
            # OPTIMIZATION 2: Robust contrast adjustment (fixed)
            min_val = self.video_contrast_min
            max_val = self.video_contrast_max

            if max_val is None:
                if processed_frame.dtype == np.uint16:
                    max_val = 65535
                else:
                    max_val = 255
            
            # OPTIMIZATION 3: Improved contrast scaling with proper range handling
            if max_val > min_val:
                # Clip values to the specified range
                processed_frame = np.clip(processed_frame, min_val, max_val)
                
                # Scale to 0-255 range using float for precision, then convert to uint8
                if max_val > min_val:
                    # Use float32 for accurate scaling, then convert to uint8
                    processed_frame = ((processed_frame.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    # Edge case: min_val == max_val
                    processed_frame = np.full_like(processed_frame, 127, dtype=np.uint8)
            else:
                # Edge case: max_val <= min_val, return mid-gray
                height, width = processed_frame.shape[:2]
                processed_frame = np.full((height, width), 127, dtype=np.uint8)
            
            # Ensure we have uint8 output
            if processed_frame.dtype != np.uint8:
                processed_frame = processed_frame.astype(np.uint8)
            
            # OPTIMIZATION 4: Fast color space conversion
            if len(processed_frame.shape) == 2:
                # Direct array manipulation is faster than cv2.cvtColor for grayscale->RGB
                processed_frame = np.stack([processed_frame] * 3, axis=2)
            elif processed_frame.shape[2] == 1:
                processed_frame = np.repeat(processed_frame, 3, axis=2)
            
            return processed_frame, gray_level_stats
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            placeholder_frame = self._create_placeholder_frame(frame_width, frame_height, f"Processing Error: {str(e)}")
            placeholder_stats = self._calculate_gray_level_statistics(placeholder_frame)
            return placeholder_frame, placeholder_stats
            
    def _create_placeholder_frame(self, width, height, message="No Frame Available"):
        """Create a placeholder frame with error message"""
        placeholder_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, message, (10, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return placeholder_img
    
    def _decode_frame_jpeg(self, frame_data):
        """
        Decode compressed frame data back to numpy array
        
        Args:
            frame_data: dict from _encode_frame_jpeg() or get_video_frame()
        
        Returns:
            numpy array: RGB image data
        """
        try:
            if frame_data['format'] == 'jpeg':
                # Decode JPEG data
                nparr = np.frombuffer(frame_data['data'], np.uint8)
                bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr_frame is not None:
                    # Convert BGR back to RGB
                    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            elif frame_data['format'] == 'raw':
                # Raw numpy data
                height = frame_data.get('height', 750)
                width = frame_data.get('width', 750)
                return np.frombuffer(frame_data['data'], dtype=np.uint8).reshape((height, width, 3))
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
        
        # Return placeholder on error
        width = frame_data.get('width', self.buffer_frame_width)
        height = frame_data.get('height', self.buffer_frame_height)
        return self._create_placeholder_frame(width, height, "Decode Error")

    def _calculate_gray_level_statistics(self, rgb_frame):
        """Calculate comprehensive gray level statistics for microscope analysis"""
        try:
            import numpy as np
            
            # Convert RGB to grayscale for analysis (standard luminance formula)
            if len(rgb_frame.shape) == 3:
                # RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
                gray_frame = np.dot(rgb_frame[...,:3], [0.299, 0.587, 0.114])
            else:
                gray_frame = rgb_frame
            
            # Ensure we have a valid grayscale image
            if gray_frame.size == 0:
                return None
                
            # Convert to 0-100% range for analysis
            gray_normalized = (gray_frame / 255.0) * 100.0
            
            # Calculate comprehensive statistics
            stats = {
                'mean_percent': float(np.mean(gray_normalized)),
                'std_percent': float(np.std(gray_normalized)),
                'min_percent': float(np.min(gray_normalized)),
                'max_percent': float(np.max(gray_normalized)),
                'median_percent': float(np.median(gray_normalized)),
                'percentiles': {
                    'p5': float(np.percentile(gray_normalized, 5)),
                    'p25': float(np.percentile(gray_normalized, 25)),
                    'p75': float(np.percentile(gray_normalized, 75)),
                    'p95': float(np.percentile(gray_normalized, 95))
                },
                'histogram': {
                    'bins': 20,  # 20 bins for 0-100% range (5% per bin)
                    'counts': [],
                    'bin_edges': []
                }
            }
            
            # Calculate histogram (20 bins from 0-100%)
            hist_counts, bin_edges = np.histogram(gray_normalized, bins=20, range=(0, 100))
            stats['histogram']['counts'] = hist_counts.tolist()
            stats['histogram']['bin_edges'] = bin_edges.tolist()
            
            # Additional microscope-specific metrics
            stats['dynamic_range_percent'] = stats['max_percent'] - stats['min_percent']
            stats['contrast_ratio'] = stats['std_percent'] / stats['mean_percent'] if stats['mean_percent'] > 0 else 0
            
            # Exposure quality indicators
            stats['exposure_quality'] = {
                'underexposed_pixels_percent': float(np.sum(gray_normalized < 5) / gray_normalized.size * 100),
                'overexposed_pixels_percent': float(np.sum(gray_normalized > 95) / gray_normalized.size * 100),
                'well_exposed_pixels_percent': float(np.sum((gray_normalized >= 5) & (gray_normalized <= 95)) / gray_normalized.size * 100)
            }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error calculating gray level statistics: {e}")
            return None

    def _encode_frame_jpeg(self, frame, quality=85):
        """
        Encode frame to JPEG format for efficient network transmission
        
        Args:
            frame: RGB numpy array
            quality: JPEG quality (1-100, higher = better quality, larger size)
        
        Returns:
            dict: {
                'format': 'jpeg',
                'data': bytes,
                'size_bytes': int,
                'compression_ratio': float
            }
        """
        try:
            # Convert RGB to BGR for OpenCV JPEG encoding
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = frame
            
            # Encode to JPEG with specified quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, encoded_img = cv2.imencode('.jpg', bgr_frame, encode_params)
            
            if not success:
                raise ValueError("Failed to encode frame to JPEG")
            
            # Calculate compression statistics
            original_size = frame.nbytes
            compressed_size = len(encoded_img)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            return {
                'format': 'jpeg',
                'data': encoded_img.tobytes(),
                'size_bytes': compressed_size,
                'compression_ratio': compression_ratio,
                'original_size': original_size
            }
            
        except Exception as e:
            logger.error(f"Error encoding frame to JPEG: {e}")
            # Return uncompressed as fallback
            raise e

    async def _monitor_video_idle(self):
        """Monitor video request activity and stop buffering after idle timeout"""
        while self.frame_acquisition_running:
            try:
                await asyncio.sleep(1.0)  # Check every 1 second instead of 500ms
                
                # Don't stop video buffering during scanning
                if self.scanning_in_progress:
                    continue
                
                if self.last_video_request_time is None:
                    continue
                    
                # Check if we've been buffering for minimum duration
                if self.buffering_start_time is not None:
                    buffering_duration = time.time() - self.buffering_start_time
                    if buffering_duration < self.min_buffering_duration:
                        continue  # Don't stop yet, maintain minimum buffering time
                
                # Check if video has been idle too long
                idle_time = time.time() - self.last_video_request_time
                if idle_time > self.video_idle_timeout:
                    logger.info(f"Video idle for {idle_time:.1f}s (timeout: {self.video_idle_timeout}s), stopping buffering")
                    await self.stop_video_buffering()
                    break
            
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in video idle monitoring: {e}")
                await asyncio.sleep(2.0)  # Longer sleep on error
                
        logger.info("Video idle monitoring stopped")

    @schema_function(skip_self=True)
    def get_current_well_location(self, wellplate_type: str=Field('96', description="Type of the well plate (e.g., '6', '12', '24', '96', '384')"), context=None):
        """
        Get the current well location based on the stage position.
        Returns: Dictionary with well location information including row, column, well_id, and position status
        """
        task_name = "get_current_well_location"
        if task_name not in self.task_status:
            self.task_status[task_name] = "not_started"
        self.task_status[task_name] = "started"
        try:
            well_info = self.squidController.get_well_from_position(wellplate_type)
            logger.info(f'Current well location: {well_info["well_id"]} ({well_info["position_status"]})')
            self.task_status[task_name] = "finished"
            return well_info
        except Exception as e:
            self.task_status[task_name] = "failed"
            logger.error(f"Failed to get current well location: {e}")
            raise e

    @schema_function(skip_self=True)
    def configure_video_buffer_frame_size(self, frame_width: int = Field(750, description="Width of the video buffer frames"), frame_height: int = Field(750, description="Height of the video buffer frames"), context=None):
        """Configure video buffer frame size for optimal streaming performance."""
        try:
            # Validate frame size parameters
            frame_width = max(64, min(4096, frame_width))  # Clamp between 64-4096 pixels
            frame_height = max(64, min(4096, frame_height))  # Clamp between 64-4096 pixels
            
            old_width = self.buffer_frame_width
            old_height = self.buffer_frame_height
            
            # Update buffer frame size
            self.buffer_frame_width = frame_width
            self.buffer_frame_height = frame_height
            
            # If buffer is running and size changed, restart it to use new size
            restart_needed = (frame_width != old_width or frame_height != old_height) and self.frame_acquisition_running
            
            if restart_needed:
                logger.info(f"Buffer frame size changed from {old_width}x{old_height} to {frame_width}x{frame_height}, restarting buffer")
                # Clear existing buffer to remove old-sized frames
                self.video_buffer.clear()
                # Note: The frame acquisition loop will automatically use the new size for subsequent frames
            
            # Update WebRTC video track if it exists
            if hasattr(self, 'video_track') and self.video_track:
                self.video_track.frame_width = frame_width
                self.video_track.frame_height = frame_height
                logger.info(f"Updated WebRTC video track frame size to {frame_width}x{frame_height}")
            
            logger.info(f"Video buffer frame size configured: {frame_width}x{frame_height} (was {old_width}x{old_height})")
            
            return {
                "success": True,
                "message": f"Video buffer frame size configured to {frame_width}x{frame_height}",
                "previous_size": {"width": old_width, "height": old_height},
                "new_size": {"width": frame_width, "height": frame_height},
                "buffer_restarted": restart_needed
            }
        except Exception as e:
            logger.error(f"Failed to configure video buffer frame size: {e}")
            return {
                "success": False,
                "message": f"Failed to configure video buffer frame size: {str(e)}"
            }

    @schema_function(skip_self=True)
    def get_microscope_configuration(self, config_section: str = Field("all", description="Configuration section to retrieve ('all', 'camera', 'stage', 'illumination', 'acquisition', 'limits', 'hardware', 'wellplate', 'optics', 'autofocus')"), include_defaults: bool = Field(True, description="Whether to include default values from config.py"), context=None):
        """
        Get microscope configuration information in JSON format.
        Input: config_section: str = Field("all", description="Configuration section to retrieve ('all', 'camera', 'stage', 'illumination', 'acquisition', 'limits', 'hardware', 'wellplate', 'optics', 'autofocus')"), include_defaults: bool = Field(True, description="Whether to include default values from config.py")
        Returns: Configuration data as a JSON object
        """
        try:
            from squid_control.control.config import get_microscope_configuration_data
            
            # Call the configuration function from config.py
            result = get_microscope_configuration_data(
                config_section=config_section,
                include_defaults=include_defaults,
                is_simulation=self.is_simulation,
                is_local=self.is_local,
                squid_controller=self.squidController
            )
            
            logger.info(f"Retrieved microscope configuration for section: {config_section}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get microscope configuration: {e}")
            return {
                "success": False,
                "error": str(e),
                "section": config_section
            }

    @schema_function(skip_self=True)
    async def get_canvas_chunk(self, x_mm: float = Field(..., description="X coordinate of the stage location in millimeters"), y_mm: float = Field(..., description="Y coordinate of the stage location in millimeters"), scale_level: int = Field(1, description="Scale level for the chunk (0-2, where 0 is highest resolution)"), context=None):
        """Get a canvas chunk based on microscope stage location (available only in simulation mode when not running locally)"""
        
        # Check if this function is available in current mode
        if self.is_local:
            return {
                "success": False,
                "error": "get_canvas_chunk is not available in local mode"
            }
        
        if not self.is_simulation:
            return {
                "success": False,
                "error": "get_canvas_chunk is only available in simulation mode"
            }
        
        try:
            logger.info(f"Getting canvas chunk at position: x={x_mm}mm, y={y_mm}mm, scale_level={scale_level}")
            
            # Initialize ZarrImageManager if not already initialized
            if not hasattr(self, 'zarr_image_manager') or self.zarr_image_manager is None:
                from squid_control.hypha_tools.artifact_manager.artifact_manager import ZarrImageManager
                self.zarr_image_manager = ZarrImageManager()
                success = await self.zarr_image_manager.connect(server_url=self.server_url)
                if not success:
                    raise RuntimeError("Failed to connect to ZarrImageManager")
                logger.info("ZarrImageManager initialized for get_canvas_chunk")
            
            # Use the current simulated sample data alias
            dataset_id = self.get_simulated_sample_data_alias()
            channel_name = 'BF_LED_matrix_full'  # Always use brightfield channel
            
            # Use parameters similar to the simulation camera
            pixel_size_um = 0.333  # Default pixel size used in simulation
            
            # Get scale factor based on scale level
            scale_factors = {0: 1, 1: 4, 2: 16}  # scale0=1x, scale1=1/4x, scale2=1/16x
            scale_factor = scale_factors.get(scale_level, 4)  # Default to scale1
            
            # Convert microscope coordinates (mm) to pixel coordinates
            pixel_x = int((x_mm / pixel_size_um) * 1000 / scale_factor)
            pixel_y = int((y_mm / pixel_size_um) * 1000 / scale_factor)
            
            # Convert pixel coordinates to chunk coordinates
            chunk_size = 256  # Default chunk size used by ZarrImageManager
            chunk_x = pixel_x // chunk_size
            chunk_y = pixel_y // chunk_size
            
            logger.info(f"Converted coordinates: x={x_mm}mm, y={y_mm}mm to pixel coords: x={pixel_x}, y={pixel_y}, chunk coords: x={chunk_x}, y={chunk_y} (scale{scale_level})")
            
            # Get the single chunk data from ZarrImageManager
            region_data = await self.zarr_image_manager.get_region_np_data(
                dataset_id, 
                channel_name, 
                scale_level,
                chunk_x,  # Chunk X coordinate
                chunk_y,  # Chunk Y coordinate
                direct_region=None,  # Don't use direct_region, use chunk coordinates instead
                width=chunk_size,
                height=chunk_size
            )
            
            if region_data is None:
                return {
                    "success": False,
                    "error": "Failed to retrieve chunk data from Zarr storage"
                }
            
            # Convert numpy array to base64 encoded PNG for transmission
            try:
                # Ensure data is in uint8 format
                if region_data.dtype != np.uint8:
                    if region_data.dtype == np.float32 or region_data.dtype == np.float64:
                        # Normalize floating point data
                        if region_data.max() > 0:
                            region_data = (region_data / region_data.max() * 255).astype(np.uint8)
                        else:
                            region_data = np.zeros(region_data.shape, dtype=np.uint8)
                    else:
                        # For other integer types, scale appropriately
                        region_data = (region_data / region_data.max() * 255).astype(np.uint8) if region_data.max() > 0 else region_data.astype(np.uint8)
                        
                # Convert to PIL Image and then to base64
                pil_image = Image.fromarray(region_data)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return {
                    "data": img_base64,
                    "format": "png_base64",
                    "scale_level": scale_level,
                    "stage_location": {"x_mm": x_mm, "y_mm": y_mm},
                    "chunk_coordinates": {"chunk_x": chunk_x, "chunk_y": chunk_y}
                }
                
            except Exception as e:
                logger.error(f"Error converting chunk data to base64: {e}")
                return {
                    "success": False,
                    "error": f"Failed to convert chunk data: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error in get_canvas_chunk: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to get canvas chunk: {str(e)}"
            }

    @schema_function(skip_self=True)
    def set_stage_velocity(self, velocity_x_mm_per_s: Optional[float] = Field(None, description="Maximum velocity for X axis in mm/s (default: uses configuration value)"), velocity_y_mm_per_s: Optional[float] = Field(None, description="Maximum velocity for Y axis in mm/s (default: uses configuration value)"), context=None):
        """
        Set the maximum velocity for X and Y stage axes.
        
        This function allows you to control how fast the microscope stage moves.
        Lower velocities provide more precision but slower movement.
        Higher velocities enable faster navigation but may reduce precision.
        
        Args:
            velocity_x_mm_per_s: Maximum velocity for X axis in mm/s. If not specified, uses default from configuration.
            velocity_y_mm_per_s: Maximum velocity for Y axis in mm/s. If not specified, uses default from configuration.
            
        Returns:
            dict: Status and current velocity settings
        """
        logger.info(f"Setting stage velocity - X: {velocity_x_mm_per_s} mm/s, Y: {velocity_y_mm_per_s} mm/s")
        
        try:
            # Call the SquidController method
            result = self.squidController.set_stage_velocity(
                velocity_x_mm_per_s=velocity_x_mm_per_s,
                velocity_y_mm_per_s=velocity_y_mm_per_s
            )
            
            logger.info(f"Stage velocity set successfully: {result}")
            return result
            
        except ValueError as e:
            logger.error(f"Invalid velocity parameters: {e}")
            return {
                "status": "error",
                "message": f"Invalid velocity parameters: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error setting stage velocity: {e}")
            return {
                "status": "error", 
                "message": f"Failed to set stage velocity: {str(e)}"
            }

    def get_microscope_configuration_schema(self, config: GetMicroscopeConfigurationInput, context=None):
        return self.get_microscope_configuration(config.config_section, config.include_defaults, context)

    def set_stage_velocity_schema(self, config: SetStageVelocityInput, context=None):
        return self.set_stage_velocity(config.velocity_x_mm_per_s, config.velocity_y_mm_per_s, context)

    @schema_function(skip_self=True)
    async def normal_scan_with_stitching(self, start_x_mm: float = Field(20, description="Starting X position in millimeters"), 
                                       start_y_mm: float = Field(20, description="Starting Y position in millimeters"),
                                       Nx: int = Field(5, description="Number of positions in X direction"),
                                       Ny: int = Field(5, description="Number of positions in Y direction"),
                                       dx_mm: float = Field(0.9, description="Interval between positions in X (millimeters)"),
                                       dy_mm: float = Field(0.9, description="Interval between positions in Y (millimeters)"),
                                       illumination_settings: Optional[List[dict]] = Field(None, description="List of channel settings"),
                                       do_contrast_autofocus: bool = Field(False, description="Whether to perform contrast-based autofocus"),
                                       do_reflection_af: bool = Field(False, description="Whether to perform reflection-based autofocus"),
                                       action_ID: str = Field('normal_scan_stitching', description="Identifier for this scan"),
                                       context=None):
        """
        Perform a normal scan with live stitching to OME-Zarr canvas.
        The images are saved to a zarr file that represents a spatial map of the scanned area.
        
        Args:
            start_x_mm: Starting X position in millimeters
            start_y_mm: Starting Y position in millimeters
            Nx: Number of positions to scan in X direction
            Ny: Number of positions to scan in Y direction
            dx_mm: Distance between positions in X direction (millimeters)
            dy_mm: Distance between positions in Y direction (millimeters)
            illumination_settings: List of dictionaries with channel settings (optional)
            do_contrast_autofocus: Enable contrast-based autofocus
            do_reflection_af: Enable reflection-based autofocus
            action_ID: Unique identifier for this scan
            
        Returns:
            dict: Status of the scan
        """
        try:
            # Set default illumination settings if not provided
            if illumination_settings is None:
                illumination_settings = [{'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100}]
            
            logger.info(f"Starting normal scan with stitching: {Nx}x{Ny} positions from ({start_x_mm}, {start_y_mm})")
            
            # Check if video buffering is active and stop it during scanning
            video_buffering_was_active = self.frame_acquisition_running
            if video_buffering_was_active:
                logger.info("Video buffering is active, stopping it temporarily during scanning")
                await self.stop_video_buffering()
                # Wait additional time to ensure camera fully settles after stopping video buffering
                logger.info("Waiting for camera to settle after stopping video buffering...")
                await asyncio.sleep(0.5)
            
            # Set scanning flag to prevent automatic video buffering restart during scan
            self.scanning_in_progress = True
            
            # Ensure the squid controller has the new method
            await self.squidController.normal_scan_with_stitching(
                start_x_mm=start_x_mm,
                start_y_mm=start_y_mm,
                Nx=Nx,
                Ny=Ny,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                illumination_settings=illumination_settings,
                do_contrast_autofocus=do_contrast_autofocus,
                do_reflection_af=do_reflection_af,
                action_ID=action_ID
            )
            
            return {
                "success": True,
                "message": f"Normal scan with stitching completed successfully",
                "scan_parameters": {
                    "start_position": {"x_mm": start_x_mm, "y_mm": start_y_mm},
                    "grid_size": {"nx": Nx, "ny": Ny},
                    "step_size": {"dx_mm": dx_mm, "dy_mm": dy_mm},
                    "total_area_mm2": (Nx * dx_mm) * (Ny * dy_mm)
                }
            }
        except Exception as e:
            logger.error(f"Failed to perform normal scan with stitching: {e}")
            return {
                "success": False,
                "message": f"Failed to perform normal scan: {str(e)}"
            }
        finally:
            # Always reset the scanning flag, regardless of success or failure
            self.scanning_in_progress = False
            logger.info("Scanning completed, video buffering auto-start is now re-enabled")
    
    @schema_function(skip_self=True)
    def get_stitched_region(self, start_x_mm: float = Field(..., description="Starting X position in millimeters (top-left corner)"),
                           start_y_mm: float = Field(..., description="Starting Y position in millimeters (top-left corner)"),
                           width_mm: float = Field(5.0, description="Width of region in millimeters"),
                           height_mm: float = Field(5.0, description="Height of region in millimeters"),
                           scale_level: int = Field(0, description="Scale level (0=full resolution, 1=1/4, 2=1/16, etc)"),
                           channel_name: str = Field('BF LED matrix full', description="Name of channel to retrieve"),
                           output_format: str = Field('base64', description="Output format: 'base64' or 'array'"),
                           context=None):
        """
        Get a region from the stitched canvas.
        
        This function retrieves a rectangular region from the stitched microscope image canvas.
        The region is specified by its starting position (top-left corner) and dimensions in millimeters.
        
        Args:
            start_x_mm: X coordinate of region starting position (top-left) in millimeters
            start_y_mm: Y coordinate of region starting position (top-left) in millimeters
            width_mm: Width of the region in millimeters
            height_mm: Height of the region in millimeters
            scale_level: Pyramid level to retrieve (0=highest resolution)
            channel_name: Name of the channel to retrieve
            output_format: Format for the output ('base64' for compressed image, 'array' for numpy array)
            
        Returns:
            dict: Retrieved image data with metadata
        """
        try:
            # Calculate center coordinates for the underlying function
            center_x_mm = start_x_mm + width_mm / 2
            center_y_mm = start_y_mm + height_mm / 2
            
            # Get the region from the zarr canvas
            region = self.squidController.get_stitched_region(
                center_x_mm=center_x_mm,
                center_y_mm=center_y_mm,
                width_mm=width_mm,
                height_mm=height_mm,
                scale_level=scale_level,
                channel_name=channel_name
            )
            
            if output_format == 'base64':
                # Convert to base64 encoded PNG
                import base64
                from PIL import Image
                import io
                
                # Ensure data is uint8
                if region.dtype != np.uint8:
                    region = (region / region.max() * 255).astype(np.uint8) if region.max() > 0 else region.astype(np.uint8)
                
                # Convert to PIL Image and encode
                img = Image.fromarray(region)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return {
                    "success": True,
                    "data": img_base64,
                    "format": "png_base64",
                    "shape": region.shape,
                    "dtype": str(region.dtype),
                    "region": {
                        "start_x_mm": start_x_mm,
                        "start_y_mm": start_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "scale_level": scale_level,
                        "channel": channel_name
                    }
                }
            else:
                # Return as array (will be serialized by Hypha)
                return {
                    "success": True,
                    "data": region.tolist(),  # Convert to list for JSON serialization
                    "format": "array",
                    "shape": region.shape,
                    "dtype": str(region.dtype),
                    "region": {
                        "start_x_mm": start_x_mm,
                        "start_y_mm": start_y_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                        "scale_level": scale_level,
                        "channel": channel_name
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get stitched region: {e}")
            return {
                "success": False,
                "message": f"Failed to retrieve stitched region: {str(e)}"
            }
    
    @schema_function(skip_self=True)
    def reset_stitching_canvas(self, context=None):
        """
        Reset the stitching canvas, clearing all stored images.
        
        This will delete the existing zarr canvas and prepare for a new scan.
        
        Returns:
            dict: Status of the reset operation
        """
        try:
            if hasattr(self.squidController, 'zarr_canvas') and self.squidController.zarr_canvas is not None:
                # Close the existing canvas
                self.squidController.zarr_canvas.close()
                
                # Delete the zarr directory
                import shutil
                if self.squidController.zarr_canvas.zarr_path.exists():
                    shutil.rmtree(self.squidController.zarr_canvas.zarr_path)
                
                # Clear the reference
                self.squidController.zarr_canvas = None
                
                logger.info("Stitching canvas reset successfully")
                return {
                    "success": True,
                    "message": "Stitching canvas has been reset"
                }
            else:
                return {
                    "success": True,
                    "message": "No stitching canvas to reset"
                }
        except Exception as e:
            logger.error(f"Failed to reset stitching canvas: {e}")
            return {
                "success": False,
                "message": f"Failed to reset canvas: {str(e)}"
            }

    @schema_function(skip_self=True)
    async def quick_scan_with_stitching(self, wellplate_type: str = Field('96', description="Well plate type ('6', '12', '24', '96', '384')"),
                                      exposure_time: float = Field(5, description="Camera exposure time in milliseconds (max 30ms)"),
                                      intensity: float = Field(70, description="Brightfield LED intensity (0-100)"),
                                      velocity_mm_per_s: float = Field(10, description="Stage velocity in mm/s for scanning"),
                                      fps_target: int = Field(20, description="Target frame rate for acquisition"),
                                      action_ID: str = Field('quick_scan_stitching', description="Identifier for this scan"),
                                      context=None):
        """
        Perform a quick scan with live stitching to OME-Zarr canvas - brightfield only.
        Uses continuous movement with high-speed frame acquisition for rapid well plate scanning.
        Only supports brightfield channel with exposure time ≤ 30ms.
        
        Args:
            wellplate_type: Well plate format ('6', '12', '24', '96', '384')
            exposure_time: Camera exposure time in milliseconds (must be ≤ 30ms)
            intensity: Brightfield LED intensity (0-100)
            velocity_mm_per_s: Stage velocity in mm/s for scanning (default 20mm/s)
            fps_target: Target frame rate for acquisition (default 20fps)
            action_ID: Unique identifier for this scan
            
        Returns:
            dict: Status of the scan with performance metrics
        """
        try:
            # Validate exposure time early
            if exposure_time > 30:
                return {
                    "success": False,
                    "message": f"Quick scan exposure time must not exceed 30ms (got {exposure_time}ms)"
                }
            
            logger.info(f"Starting quick scan with stitching: {wellplate_type} well plate, velocity={velocity_mm_per_s}mm/s, fps={fps_target}")
            
            # Check if video buffering is active and stop it during scanning
            video_buffering_was_active = self.frame_acquisition_running
            if video_buffering_was_active:
                logger.info("Video buffering is active, stopping it temporarily during quick scanning")
                await self.stop_video_buffering()
                # Wait for camera to settle after stopping video buffering
                logger.info("Waiting for camera to settle after stopping video buffering...")
                await asyncio.sleep(0.5)
            
            # Set scanning flag to prevent automatic video buffering restart during scan
            self.scanning_in_progress = True
            
            # Record start time for performance metrics
            start_time = time.time()
            
            # Perform the quick scan
            await self.squidController.quick_scan_with_stitching(
                wellplate_type=wellplate_type,
                exposure_time=exposure_time,
                intensity=intensity,
                velocity_mm_per_s=velocity_mm_per_s,
                fps_target=fps_target,
                action_ID=action_ID
            )
            
            # Calculate performance metrics
            scan_duration = time.time() - start_time
            
            # Calculate well plate dimensions for area estimation
            wellplate_configs = {
                '6': {'rows': 2, 'cols': 3},
                '12': {'rows': 3, 'cols': 4},
                '24': {'rows': 4, 'cols': 6},
                '96': {'rows': 8, 'cols': 12},
                '384': {'rows': 16, 'cols': 24}
            }
            
            config = wellplate_configs.get(wellplate_type, wellplate_configs['96'])
            
            return {
                "success": True,
                "message": f"Quick scan with stitching completed successfully",
                "scan_parameters": {
                    "wellplate_type": wellplate_type,
                    "rows_scanned": config['rows'],
                    "columns_per_row": config['cols'],
                    "exposure_time_ms": exposure_time,
                    "intensity": intensity,
                    "velocity_mm_per_s": velocity_mm_per_s,
                    "target_fps": fps_target
                },
                "performance_metrics": {
                    "total_scan_time_seconds": round(scan_duration, 2),
                    "scan_time_per_row_seconds": round(scan_duration / config['rows'], 2),
                    "estimated_frames_acquired": int(scan_duration * fps_target)
                },
                "stitching_info": {
                    "zarr_scales_updated": "1-5 (scale 0 skipped for performance)",
                    "channel": "BF LED matrix full",
                    "action_id": action_ID
                }
            }
            
        except ValueError as e:
            logger.error(f"Validation error in quick scan: {e}")
            return {
                "success": False,
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Failed to perform quick scan with stitching: {e}")
            return {
                "success": False,
                "message": f"Failed to perform quick scan: {str(e)}"
            }
        finally:
            # Always reset the scanning flag, regardless of success or failure
            self.scanning_in_progress = False
            logger.info("Quick scanning completed, video buffering auto-start is now re-enabled")

# Define a signal handler for graceful shutdown
def signal_handler(sig, frame):
    logger.info('Signal received, shutting down gracefully...')
    
    # Stop video buffering
    if hasattr(microscope, 'frame_acquisition_running') and microscope.frame_acquisition_running:
        logger.info('Stopping video buffering...')
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(microscope.stop_video_buffering())
            else:
                loop.run_until_complete(microscope.stop_video_buffering())
        except Exception as e:
            logger.error(f'Error stopping video buffering: {e}')
    
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