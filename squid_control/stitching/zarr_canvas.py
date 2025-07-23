import os
import numpy as np
import zarr
import logging
import threading
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from PIL import Image
from datetime import datetime
import cv2
import tempfile
import shutil
import zipfile

# Get the logger for this module
logger = logging.getLogger(__name__)

# Ensure the logger has the same level as the root logger
# This ensures our INFO messages are actually displayed
if not logger.handlers:
    # If no handlers are set up, inherit from the root logger
    logger.setLevel(logging.INFO)
    # Add a handler that matches the main service format
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent double logging

class WellZarrCanvasBase:
    """
    Base class for well-specific zarr canvas functionality.
    Contains the core stitching and zarr management functionality without single-canvas assumptions.
    """
    
    def __init__(self, base_path: str, pixel_size_xy_um: float, stage_limits: Dict[str, float], 
                 channels: List[str] = None, chunk_size: int = 256, rotation_angle_deg: float = 0.0,
                 initial_timepoints: int = 20, timepoint_expansion_chunk: int = 10, fileset_name: str = "live_stitching",
                 initialize_new: bool = False):
        """
        Initialize the Zarr canvas.
        
        Args:
            base_path: Base directory for zarr storage (from ZARR_PATH env variable)
            pixel_size_xy_um: Pixel size in micrometers
            stage_limits: Dictionary with x_positive, x_negative, y_positive, y_negative in mm
            channels: List of channel names (human-readable names)
            chunk_size: Size of chunks in pixels (default 256)
            rotation_angle_deg: Rotation angle for stitching in degrees (positive=clockwise, negative=counterclockwise)
            initial_timepoints: Number of timepoints to pre-allocate during initialization (default 20)
            timepoint_expansion_chunk: Number of timepoints to add when expansion is needed (default 10)
            fileset_name: Name of the zarr fileset (default 'live_stitching')
            initialize_new: If True, create a new fileset (deletes existing). If False, open existing if present.
        """
        self.base_path = Path(base_path)
        self.pixel_size_xy_um = pixel_size_xy_um
        self.stage_limits = stage_limits
        self.channels = channels or ['BF LED matrix full']
        self.chunk_size = chunk_size
        self.rotation_angle_deg = rotation_angle_deg
        self.fileset_name = fileset_name
        self.zarr_path = self.base_path / f"{fileset_name}.zarr"
        
        # Timepoint allocation strategy
        self.initial_timepoints = max(1, initial_timepoints)  # Ensure at least 1
        self.timepoint_expansion_chunk = max(1, timepoint_expansion_chunk)  # Ensure at least 1
        
        # Create channel mapping: channel_name -> local_zarr_index
        self.channel_to_zarr_index = {channel_name: idx for idx, channel_name in enumerate(self.channels)}
        self.zarr_index_to_channel = {idx: channel_name for idx, channel_name in enumerate(self.channels)}
        
        logger.info(f"Channel mapping: {self.channel_to_zarr_index}")
        
        # Calculate canvas dimensions in pixels based on stage limits
        self.stage_width_mm = stage_limits['x_positive'] - stage_limits['x_negative']
        self.stage_height_mm = stage_limits['y_positive'] - stage_limits['y_negative']
        
        # Convert to pixels (with some padding)
        padding_factor = 1.1  # 10% padding
        self.canvas_width_px = int((self.stage_width_mm * 1000 / pixel_size_xy_um) * padding_factor)
        self.canvas_height_px = int((self.stage_height_mm * 1000 / pixel_size_xy_um) * padding_factor)
        
        # Make dimensions divisible by chunk_size
        self.canvas_width_px = ((self.canvas_width_px + chunk_size - 1) // chunk_size) * chunk_size
        self.canvas_height_px = ((self.canvas_height_px + chunk_size - 1) // chunk_size) * chunk_size
        
        # Number of pyramid levels (scale0 is full res, scale1 is 1/4, scale2 is 1/16, etc)
        self.num_scales = self._calculate_num_scales()
        
        # Thread pool for async zarr operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Lock for thread-safe zarr access
        self.zarr_lock = threading.RLock()
        
        # Queue for frame stitching - increased size for stable FPS with non-blocking puts
        self.stitch_queue = asyncio.Queue(maxsize=500)
        self.stitching_task = None
        self.is_stitching = False
        
        # Track available timepoints
        self.available_timepoints = [0]  # Start with timepoint 0 as a list
        
        # Only initialize or open
        if initialize_new or not self.zarr_path.exists():
            self.initialize_canvas()
        else:
            self.open_existing_canvas()
        
        logger.info(f"ZarrCanvas initialized: {self.canvas_width_px}x{self.canvas_height_px} px, "
                    f"{self.num_scales} scales, chunk_size={chunk_size}, "
                    f"initial_timepoints={self.initial_timepoints}, expansion_chunk={self.timepoint_expansion_chunk}")
    
    def _calculate_num_scales(self) -> int:
        """Calculate the number of pyramid levels needed."""
        min_size = 64  # Minimum size for lowest resolution
        num_scales = 1
        width, height = self.canvas_width_px, self.canvas_height_px
        
        while width > min_size and height > min_size:
            width //= 4
            height //= 4
            num_scales += 1
            
        return min(num_scales, 6)  # Cap at 6 levels
    
    def get_zarr_channel_index(self, channel_name: str) -> int:
        """
        Get the local zarr array index for a channel name.
        
        Args:
            channel_name: Human-readable channel name
            
        Returns:
            int: Local index in the zarr array (0, 1, 2, etc.)
            
        Raises:
            ValueError: If channel name is not found
        """
        if channel_name not in self.channel_to_zarr_index:
            raise ValueError(f"Channel '{channel_name}' not found in zarr canvas. Available channels: {list(self.channel_to_zarr_index.keys())}")
        return self.channel_to_zarr_index[channel_name]
    
    def get_channel_name_by_zarr_index(self, zarr_index: int) -> str:
        """
        Get the channel name for a local zarr array index.
        
        Args:
            zarr_index: Local index in the zarr array
            
        Returns:
            str: Human-readable channel name
            
        Raises:
            ValueError: If zarr index is not found
        """
        if zarr_index not in self.zarr_index_to_channel:
            raise ValueError(f"Zarr index {zarr_index} not found. Available indices: {list(self.zarr_index_to_channel.keys())}")
        return self.zarr_index_to_channel[zarr_index]
    
    def get_available_timepoints(self) -> List[int]:
        """
        Get a list of available timepoints in the zarr array.
        
        Returns:
            List[int]: Sorted list of available timepoint indices
        """
        with self.zarr_lock:
            return sorted(self.available_timepoints)
    
    def create_timepoint(self, timepoint: int):
        """
        Create a new timepoint in the zarr array.
        This is now a lightweight operation that just adds to the available list.
        Zarr array expansion happens lazily when actually writing data.
        
        Args:
            timepoint: The timepoint index to create
            
        Raises:
            ValueError: If timepoint already exists or is negative
        """
        if timepoint < 0:
            raise ValueError(f"Timepoint must be non-negative, got {timepoint}")
            
        with self.zarr_lock:
            if timepoint in self.available_timepoints:
                logger.info(f"Timepoint {timepoint} already exists")
                return
            
            logger.info(f"Creating new timepoint {timepoint} (lightweight)")
            
            # Simply add to available timepoints - zarr arrays will be expanded when needed
            self.available_timepoints.append(timepoint)
            self.available_timepoints.sort()  # Keep sorted for consistency
            
            # Update metadata
            self._update_timepoint_metadata()
    
    def pre_allocate_timepoints(self, max_timepoint: int):
        """
        Pre-allocate zarr arrays to accommodate timepoints up to max_timepoint.
        This is useful for time-lapse experiments where you know the number of timepoints in advance.
        Performing this operation early avoids delays during scanning.
        
        Args:
            max_timepoint: The maximum timepoint index to pre-allocate for
            
        Raises:
            ValueError: If max_timepoint is negative
        """
        if max_timepoint < 0:
            raise ValueError(f"Max timepoint must be non-negative, got {max_timepoint}")
        
        with self.zarr_lock:
            logger.info(f"Pre-allocating zarr arrays for timepoints up to {max_timepoint}")
            start_time = time.time()
            
            # Check if any arrays need expansion
            expansion_needed = False
            for scale in range(self.num_scales):
                if scale in self.zarr_arrays:
                    zarr_array = self.zarr_arrays[scale]
                    if max_timepoint >= zarr_array.shape[0]:
                        expansion_needed = True
                        break
            
            if not expansion_needed:
                logger.info(f"Zarr arrays already accommodate timepoint {max_timepoint}")
                return
                
            # Expand all arrays to accommodate max_timepoint
            self._ensure_timepoint_exists_in_zarr(max_timepoint)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Pre-allocation completed in {elapsed_time:.2f} seconds")
    
    def remove_timepoint(self, timepoint: int):
        """
        Remove a timepoint from the zarr array by deleting its chunk files.
        
        Args:
            timepoint: The timepoint index to remove
            
        Raises:
            ValueError: If timepoint doesn't exist or is the last remaining timepoint
        """
        with self.zarr_lock:
            if timepoint not in self.available_timepoints:
                raise ValueError(f"Timepoint {timepoint} does not exist")
            
            if len(self.available_timepoints) == 1:
                raise ValueError("Cannot remove the last timepoint")
            
            logger.info(f"Removing timepoint {timepoint} and deleting chunk files")
            
            # Delete chunk files for this timepoint
            self._delete_timepoint_chunks(timepoint)
            
            # Remove from available timepoints list
            self.available_timepoints.remove(timepoint)
            
            # Update metadata
            self._update_timepoint_metadata()
    
    def clear_timepoint(self, timepoint: int):
        """
        Clear all data from a specific timepoint by deleting its chunk files.
        
        Args:
            timepoint: The timepoint index to clear
            
        Raises:
            ValueError: If timepoint doesn't exist
        """
        with self.zarr_lock:
            if timepoint not in self.available_timepoints:
                raise ValueError(f"Timepoint {timepoint} does not exist")
            
            logger.info(f"Clearing data from timepoint {timepoint} by deleting chunk files")
            
            # Delete chunk files for this timepoint
            self._delete_timepoint_chunks(timepoint)
    
    def _delete_timepoint_chunks(self, timepoint: int):
        """
        Delete all chunk files for a specific timepoint across all scales.
        This is much more efficient than zeroing out data.
        
        Args:
            timepoint: The timepoint index to delete chunks for
        """
        try:
            # For each scale, find and delete chunk files containing this timepoint
            for scale in range(self.num_scales):
                scale_path = self.zarr_path / str(scale)
                if not scale_path.exists():
                    continue
                
                # Zarr stores chunks in directories, timepoint is the first dimension
                # Chunk filename format: "t.c.z.y.x" where t is timepoint
                deleted_count = 0
                
                try:
                    # Look for chunk files that start with this timepoint
                    for chunk_file in scale_path.iterdir():
                        if chunk_file.is_file() and chunk_file.name.startswith(f"{timepoint}."):
                            try:
                                chunk_file.unlink()  # Delete the file
                                deleted_count += 1
                            except OSError as e:
                                logger.warning(f"Could not delete chunk file {chunk_file}: {e}")
                
                except OSError as e:
                    logger.warning(f"Could not access scale directory {scale_path}: {e}")
                
                if deleted_count > 0:
                    logger.debug(f"Deleted {deleted_count} chunk files for timepoint {timepoint} at scale {scale}")
                    
        except Exception as e:
            logger.error(f"Error deleting timepoint chunks: {e}")
    
    def _ensure_timepoint_exists_in_zarr(self, timepoint: int):
        """
        Ensure that the zarr arrays are large enough to accommodate the given timepoint.
        This is called lazily only when actually writing data.
        Expands arrays in chunks to minimize expensive resize operations.
        
        Args:
            timepoint: The timepoint index that needs to exist in zarr
        """
        # Check if we need to expand any zarr arrays
        for scale in range(self.num_scales):
            if scale in self.zarr_arrays:
                zarr_array = self.zarr_arrays[scale]
                current_shape = zarr_array.shape
                
                # If the timepoint is beyond current array size, resize in chunks
                if timepoint >= current_shape[0]:
                    # Calculate new size with expansion chunk strategy
                    # Round up to the next chunk boundary to minimize future resizes
                    required_size = timepoint + 1
                    chunks_needed = (required_size + self.timepoint_expansion_chunk - 1) // self.timepoint_expansion_chunk
                    new_timepoint_count = chunks_needed * self.timepoint_expansion_chunk
                    
                    new_shape = list(current_shape)
                    new_shape[0] = new_timepoint_count
                    
                    # Resize the array with chunk-based expansion
                    logger.info(f"Expanding zarr scale {scale} from {current_shape[0]} to {new_timepoint_count} timepoints "
                               f"(required: {required_size}, chunk_size: {self.timepoint_expansion_chunk})")
                    start_time = time.time()
                    zarr_array.resize(new_shape)
                    elapsed_time = time.time() - start_time
                    logger.info(f"Zarr scale {scale} resize completed in {elapsed_time:.2f} seconds")
    
    def _update_timepoint_metadata(self):
        """Update the OME-Zarr metadata to reflect current timepoints."""
        if hasattr(self, 'zarr_root'):
            root = self.zarr_root
            if 'omero' in root.attrs:
                if self.available_timepoints:
                    root.attrs['omero']['rdefs']['defaultT'] = min(self.available_timepoints)
            
            # Update custom metadata
            if 'squid_canvas' in root.attrs:
                root.attrs['squid_canvas']['available_timepoints'] = sorted(self.available_timepoints)
                root.attrs['squid_canvas']['num_timepoints'] = len(self.available_timepoints)
    
    def initialize_canvas(self):
        """Initialize the OME-Zarr structure with proper metadata."""
        logger.info(f"Initializing OME-Zarr canvas at {self.zarr_path}")
        
        try:
            # Ensure the parent directory exists
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Remove existing zarr if it exists and is corrupted
            if self.zarr_path.exists():
                import shutil
                shutil.rmtree(self.zarr_path)
                logger.info(f"Removed existing zarr directory: {self.zarr_path}")
            
            # Create the root group
            store = zarr.DirectoryStore(str(self.zarr_path))
            root = zarr.open_group(store=store, mode='w')
            self.zarr_root = root  # Store reference for metadata updates
            
            # Import ChannelMapper for better metadata
            from squid_control.control.config import ChannelMapper
            
            # Create enhanced channel metadata with proper colors and info
            omero_channels = []
            for ch in self.channels:
                try:
                    channel_info = ChannelMapper.get_channel_by_human_name(ch)
                    # Assign colors based on channel type
                    if channel_info.channel_id == 0:  # BF
                        color = "FFFFFF"  # White
                    elif channel_info.channel_id == 11:  # 405nm
                        color = "8000FF"  # Blue-violet
                    elif channel_info.channel_id == 12:  # 488nm
                        color = "00FF00"  # Green
                    elif channel_info.channel_id == 13:  # 638nm
                        color = "FF0000"  # Red
                    elif channel_info.channel_id == 14:  # 561nm
                        color = "FFFF00"  # Yellow
                    elif channel_info.channel_id == 15:  # 730nm
                        color = "FF00FF"  # Magenta
                    else:
                        color = "FFFFFF"  # Default white
                        
                    omero_channels.append({
                        "label": ch,
                        "color": color,
                        "active": True,
                        "window": {"start": 0, "end": 255},
                        "family": "linear",
                        "coefficient": 1.0
                    })
                except ValueError:
                    # Fallback for unknown channels
                    omero_channels.append({
                        "label": ch,
                        "color": "FFFFFF",
                        "active": True,
                        "window": {"start": 0, "end": 255},
                        "family": "linear",
                        "coefficient": 1.0
                    })
            
            # Create OME-Zarr metadata
            multiscales_metadata = {
                "multiscales": [{
                    "axes": [
                        {"name": "t", "type": "time", "unit": "second"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"}
                    ],
                    "datasets": [],
                    "name": self.fileset_name,
                    "version": "0.4"
                }],
                "omero": {
                    "id": 1,
                    "name": f"Squid Microscope Live Stitching ({self.fileset_name})",
                    "channels": omero_channels,
                    "rdefs": {
                        "defaultT": 0,
                        "defaultZ": 0,
                        "model": "color"
                    }
                },
                "squid_canvas": {
                    "channel_mapping": self.channel_to_zarr_index,
                    "zarr_index_mapping": self.zarr_index_to_channel,
                    "rotation_angle_deg": self.rotation_angle_deg,
                    "pixel_size_xy_um": self.pixel_size_xy_um,
                    "stage_limits": self.stage_limits,
                    "available_timepoints": sorted(self.available_timepoints),
                    "num_timepoints": len(self.available_timepoints),
                    "version": "1.0",
                    "fileset_name": self.fileset_name
                }
            }
            
            # Create arrays for each scale level
            for scale in range(self.num_scales):
                scale_factor = 4 ** scale
                width = self.canvas_width_px // scale_factor
                height = self.canvas_height_px // scale_factor
                
                # Create the array (T, C, Z, Y, X)
                # Pre-allocate initial timepoints to avoid frequent resizing
                # Use no compression for direct access and fastest performance
                array = root.create_dataset(
                    str(scale),
                    shape=(self.initial_timepoints, len(self.channels), 1, height, width),
                    chunks=(1, 1, 1, self.chunk_size, self.chunk_size),
                    dtype='uint8',
                    fill_value=0,
                    overwrite=True,
                    compressor=None  # No compression for raw data access
                )
                
                # Add scale metadata
                scale_transform = self.pixel_size_xy_um * scale_factor
                dataset_meta = {
                    "path": str(scale),
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0, scale_transform, scale_transform]
                    }]
                }
                multiscales_metadata["multiscales"][0]["datasets"].append(dataset_meta)
            
            # Write metadata
            root.attrs.update(multiscales_metadata)
            
            # Store references to arrays
            self.zarr_arrays = {}
            for scale in range(self.num_scales):
                self.zarr_arrays[scale] = root[str(scale)]
                
            logger.info(f"OME-Zarr canvas initialized successfully with {self.num_scales} scales")
            
        except Exception as e:
            logger.error(f"Failed to initialize OME-Zarr canvas: {e}")
            raise RuntimeError(f"Cannot initialize zarr canvas: {e}")
    
    def open_existing_canvas(self):
        """Open an existing OME-Zarr structure from disk without deleting data."""
        import zarr
        store = zarr.DirectoryStore(str(self.zarr_path))
        root = zarr.open_group(store=store, mode='r+')
        self.zarr_root = root
        # Load arrays for each scale
        self.zarr_arrays = {}
        for scale in range(self.num_scales):
            if str(scale) in root:
                self.zarr_arrays[scale] = root[str(scale)]
        # Try to load available timepoints from metadata
        if 'squid_canvas' in root.attrs and 'available_timepoints' in root.attrs['squid_canvas']:
            self.available_timepoints = list(root.attrs['squid_canvas']['available_timepoints'])
        else:
            self.available_timepoints = [0]
        logger.info(f"Opened existing Zarr canvas at {self.zarr_path}")
    
    def stage_to_pixel_coords(self, x_mm: float, y_mm: float, scale: int = 0) -> Tuple[int, int]:
        """
        Convert stage coordinates (mm) to pixel coordinates for a given scale.
        
        Args:
            x_mm: X position in millimeters
            y_mm: Y position in millimeters  
            scale: Scale level (0 = full resolution)
            
        Returns:
            Tuple of (x_pixel, y_pixel) coordinates
        """
        # Offset to make all coordinates positive
        x_offset_mm = -self.stage_limits['x_negative']
        y_offset_mm = -self.stage_limits['y_negative']
        
        # Convert to pixels at scale 0 (without padding)
        x_px_no_padding = (x_mm + x_offset_mm) * 1000 / self.pixel_size_xy_um
        y_px_no_padding = (y_mm + y_offset_mm) * 1000 / self.pixel_size_xy_um
        
        # Account for 10% padding by centering in the padded canvas
        # The canvas is 1.1x larger, so we need to add 5% margin on each side
        padding_factor = 1.1
        x_padding_px = (self.canvas_width_px - (self.stage_width_mm * 1000 / self.pixel_size_xy_um)) / 2
        y_padding_px = (self.canvas_height_px - (self.stage_height_mm * 1000 / self.pixel_size_xy_um)) / 2
        
        # Add padding offset to center the image in the padded canvas
        x_px = int(x_px_no_padding + x_padding_px)
        y_px = int(y_px_no_padding + y_padding_px)
        
        # Apply scale factor
        scale_factor = 4 ** scale
        x_px //= scale_factor
        y_px //= scale_factor
        
        return x_px, y_px
    
    def _rotate_and_crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate an image by the configured angle and crop to 95% of the original size.
        
        Args:
            image: Input image array (2D)
            
        Returns:
            Rotated and cropped image array
        """
        if abs(self.rotation_angle_deg) < 0.001:  # No rotation needed
            return image
            
        height, width = image.shape[:2]
        
        # Calculate rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle_deg, 1.0)
        
        # Perform rotation, positive angle means counterclockwise rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Crop to 95% of original size to remove black borders
        crop_factor = 0.96
        image_size = min(int(height * crop_factor), int(width * crop_factor))
        
        # Calculate crop bounds (center crop)
        y_start = (height - image_size) // 2
        y_end = y_start + image_size
        x_start = (width - image_size) // 2
        x_end = x_start + image_size
        
        cropped = rotated[y_start:y_end, x_start:x_end]
        
        logger.debug(f"Rotated image by {self.rotation_angle_deg}° and cropped from {width}x{height} to {image_size}x{image_size}")
        
        return cropped
    
    def add_image_sync(self, image: np.ndarray, x_mm: float, y_mm: float, 
                       channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0):
        """
        Synchronously add an image to the canvas at the specified position and timepoint.
        Updates all pyramid levels.
        
        Args:
            image: Image array (2D)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
        """
        # Validate channel index
        if channel_idx >= len(self.channels):
            logger.error(f"Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return
        
        if channel_idx < 0:
            logger.error(f"Channel index {channel_idx} cannot be negative")
            return
        
        # Ensure timepoint exists in our tracking list
        if timepoint not in self.available_timepoints:
            with self.zarr_lock:
                if timepoint not in self.available_timepoints:
                    self.available_timepoints.append(timepoint)
                    self.available_timepoints.sort()
                    self._update_timepoint_metadata()
        
        # Apply rotation and cropping first
        processed_image = self._rotate_and_crop_image(image)
        
        with self.zarr_lock:
            # Ensure zarr arrays are sized correctly for this timepoint (lazy expansion)
            self._ensure_timepoint_exists_in_zarr(timepoint)
            
            for scale in range(self.num_scales):
                scale_factor = 4 ** scale
                
                # Get pixel coordinates for this scale
                x_px, y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)
                
                # Resize image if needed
                if scale > 0:
                    new_size = (processed_image.shape[1] // scale_factor, processed_image.shape[0] // scale_factor)
                    scaled_image = cv2.resize(processed_image, new_size, interpolation=cv2.INTER_AREA)
                else:
                    scaled_image = processed_image
                
                # Get the zarr array for this scale
                zarr_array = self.zarr_arrays[scale]
                
                # Double-check zarr array dimensions
                if channel_idx >= zarr_array.shape[1]:
                    logger.error(f"Channel index {channel_idx} exceeds zarr array channel dimension {zarr_array.shape[1]}")
                    continue
                
                # Calculate bounds
                y_start = max(0, y_px - scaled_image.shape[0] // 2)
                y_end = min(zarr_array.shape[3], y_start + scaled_image.shape[0])
                x_start = max(0, x_px - scaled_image.shape[1] // 2)
                x_end = min(zarr_array.shape[4], x_start + scaled_image.shape[1])
                
                # Crop image if it extends beyond canvas
                img_y_start = max(0, -y_px + scaled_image.shape[0] // 2)
                img_y_end = img_y_start + (y_end - y_start)
                img_x_start = max(0, -x_px + scaled_image.shape[1] // 2)
                img_x_end = img_x_start + (x_end - x_start)
                
                # CRITICAL: Always validate bounds before writing to zarr arrays
                # This prevents zero-size chunk creation and zarr write errors
                logger.info(f"ZARR_WRITE: Scale {scale} bounds check - zarr_y({y_start}:{y_end}), zarr_x({x_start}:{x_end}), img_y({img_y_start}:{img_y_end}), img_x({img_x_start}:{img_x_end})")
                
                if y_end > y_start and x_end > x_start and img_y_end > img_y_start and img_x_end > img_x_start:
                    # Additional validation to ensure image slice is within bounds
                    img_y_end = min(img_y_end, scaled_image.shape[0])
                    img_x_end = min(img_x_end, scaled_image.shape[1])
                    
                    logger.info(f"ZARR_WRITE: Scale {scale} after clamping - img_y({img_y_start}:{img_y_end}), img_x({img_x_start}:{img_x_end}), scaled_image.shape={scaled_image.shape}")
                    
                    # Final check that we still have valid bounds after clamping
                    if img_y_end > img_y_start and img_x_end > img_x_start:
                        try:
                            logger.info(f"ZARR_WRITE: Attempting to write to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}")
                            # Ensure image is uint8 before writing to zarr
                            image_to_write = scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]
                            logger.info(f"ZARR_WRITE: Original image_to_write dtype: {image_to_write.dtype}, shape: {image_to_write.shape}, min: {image_to_write.min()}, max: {image_to_write.max()}")
                            
                            if image_to_write.dtype != np.uint8:
                                # Convert to uint8 if needed
                                if image_to_write.dtype == np.uint16:
                                    image_to_write = (image_to_write / 256).astype(np.uint8)
                                    logger.info(f"ZARR_WRITE: Converted uint16 to uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                                elif image_to_write.dtype in [np.float32, np.float64]:
                                    # Normalize float data to 0-255
                                    if image_to_write.max() > image_to_write.min():
                                        image_to_write = ((image_to_write - image_to_write.min()) / 
                                                        (image_to_write.max() - image_to_write.min()) * 255).astype(np.uint8)
                                        logger.info(f"ZARR_WRITE: Normalized float to uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                                    else:
                                        image_to_write = np.zeros_like(image_to_write, dtype=np.uint8)
                                        logger.info(f"ZARR_WRITE: Created zero uint8 array")
                                else:
                                    image_to_write = image_to_write.astype(np.uint8)
                                    logger.info(f"ZARR_WRITE: Direct conversion to uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                                logger.info(f"ZARR_WRITE: Converted image from {scaled_image.dtype} to uint8")
                            else:
                                logger.info(f"ZARR_WRITE: Image already uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                            
                            # Double-check the final data type
                            if image_to_write.dtype != np.uint8:
                                logger.error(f"ZARR_WRITE: CRITICAL ERROR - image_to_write is still {image_to_write.dtype}, not uint8!")
                                # Force conversion as fallback
                                image_to_write = image_to_write.astype(np.uint8)
                                logger.info(f"ZARR_WRITE: Forced conversion to uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                            
                            zarr_array[timepoint, channel_idx, z_idx, y_start:y_end, x_start:x_end] = image_to_write
                            logger.info(f"ZARR_WRITE: Successfully wrote image to zarr at scale {scale}, channel {channel_idx}, timepoint {timepoint}")
                        except IndexError as e:
                            logger.error(f"ZARR_WRITE: IndexError writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                            logger.error(f"ZARR_WRITE: Zarr array shape: {zarr_array.shape}, trying to access timepoint {timepoint}")
                        except Exception as e:
                            logger.error(f"ZARR_WRITE: Error writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                    else:
                        logger.warning(f"ZARR_WRITE: Skipping zarr write - invalid image bounds after clamping: img_y({img_y_start}:{img_y_end}), img_x({img_x_start}:{img_x_end})")
                else:
                    logger.warning(f"ZARR_WRITE: Skipping zarr write - invalid bounds: zarr_y({y_start}:{y_end}), zarr_x({x_start}:{x_end}), img_y({img_y_start}:{img_y_end}), img_x({img_x_start}:{img_x_end})")
    
    def add_image_sync_quick(self, image: np.ndarray, x_mm: float, y_mm: float, 
                           channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0):
        """
        Synchronously add an image to the canvas for quick scan mode.
        Only updates scales 1-5 (skips scale 0 for performance).
        The input image should already be at scale1 resolution.
        
        Args:
            image: Image array (2D) - should be at scale1 resolution (1/4 of original)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
        """
        logger.info(f"QUICK_SYNC: Called add_image_sync_quick at ({x_mm:.2f}, {y_mm:.2f}), channel={channel_idx}, timepoint={timepoint}, image.shape={image.shape}")
        
        # Validate channel index
        if channel_idx >= len(self.channels):
            logger.error(f"QUICK_SYNC: Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return
        
        if channel_idx < 0:
            logger.error(f"QUICK_SYNC: Channel index {channel_idx} cannot be negative")
            return
        
        # Ensure timepoint exists in our tracking list
        if timepoint not in self.available_timepoints:
            with self.zarr_lock:
                if timepoint not in self.available_timepoints:
                    self.available_timepoints.append(timepoint)
                    self.available_timepoints.sort()
                    self._update_timepoint_metadata()
        
        # For quick scan, we skip rotation to reduce computation pressure
        # The image should already be rotated and flipped by the caller
        processed_image = image
        
        with self.zarr_lock:
            # Ensure zarr arrays are sized correctly for this timepoint (lazy expansion)
            self._ensure_timepoint_exists_in_zarr(timepoint)
            
            logger.info(f"QUICK_SYNC: Starting zarr write operations for timepoint {timepoint}, processing scales 1-{min(self.num_scales, 6)-1}")
            
            # Only process scales 1-5 (skip scale 0 for performance)
            for scale in range(1, min(self.num_scales, 6)):  # scales 1-5
                logger.info(f"QUICK_SYNC: Processing scale {scale}")
                scale_factor = 4 ** scale
                
                # Get pixel coordinates for this scale
                x_px, y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)
                
                # Resize image - note that input image is already at scale1 resolution
                # So for scale1: use image as-is
                # For scale2: resize by 1/4, scale3: by 1/16, etc.
                if scale == 1:
                    scaled_image = processed_image  # Already at scale1 resolution
                else:
                    # Scale relative to scale1 (which the input image represents)
                    relative_scale_factor = 4 ** (scale - 1)
                    new_size = (processed_image.shape[1] // relative_scale_factor, 
                               processed_image.shape[0] // relative_scale_factor)
                    scaled_image = cv2.resize(processed_image, new_size, interpolation=cv2.INTER_AREA)
                
                # Get the zarr array for this scale
                zarr_array = self.zarr_arrays[scale]
                
                # Double-check zarr array dimensions
                if channel_idx >= zarr_array.shape[1]:
                    logger.error(f"Channel index {channel_idx} exceeds zarr array channel dimension {zarr_array.shape[1]}")
                    continue
                
                # Calculate bounds
                y_start = max(0, y_px - scaled_image.shape[0] // 2)
                y_end = min(zarr_array.shape[3], y_start + scaled_image.shape[0])
                x_start = max(0, x_px - scaled_image.shape[1] // 2)
                x_end = min(zarr_array.shape[4], x_start + scaled_image.shape[1])
                
                logger.info(f"QUICK_SYNC: Scale {scale} calculated bounds - y_px={y_px}, x_px={x_px}, scaled_image.shape={scaled_image.shape}")
                
                # Crop image if it extends beyond canvas
                img_y_start = max(0, -y_px + scaled_image.shape[0] // 2)
                img_y_end = img_y_start + (y_end - y_start)
                img_x_start = max(0, -x_px + scaled_image.shape[1] // 2)
                img_x_end = img_x_start + (x_end - x_start)
                
                logger.info(f"QUICK_SYNC: Scale {scale} bounds check - zarr_y({y_start}:{y_end}), zarr_x({x_start}:{x_end}), img_y({img_y_start}:{img_y_end}), img_x({img_x_start}:{img_x_end})")
                
                # CRITICAL: Always validate bounds before writing to zarr arrays
                # This prevents zero-size chunk creation and zarr write errors
                if y_end > y_start and x_end > x_start and img_y_end > img_y_start and img_x_end > img_x_start:
                    # Additional validation to ensure image slice is within bounds
                    img_y_end = min(img_y_end, scaled_image.shape[0])
                    img_x_end = min(img_x_end, scaled_image.shape[1])
                    
                    logger.info(f"QUICK_SYNC: Scale {scale} after clamping - img_y({img_y_start}:{img_y_end}), img_x({img_x_start}:{img_x_end}), scaled_image.shape={scaled_image.shape}")
                    
                    # Final check that we still have valid bounds after clamping
                    if img_y_end > img_y_start and img_x_end > img_x_start:
                        try:
                            logger.info(f"QUICK_SYNC: Attempting to write to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}")
                            # Ensure image is uint8 before writing to zarr
                            image_to_write = scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]
                            logger.info(f"QUICK_SYNC: Original image_to_write dtype: {image_to_write.dtype}, shape: {image_to_write.shape}, min: {image_to_write.min()}, max: {image_to_write.max()}")
                            
                            if image_to_write.dtype != np.uint8:
                                # Convert to uint8 if needed
                                if image_to_write.dtype == np.uint16:
                                    image_to_write = (image_to_write / 256).astype(np.uint8)
                                    logger.info(f"QUICK_SYNC: Converted uint16 to uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                                elif image_to_write.dtype in [np.float32, np.float64]:
                                    # Normalize float data to 0-255
                                    if image_to_write.max() > image_to_write.min():
                                        image_to_write = ((image_to_write - image_to_write.min()) / 
                                                        (image_to_write.max() - image_to_write.min()) * 255).astype(np.uint8)
                                        logger.info(f"QUICK_SYNC: Normalized float to uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                                    else:
                                        image_to_write = np.zeros_like(image_to_write, dtype=np.uint8)
                                        logger.info(f"QUICK_SYNC: Created zero uint8 array")
                                else:
                                    image_to_write = image_to_write.astype(np.uint8)
                                    logger.info(f"QUICK_SYNC: Direct conversion to uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                                logger.info(f"QUICK_SYNC: Converted image from {scaled_image.dtype} to uint8")
                            else:
                                logger.info(f"QUICK_SYNC: Image already uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                            
                            # Double-check the final data type
                            if image_to_write.dtype != np.uint8:
                                logger.error(f"QUICK_SYNC: CRITICAL ERROR - image_to_write is still {image_to_write.dtype}, not uint8!")
                                # Force conversion as fallback
                                image_to_write = image_to_write.astype(np.uint8)
                                logger.info(f"QUICK_SYNC: Forced conversion to uint8: min={image_to_write.min()}, max={image_to_write.max()}")
                            
                            zarr_array[timepoint, channel_idx, z_idx, y_start:y_end, x_start:x_end] = image_to_write
                            logger.info(f"QUICK_SYNC: Successfully wrote image to zarr at scale {scale}, channel {channel_idx}, timepoint {timepoint} (quick scan)")
                        except IndexError as e:
                            logger.error(f"QUICK_SYNC: IndexError writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                            logger.error(f"QUICK_SYNC: Zarr array shape: {zarr_array.shape}, trying to access timepoint {timepoint}")
                        except Exception as e:
                            logger.error(f"QUICK_SYNC: Error writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                    else:
                        logger.warning(f"QUICK_SYNC: Skipping zarr write - invalid image bounds after clamping: img_y({img_y_start}:{img_y_end}), img_x({img_x_start}:{img_x_end}) (quick scan)")
                else:
                    logger.warning(f"QUICK_SYNC: Skipping zarr write - invalid bounds: zarr_y({y_start}:{y_end}), zarr_x({x_start}:{x_end}), img_y({img_y_start}:{img_y_end}), img_x({img_x_start}:{img_x_end}) (quick scan)")
        
        logger.info(f"QUICK_SYNC: Completed add_image_sync_quick at ({x_mm:.2f}, {y_mm:.2f}), channel={channel_idx}, timepoint={timepoint}")
    
    async def add_image_async(self, image: np.ndarray, x_mm: float, y_mm: float,
                              channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0):
        """Add image to the stitching queue for asynchronous processing."""
        await self.stitch_queue.put({
            'image': image.copy(),
            'x_mm': x_mm,
            'y_mm': y_mm,
            'channel_idx': channel_idx,
            'z_idx': z_idx,
            'timepoint': timepoint,
            'timestamp': time.time()
        })
    
    async def start_stitching(self):
        """Start the background stitching task."""
        if not self.is_stitching:
            self.is_stitching = True
            self.stitching_task = asyncio.create_task(self._stitching_loop())
            logger.info("Started background stitching task")
    
    async def stop_stitching(self):
        """Stop the background stitching task and process all remaining images in queue."""
        self.is_stitching = False
        
        # Process any remaining images in the queue
        logger.info("Processing remaining images in stitching queue before stopping...")
        remaining_count = 0
        
        while not self.stitch_queue.empty():
            try:
                frame_data = await asyncio.wait_for(
                    self.stitch_queue.get(), 
                    timeout=0.1  # Short timeout to avoid hanging
                )
                
                # Check if this is a quick scan
                is_quick_scan = frame_data.get('quick_scan', False)
                
                # Extract timepoint
                timepoint = frame_data.get('timepoint', 0)
                
                # Ensure timepoint exists in our tracking list
                if timepoint not in self.available_timepoints:
                    with self.zarr_lock:
                        if timepoint not in self.available_timepoints:
                            self.available_timepoints.append(timepoint)
                            self.available_timepoints.sort()
                            self._update_timepoint_metadata()
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                if is_quick_scan:
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync_quick,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx'],
                        timepoint
                    )
                else:
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx'],
                        timepoint
                    )
                remaining_count += 1
                
            except asyncio.TimeoutError:
                break  # No more items in queue
            except Exception as e:
                logger.error(f"Error processing remaining image in queue: {e}")
        
        if remaining_count > 0:
            logger.info(f"Processed {remaining_count} remaining images from stitching queue")
        
        # Wait for the stitching task to complete
        if self.stitching_task:
            await self.stitching_task
        
        # CRITICAL: Wait for all thread pool operations to complete
        logger.info("Waiting for all zarr operations to complete...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._wait_for_zarr_operations_complete)
        
        logger.info("Stopped background stitching task")
    
    async def _stitching_loop(self):
        """Background loop that processes the stitching queue."""
        while self.is_stitching:
            try:
                # Get frame from queue with timeout
                frame_data = await asyncio.wait_for(
                    self.stitch_queue.get(), 
                    timeout=1.0
                )
                
                # Check if this is a quick scan (only updates scales 1-5)
                is_quick_scan = frame_data.get('quick_scan', False)
                
                # Extract timepoint
                timepoint = frame_data.get('timepoint', 0)
                
                logger.info(f"STITCHING_LOOP: Processing image at ({frame_data['x_mm']:.2f}, {frame_data['y_mm']:.2f}), channel={frame_data['channel_idx']}, timepoint={timepoint}, quick_scan={is_quick_scan}")
                
                # Ensure timepoint exists in our tracking list (do this in main thread)
                if timepoint not in self.available_timepoints:
                    with self.zarr_lock:
                        if timepoint not in self.available_timepoints:
                            self.available_timepoints.append(timepoint)
                            self.available_timepoints.sort()
                            self._update_timepoint_metadata()
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                if is_quick_scan:
                    # Use quick scan method that only updates scales 1-5
                    logger.info(f"STITCHING_LOOP: Calling add_image_sync_quick for image at ({frame_data['x_mm']:.2f}, {frame_data['y_mm']:.2f})")
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync_quick,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx'],
                        timepoint
                    )
                    logger.info(f"STITCHING_LOOP: Completed add_image_sync_quick for image at ({frame_data['x_mm']:.2f}, {frame_data['y_mm']:.2f})")
                else:
                    # Use normal method that updates all scales
                    logger.info(f"STITCHING_LOOP: Calling add_image_sync for image at ({frame_data['x_mm']:.2f}, {frame_data['y_mm']:.2f})")
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx'],
                        timepoint
                    )
                    logger.info(f"STITCHING_LOOP: Completed add_image_sync for image at ({frame_data['x_mm']:.2f}, {frame_data['y_mm']:.2f})")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in stitching loop: {e}")
        
        # Process any final images that might have been added during the last iteration
        logger.debug("Stitching loop exited, checking for any final images in queue...")
        final_count = 0
        while not self.stitch_queue.empty():
            try:
                frame_data = await asyncio.wait_for(
                    self.stitch_queue.get(), 
                    timeout=0.1
                )
                
                # Check if this is a quick scan
                is_quick_scan = frame_data.get('quick_scan', False)
                
                # Extract timepoint
                timepoint = frame_data.get('timepoint', 0)
                
                # Ensure timepoint exists in our tracking list
                if timepoint not in self.available_timepoints:
                    with self.zarr_lock:
                        if timepoint not in self.available_timepoints:
                            self.available_timepoints.append(timepoint)
                            self.available_timepoints.sort()
                            self._update_timepoint_metadata()
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                if is_quick_scan:
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync_quick,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx'],
                        timepoint
                    )
                else:
                    await loop.run_in_executor(
                        self.executor,
                        self.add_image_sync,
                        frame_data['image'],
                        frame_data['x_mm'],
                        frame_data['y_mm'],
                        frame_data['channel_idx'],
                        frame_data['z_idx'],
                        timepoint
                    )
                final_count += 1
                
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing final image in stitching loop: {e}")
        
        if final_count > 0:
            logger.info(f"Stitching loop processed {final_count} final images before exiting")
    
    def get_canvas_region(self, x_mm: float, y_mm: float, width_mm: float, height_mm: float,
                          scale: int = 0, channel_idx: int = 0, timepoint: int = 0) -> np.ndarray:
        """
        Get a region from the canvas by zarr channel index.
        
        Args:
            x_mm: Center X position in millimeters
            y_mm: Center Y position in millimeters
            width_mm: Width in millimeters
            height_mm: Height in millimeters
            scale: Scale level to retrieve from
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            timepoint: Timepoint index (default 0)
            
        Returns:
            Retrieved image region as numpy array
        """
        # Validate channel index
        if channel_idx >= len(self.channels) or channel_idx < 0:
            logger.error(f"Channel index {channel_idx} out of bounds. Available channels: {len(self.channels)} (indices 0-{len(self.channels)-1})")
            return None
        
        # Validate timepoint
        if timepoint not in self.available_timepoints:
            logger.error(f"Timepoint {timepoint} not available. Available timepoints: {sorted(self.available_timepoints)}")
            return None
        
        with self.zarr_lock:
            # Validate zarr arrays exist
            if not hasattr(self, 'zarr_arrays') or scale not in self.zarr_arrays:
                logger.error(f"Zarr arrays not initialized or scale {scale} not available")
                return None
                
            zarr_array = self.zarr_arrays[scale]
            
            # Check if timepoint exists in zarr array (it might not if we're reading before writing)
            if timepoint >= zarr_array.shape[0]:
                logger.warning(f"Timepoint {timepoint} not yet written to zarr array (shape: {zarr_array.shape})")
                # Return zeros of the expected size
                scale_factor = 4 ** scale
                width_px = int(width_mm * 1000 / (self.pixel_size_xy_um * scale_factor))
                height_px = int(height_mm * 1000 / (self.pixel_size_xy_um * scale_factor))
                return np.zeros((height_px, width_px), dtype=zarr_array.dtype)
            
            # Double-check zarr array dimensions
            if channel_idx >= zarr_array.shape[1]:
                logger.error(f"Channel index {channel_idx} exceeds zarr array channel dimension {zarr_array.shape[1]}")
                return None
            
            # Convert to pixel coordinates
            center_x_px, center_y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)
            
            scale_factor = 4 ** scale
            width_px = int(width_mm * 1000 / (self.pixel_size_xy_um * scale_factor))
            height_px = int(height_mm * 1000 / (self.pixel_size_xy_um * scale_factor))
            
            # Calculate bounds
            x_start = max(0, center_x_px - width_px // 2)
            x_end = min(zarr_array.shape[4], x_start + width_px)
            y_start = max(0, center_y_px - height_px // 2)
            y_end = min(zarr_array.shape[3], y_start + height_px)
            
            # Read from zarr
            try:
                region = zarr_array[timepoint, channel_idx, 0, y_start:y_end, x_start:x_end]
                logger.debug(f"Successfully retrieved region from zarr at scale {scale}, channel {channel_idx}, timepoint {timepoint}")
                return region
            except IndexError as e:
                logger.error(f"IndexError reading from zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                logger.error(f"Zarr array shape: {zarr_array.shape}, trying to access timepoint {timepoint}")
                return None
            except Exception as e:
                logger.error(f"Error reading from zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                return None
    
    def get_canvas_region_by_channel_name(self, x_mm: float, y_mm: float, width_mm: float, height_mm: float,
                                         channel_name: str, scale: int = 0, timepoint: int = 0) -> np.ndarray:
        """
        Get a region from the canvas by channel name.
        
        Args:
            x_mm: Center X position in millimeters
            y_mm: Center Y position in millimeters
            width_mm: Width in millimeters
            height_mm: Height in millimeters
            channel_name: Human-readable channel name
            scale: Scale level to retrieve from
            timepoint: Timepoint index (default 0)
            
        Returns:
            Retrieved image region as numpy array
        """
        # Get the local zarr index for this channel
        try:
            channel_idx = self.get_zarr_channel_index(channel_name)
        except ValueError as e:
            logger.error(f"Channel not found: {e}")
            return None
            
        return self.get_canvas_region(x_mm, y_mm, width_mm, height_mm, scale, channel_idx, timepoint)
    
    def close(self):
        """Close the canvas and clean up resources."""
        if hasattr(self, 'zarr_array') and self.zarr_array is not None:
            self.zarr_array = None
        logger.info(f"Closed well canvas: {self.fileset_name}")
    
    def export_to_zip(self, zip_path):
        """
        Export the well canvas to a ZIP file.
        
        Args:
            zip_path (str): Path to the output ZIP file
        """
        
        try:
            # Create a temporary directory for the zarr data
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zarr_path = os.path.join(temp_dir, "data.zarr")
                
                # Copy the zarr data to the temporary location
                if self.zarr_path.exists():
                    shutil.copytree(self.zarr_path, temp_zarr_path)
                else:
                    logger.warning(f"Zarr path does not exist: {self.zarr_path}")
                    return
                
                # Create the ZIP file
                with zipfile.ZipFile(zip_path, 'w', allowZip64=True, compression=zipfile.ZIP_DEFLATED) as zf:
                    # Walk through the temporary zarr directory and add all files
                    for root, dirs, files in os.walk(temp_zarr_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Calculate relative path for the ZIP
                            relative_path = os.path.relpath(file_path, temp_dir)
                            # Use forward slashes for ZIP paths
                            arcname = relative_path.replace(os.sep, '/')
                            zf.write(file_path, arcname)
                
                logger.info(f"Exported well canvas to ZIP: {zip_path}")
                
        except Exception as e:
            logger.error(f"Failed to export well canvas to ZIP: {e}")
            raise
    
    def save_preview(self, action_ID: str = "canvas_preview"):
        """Save a preview image of the canvas at different scales."""
        try:
            preview_dir = self.base_path / "previews"
            preview_dir.mkdir(exist_ok=True)
            
            for scale in range(min(2, self.num_scales)):  # Save first 2 scales
                if scale in self.zarr_arrays:
                    # Get the first channel (usually brightfield)
                    array = self.zarr_arrays[scale]
                    if array.shape[1] > 0:  # Check if we have channels
                        # Get the image data (T=0, C=0, Z=0, :, :)
                        image_data = array[0, 0, 0, :, :]
                        
                        # Convert to PIL Image and save
                        if image_data.max() > image_data.min():  # Only save if there's actual data
                            # Normalize to 0-255
                            normalized = ((image_data - image_data.min()) / 
                                        (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                            image = Image.fromarray(normalized)
                            preview_path = preview_dir / f"{action_ID}_scale{scale}.png"
                            image.save(preview_path)
                            logger.info(f"Saved preview: {preview_path}")
                        
        except Exception as e:
            logger.warning(f"Failed to save preview: {e}")
    
    def _flush_and_sync_zarr_arrays(self):
        """
        Flush and synchronize all zarr arrays to ensure all data is written to disk.
        This is critical before ZIP export to prevent race conditions.
        """
        try:
            with self.zarr_lock:
                if hasattr(self, 'zarr_arrays'):
                    for scale, zarr_array in self.zarr_arrays.items():
                        try:
                            # Flush any pending writes to disk
                            if hasattr(zarr_array, 'flush'):
                                zarr_array.flush()
                            # Sync the underlying store
                            if hasattr(zarr_array.store, 'sync'):
                                zarr_array.store.sync()
                            logger.debug(f"Flushed and synced zarr array scale {scale}")
                        except Exception as e:
                            logger.warning(f"Error flushing zarr array scale {scale}: {e}")
                
                # Also shutdown and recreate the thread pool to ensure all tasks are complete
                if hasattr(self, 'executor'):
                    self.executor.shutdown(wait=True)
                    self.executor = ThreadPoolExecutor(max_workers=4)
                    logger.info("Thread pool shutdown and recreated to ensure all zarr operations complete")
                
                # Give the filesystem a moment to complete any pending I/O
                import time
                time.sleep(0.1)
                
                logger.info("All zarr arrays flushed and synchronized")
                
        except Exception as e:
            logger.error(f"Error during zarr flush and sync: {e}")
            raise RuntimeError(f"Failed to flush zarr arrays: {e}")

    def export_as_zip(self) -> bytes:
        """
        Export the entire zarr canvas as a zip file for upload to artifact manager.
        Uses robust ZIP64 creation that's compatible with S3 ZIP parsers.
        Creates ZIP directly to temporary file to avoid memory corruption with large files.
        
        Returns:
            bytes: The zip file content containing the entire zarr directory structure
        """
        # Use the file-based export and read into memory for backward compatibility
        temp_path = self.export_as_zip_file()
        try:
            with open(temp_path, 'rb') as f:
                return f.read()
        finally:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary ZIP file {temp_path}: {e}")
    
    def export_as_zip_file(self) -> str:
        """
        Export the entire zarr canvas as a zip file to a temporary file.
        Uses robust ZIP64 creation that's compatible with S3 ZIP parsers.
        Avoids memory corruption by writing directly to file.
        
        Returns:
            str: Path to the temporary ZIP file (caller must clean up)
        """
        import zipfile
        import tempfile
        import os
        
        # Create temporary file for ZIP creation to avoid memory issues
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zip', prefix='zarr_export_')
        
        try:
            # Close file descriptor immediately to avoid issues
            os.close(temp_fd)
            temp_fd = None  # Mark as closed
            
            # CRITICAL: Ensure all zarr operations are complete before ZIP export
            logger.info("Preparing zarr canvas for ZIP export...")
            self._flush_and_sync_zarr_arrays()
            
            # Force ZIP64 format explicitly for compatibility with S3 parser
            # Use minimal compression for reliability with many small files
            zip_kwargs = {
                'mode': 'w',
                'compression': zipfile.ZIP_STORED,  # No compression for reliability
                'allowZip64': True,
                'strict_timestamps': False  # Handle timestamp edge cases
            }
            
            # Create ZIP file with explicit ZIP64 support
            with zipfile.ZipFile(temp_path, **zip_kwargs) as zip_file:
                logger.info("Creating ZIP archive with explicit ZIP64 support...")
                
                # Build file list first to validate and count
                files_to_add = []
                total_size = 0
                
                for root, dirs, files in os.walk(self.zarr_path):
                    for file in files:
                        file_path = Path(root) / file
                        
                        # Skip files that don't exist or can't be read
                        if not file_path.exists() or not file_path.is_file():
                            logger.warning(f"Skipping non-existent or non-file: {file_path}")
                            continue
                            
                        try:
                            # Verify file is readable and get size
                            file_size = file_path.stat().st_size
                            total_size += file_size
                            
                            # Create relative path for ZIP archive
                            relative_path = file_path.relative_to(self.zarr_path)
                            # Use forward slashes for ZIP compatibility (standard requirement)
                            arcname = "data.zarr/" + str(relative_path).replace(os.sep, '/')
                            
                            files_to_add.append((file_path, arcname, file_size))
                            
                        except (OSError, IOError) as e:
                            logger.warning(f"Skipping unreadable file {file_path}: {e}")
                            continue
                
                logger.info(f"Validated {len(files_to_add)} files for ZIP archive (total: {total_size / (1024*1024):.1f} MB)")
                
                # Check if we need ZIP64 format (more than 65535 files or 4GB total)
                needs_zip64 = len(files_to_add) >= 65535 or total_size >= (4 * 1024 * 1024 * 1024)
                if needs_zip64:
                    logger.info(f"ZIP64 format required: {len(files_to_add)} files, {total_size / (1024*1024):.1f} MB")
                
                # Add files to ZIP in sorted order for consistent central directory
                files_to_add.sort(key=lambda x: x[1])  # Sort by arcname
                
                processed_files = 0
                for file_path, arcname, file_size in files_to_add:
                    try:
                        # Add file with explicit error handling
                        zip_file.write(file_path, arcname=arcname)
                        processed_files += 1
                        
                        # Progress logging every 1000 files
                        if processed_files % 1000 == 0:
                            logger.info(f"ZIP progress: {processed_files}/{len(files_to_add)} files processed")
                            
                    except Exception as e:
                        logger.error(f"Failed to add file to ZIP: {file_path} -> {arcname}: {e}")
                        continue
                
                # Add metadata with proper JSON formatting
                metadata = {
                    "canvas_info": {
                        "pixel_size_xy_um": self.pixel_size_xy_um,
                        "rotation_angle_deg": self.rotation_angle_deg,
                        "stage_limits": self.stage_limits,
                        "channels": self.channels,
                        "num_scales": self.num_scales,
                        "canvas_size_px": {
                            "width": self.canvas_width_px,
                            "height": self.canvas_height_px
                        },
                        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                        "squid_canvas_version": "1.0",
                        "zip_format": "ZIP64" if needs_zip64 else "standard"
                    }
                }
                
                metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
                zip_file.writestr("squid_canvas_metadata.json", metadata_json.encode('utf-8'))
                processed_files += 1   
                
                logger.info(f"ZIP creation completed: {processed_files} files processed")
                
            # Get file size for validation
            zip_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            
            # Enhanced ZIP validation specifically for S3 compatibility
            with open(temp_path, 'rb') as f:
                zip_content_for_validation = f.read()
            self._validate_zip_structure_for_s3(zip_content_for_validation)
            
            logger.info(f"ZIP export successful: {zip_size_mb:.2f} MB, {processed_files} files")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to export zarr canvas as zip: {e}")
            # Clean up temp file on error
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
            raise RuntimeError(f"Cannot export zarr canvas: {e}")
        finally:
            # Clean up file descriptor if still open
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except Exception:
                    pass  # Ignore errors closing fd

    def _validate_zip_structure_for_s3(self, zip_content: bytes) -> None:
        """
        Validate ZIP file structure specifically for S3 ZIP parser compatibility.
        Checks for proper central directory structure and ZIP64 format compliance.
        
        Args:
            zip_content (bytes): The ZIP file content to validate
            
        Raises:
            RuntimeError: If ZIP file structure is incompatible with S3 parser
        """
        try:
            import zipfile
            import io
            
            # Basic ZIP file validation
            zip_buffer = io.BytesIO(zip_content)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                file_list = zip_file.namelist()
                if not file_list:
                    raise RuntimeError("ZIP file is empty")
                
                zip_size_mb = len(zip_content) / (1024 * 1024)
                file_count = len(file_list)
                
                logger.info(f"Basic ZIP validation passed: {file_count} files, {zip_size_mb:.2f} MB")
                
                # Check for ZIP64 indicators (critical for S3 parser)
                is_zip64 = file_count >= 65535 or zip_size_mb >= 4000
                
                if is_zip64:
                    # For ZIP64 files, check that central directory can be found
                    # This mimics what the S3 ZIP parser does
                    logger.info("Validating ZIP64 central directory structure...")
                    
                    # Look for ZIP64 signatures in the file
                    zip64_eocd_locator = b"PK\x06\x07"  # ZIP64 End of Central Directory Locator
                    zip64_eocd = b"PK\x06\x06"          # ZIP64 End of Central Directory
                    standard_eocd = b"PK\x05\x06"       # Standard End of Central Directory
                    
                    # Check the last 128KB for these signatures (like S3 parser does)
                    tail_size = min(128 * 1024, len(zip_content))
                    tail_data = zip_content[-tail_size:]
                    
                    has_zip64_locator = zip64_eocd_locator in tail_data
                    has_zip64_eocd = zip64_eocd in tail_data
                    has_standard_eocd = standard_eocd in tail_data
                    
                    logger.info(f"ZIP64 structure check: locator={has_zip64_locator}, eocd={has_zip64_eocd}, standard_eocd={has_standard_eocd}")
                    
                    # ZIP64 files should have proper directory structures
                    if not (has_zip64_locator and has_standard_eocd):
                        logger.warning("ZIP64 format validation issues detected")
                    
                    # Verify we can read file info (this is what S3 parser tries to do)
                    test_files = min(10, len(file_list))
                    for i in range(test_files):
                        try:
                            info = zip_file.getinfo(file_list[i])
                            # Try to access file info that S3 parser needs
                            _ = info.filename
                            _ = info.file_size
                            _ = info.compress_size
                            _ = info.date_time
                        except Exception as e:
                            logger.warning(f"File info access issue for {file_list[i]}: {e}")
                
                # Test random file access (S3 parser does this)
                test_count = min(5, len(file_list))
                for i in range(0, len(file_list), max(1, len(file_list) // test_count)):
                    try:
                        with zip_file.open(file_list[i]) as f:
                            # Read just 1 byte to verify file can be opened
                            f.read(1)
                    except Exception as e:
                        logger.warning(f"File access test failed for {file_list[i]}: {e}")
                
                logger.info("S3-compatible ZIP validation completed successfully")
                
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file format: {e}")
            raise RuntimeError(f"Invalid ZIP file format: {e}")
        except Exception as e:
            logger.error(f"ZIP validation failed: {e}")
            raise RuntimeError(f"ZIP validation failed: {e}")

    def get_export_info(self) -> dict:
        """
        Get information about the current canvas for export planning.
        
        Returns:
            dict: Information about canvas size, data, and export feasibility
        """
        try:
            # Calculate actual disk usage instead of theoretical array size
            total_size_bytes = 0
            data_arrays = 0
            file_count = 0
            
            # Get actual file size on disk by walking the zarr directory
            if self.zarr_path.exists():
                try:
                    for file_path in self.zarr_path.rglob('*'):
                        if file_path.is_file():
                            try:
                                size = file_path.stat().st_size
                                total_size_bytes += size
                                file_count += 1
                            except (OSError, PermissionError) as e:
                                logger.warning(f"Could not read size of {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error walking zarr directory {self.zarr_path}: {e}")
                    # Fallback: try to get directory size using os.path.getsize
                    try:
                        import os
                        total_size_bytes = sum(os.path.getsize(os.path.join(dirpath, filename))
                                             for dirpath, dirnames, filenames in os.walk(self.zarr_path)
                                             for filename in filenames)
                    except Exception as fallback_e:
                        logger.error(f"Fallback size calculation also failed: {fallback_e}")
                        total_size_bytes = 0
            else:
                logger.warning(f"Zarr path does not exist: {self.zarr_path}")
            
            # Check which arrays have actual data
            for scale in range(self.num_scales):
                if scale in self.zarr_arrays:
                    array = self.zarr_arrays[scale]
                    
                    # Check if array has any data (non-zero values)
                    if array.size > 0:
                        try:
                            # Sample a small region to check for data
                            sample_size = min(100, array.shape[3], array.shape[4])
                            sample = array[0, 0, 0, :sample_size, :sample_size]
                            if sample.max() > 0:
                                data_arrays += 1
                        except Exception as e:
                            logger.warning(f"Could not sample array at scale {scale}: {e}")
            
            # For empty arrays, estimate zip size based on actual disk usage
            # Zarr metadata and empty arrays compress very well
            if data_arrays == 0:
                # Empty zarr structures are mostly metadata, compress to ~10% of disk size
                estimated_zip_size_mb = (total_size_bytes * 0.1) / (1024 * 1024)
            else:
                # Arrays with data compress moderately (20-40% depending on content)
                estimated_zip_size_mb = (total_size_bytes * 0.3) / (1024 * 1024)
            
            logger.info(f"Export info: {total_size_bytes / (1024*1024):.1f} MB on disk ({file_count} files), "
                       f"{data_arrays} arrays with data, estimated zip: {estimated_zip_size_mb:.1f} MB")
            
            return {
                "canvas_path": str(self.zarr_path),
                "total_size_bytes": total_size_bytes,
                "total_size_mb": total_size_bytes / (1024 * 1024),
                "estimated_zip_size_mb": estimated_zip_size_mb,
                "file_count": file_count,
                "num_scales": self.num_scales,
                "num_channels": len(self.channels),
                "channels": self.channels,
                "arrays_with_data": data_arrays,
                "canvas_dimensions": {
                    "width_px": self.canvas_width_px,
                    "height_px": self.canvas_height_px,
                    "pixel_size_um": self.pixel_size_xy_um
                },
                "export_feasible": True  # Removed arbitrary size limit - let S3 handle large files
            }
            
        except Exception as e:
            logger.error(f"Failed to get export info: {e}")
            return {
                "error": str(e),
                "export_feasible": False
            } 

    def _wait_for_zarr_operations_complete(self):
        """
        Wait for all zarr operations to complete and ensure filesystem sync.
        This prevents race conditions with ZIP export.
        """
        import time
        
        with self.zarr_lock:
            # Shutdown thread pool and wait for all tasks to complete
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
                self.executor = ThreadPoolExecutor(max_workers=4)
                logger.debug("Thread pool shutdown and recreated after stitching")
            
            # Flush all zarr arrays to ensure data is written
            if hasattr(self, 'zarr_arrays'):
                for scale, zarr_array in self.zarr_arrays.items():
                    try:
                        if hasattr(zarr_array, 'flush'):
                            zarr_array.flush()
                        if hasattr(zarr_array.store, 'sync'):
                            zarr_array.store.sync()
                    except Exception as e:
                        logger.warning(f"Error flushing zarr array scale {scale}: {e}")
            
            # Small delay to ensure filesystem operations complete
            time.sleep(0.2)
            
        logger.info("All zarr operations completed and synchronized") 

class WellZarrCanvas(WellZarrCanvasBase):
    """
    Well-specific zarr canvas for individual well imaging with well-center-relative coordinates.
    
    This class extends WellZarrCanvasBase to provide well-specific functionality:
    - Well-center-relative coordinate system (0,0 at well center)
    - Automatic well center calculation from well plate formats
    - Canvas size based on well diameter + configurable padding
    - Well-specific fileset naming (well_{row}{column}_{wellplate_type})
    """
    
    def __init__(self, well_row: str, well_column: int, wellplate_type: str = '96',
                 padding_mm: float = 1.0, base_path: str = None, 
                 pixel_size_xy_um: float = 0.333, channels: List[str] = None, **kwargs):
        """
        Initialize well-specific canvas.
        
        Args:
            well_row: Well row (e.g., 'A', 'B')
            well_column: Well column (e.g., 1, 2, 3)
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            padding_mm: Padding around well in mm (default 2.0)
            base_path: Base directory for zarr storage
            pixel_size_xy_um: Pixel size in micrometers
            channels: List of channel names
            **kwargs: Additional arguments passed to ZarrCanvas
        """
        # Import well plate format classes
        from squid_control.control.config import (
            WELLPLATE_FORMAT_6, WELLPLATE_FORMAT_12, WELLPLATE_FORMAT_24,
            WELLPLATE_FORMAT_96, WELLPLATE_FORMAT_384, CONFIG
        )
        
        # Get well plate format
        self.wellplate_format = self._get_wellplate_format(wellplate_type)
        
        # Store well information
        self.well_row = well_row
        self.well_column = well_column
        self.wellplate_type = wellplate_type
        self.padding_mm = padding_mm
        
        # Calculate well center coordinates (absolute stage coordinates)
        if hasattr(CONFIG, 'WELLPLATE_OFFSET_X_MM') and hasattr(CONFIG, 'WELLPLATE_OFFSET_Y_MM'):
            # Use offsets if available (hardware mode)
            x_offset = CONFIG.WELLPLATE_OFFSET_X_MM
            y_offset = CONFIG.WELLPLATE_OFFSET_Y_MM
        else:
            # No offsets (simulation mode)
            x_offset = 0
            y_offset = 0
            
        self.well_center_x = (self.wellplate_format.A1_X_MM + x_offset + 
                             (well_column - 1) * self.wellplate_format.WELL_SPACING_MM)
        self.well_center_y = (self.wellplate_format.A1_Y_MM + y_offset + 
                             (ord(well_row) - ord('A')) * self.wellplate_format.WELL_SPACING_MM)
        
        # Calculate canvas size (well diameter + padding)
        canvas_size_mm = self.wellplate_format.WELL_SIZE_MM + (2 * padding_mm)
        
        # Define well-relative stage limits (centered around 0,0)
        stage_limits = {
            'x_positive': canvas_size_mm / 2,
            'x_negative': -canvas_size_mm / 2,
            'y_positive': canvas_size_mm / 2,
            'y_negative': -canvas_size_mm / 2,
            'z_positive': 6
        }
        
        # Create well-specific fileset name
        fileset_name = f"well_{well_row}{well_column}_{wellplate_type}"
        
        # Initialize parent ZarrCanvas with well-specific parameters
        super().__init__(
            base_path=base_path,
            pixel_size_xy_um=pixel_size_xy_um,
            stage_limits=stage_limits,
            channels=channels,
            fileset_name=fileset_name,
            **kwargs
        )
        
        logger.info(f"WellZarrCanvas initialized for well {well_row}{well_column} ({wellplate_type})")
        logger.info(f"Well center: ({self.well_center_x:.2f}, {self.well_center_y:.2f}) mm")
        logger.info(f"Canvas size: {canvas_size_mm:.2f} mm, padding: {padding_mm:.2f} mm")
    
    def _get_wellplate_format(self, wellplate_type: str):
        """Get well plate format configuration."""
        from squid_control.control.config import (
            WELLPLATE_FORMAT_6, WELLPLATE_FORMAT_12, WELLPLATE_FORMAT_24,
            WELLPLATE_FORMAT_96, WELLPLATE_FORMAT_384
        )
        
        if wellplate_type == '6':
            return WELLPLATE_FORMAT_6
        elif wellplate_type == '12':
            return WELLPLATE_FORMAT_12
        elif wellplate_type == '24':
            return WELLPLATE_FORMAT_24
        elif wellplate_type == '96':
            return WELLPLATE_FORMAT_96
        elif wellplate_type == '384':
            return WELLPLATE_FORMAT_384
        else:
            return WELLPLATE_FORMAT_96  # Default
    
    def stage_to_pixel_coords(self, x_mm: float, y_mm: float, scale: int = 0) -> Tuple[int, int]:
        """
        Convert absolute stage coordinates to well-relative pixel coordinates.
        
        Args:
            x_mm: Absolute X position in mm
            y_mm: Absolute Y position in mm
            scale: Scale level
            
        Returns:
            Tuple of (x_pixel, y_pixel) coordinates relative to well center
        """
        # Convert absolute coordinates to well-relative coordinates
        well_relative_x = x_mm - self.well_center_x
        well_relative_y = y_mm - self.well_center_y
        
        # Use parent's coordinate conversion with well-relative coordinates
        return super().stage_to_pixel_coords(well_relative_x, well_relative_y, scale)
    
    def get_well_info(self) -> dict:
        """
        Get comprehensive information about this well canvas.
        
        Returns:
            dict: Well information including coordinates, size, and metadata
        """
        return {
            "well_info": {
                "row": self.well_row,
                "column": self.well_column,
                "well_id": f"{self.well_row}{self.well_column}",
                "wellplate_type": self.wellplate_type,
                "well_center_x_mm": self.well_center_x,
                "well_center_y_mm": self.well_center_y,
                "well_diameter_mm": self.wellplate_format.WELL_SIZE_MM,
                "well_spacing_mm": self.wellplate_format.WELL_SPACING_MM,
                "padding_mm": self.padding_mm
            },
            "canvas_info": {
                "canvas_width_mm": self.stage_limits['x_positive'] - self.stage_limits['x_negative'],
                "canvas_height_mm": self.stage_limits['y_positive'] - self.stage_limits['y_negative'],
                "coordinate_system": "well_relative",
                "origin": "well_center",
                "canvas_width_px": self.canvas_width_px,
                "canvas_height_px": self.canvas_height_px,
                "pixel_size_xy_um": self.pixel_size_xy_um
            }
        }
    
    def add_image_from_absolute_coords(self, image: np.ndarray, absolute_x_mm: float, absolute_y_mm: float,
                                     channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0):
        """
        Add an image using absolute stage coordinates (converts to well-relative internally).
        
        Args:
            image: Image array (2D)
            absolute_x_mm: Absolute X position in mm
            absolute_y_mm: Absolute Y position in mm
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
        """
        # Convert absolute coordinates to well-relative
        well_relative_x = absolute_x_mm - self.well_center_x
        well_relative_y = absolute_y_mm - self.well_center_y
        
        # Use parent's add_image_sync with well-relative coordinates
        self.add_image_sync(image, well_relative_x, well_relative_y, channel_idx, z_idx, timepoint)
        
        logger.debug(f"Added image at absolute coords ({absolute_x_mm:.2f}, {absolute_y_mm:.2f}) "
                    f"-> well-relative ({well_relative_x:.2f}, {well_relative_y:.2f}) for well {self.well_row}{self.well_column}")
    
    async def add_image_from_absolute_coords_async(self, image: np.ndarray, absolute_x_mm: float, absolute_y_mm: float,
                                                 channel_idx: int = 0, z_idx: int = 0, timepoint: int = 0):
        """
        Asynchronously add an image using absolute stage coordinates.
        
        Args:
            image: Image array (2D)
            absolute_x_mm: Absolute X position in mm
            absolute_y_mm: Absolute Y position in mm
            channel_idx: Local zarr channel index (0, 1, 2, etc.)
            z_idx: Z-slice index (default 0)
            timepoint: Timepoint index (default 0)
        """
        # Convert absolute coordinates to well-relative
        well_relative_x = absolute_x_mm - self.well_center_x
        well_relative_y = absolute_y_mm - self.well_center_y
        
        # Use parent's add_image_async with well-relative coordinates
        await self.add_image_async(image, well_relative_x, well_relative_y, channel_idx, z_idx, timepoint) 

class ExperimentManager:
    """
    Manages experiment folders containing well-specific zarr canvases.
    
    Each experiment is a folder containing multiple well canvases:
    ZARR_PATH/experiment_name/A1_96.zarr, A2_96.zarr, etc.
    
    This replaces the single-canvas system with a well-separated approach.
    """
    
    def __init__(self, base_path: str, pixel_size_xy_um: float):
        """
        Initialize the experiment manager.
        
        Args:
            base_path: Base directory for zarr storage (from ZARR_PATH env variable)
            pixel_size_xy_um: Pixel size in micrometers
        """
        self.base_path = Path(base_path)
        self.pixel_size_xy_um = pixel_size_xy_um
        self.current_experiment = None  # Current experiment name
        self.well_canvases = {}  # {well_id: WellZarrCanvas} for current experiment
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Set 'default' as the default experiment
        self._ensure_default_experiment()
        
        logger.info(f"ExperimentManager initialized at {self.base_path}")
    
    def _ensure_default_experiment(self):
        """
        Ensure that a 'default' experiment exists and is set as the current experiment.
        Creates the experiment if it doesn't exist.
        """
        default_experiment_name = 'default'
        default_experiment_path = self.base_path / default_experiment_name
        
        # Create default experiment if it doesn't exist
        if not default_experiment_path.exists():
            default_experiment_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created default experiment '{default_experiment_name}'")
        
        # Set as current experiment
        self.current_experiment = default_experiment_name
        logger.info(f"Set '{default_experiment_name}' as default experiment")
    
    @property
    def current_experiment_name(self) -> str:
        """Get the current experiment name."""
        return self.current_experiment
    
    def create_experiment(self, experiment_name: str, wellplate_type: str = '96', 
                         well_padding_mm: float = 1.0, initialize_all_wells: bool = False):
        """
        Create a new experiment folder and optionally initialize all well canvases.
        
        Args:
            experiment_name: Name of the experiment
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            well_padding_mm: Padding around each well in mm
            initialize_all_wells: If True, create canvases for all wells in the plate
            
        Returns:
            dict: Information about the created experiment
        """
        experiment_path = self.base_path / experiment_name
        
        if experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' already exists")
        
        # Create experiment directory
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Set as current experiment
        self.current_experiment = experiment_name
        self.well_canvases = {}
        
        logger.info(f"Created experiment '{experiment_name}' at {experiment_path}")
        
        # Optionally initialize all wells
        initialized_wells = []
        if initialize_all_wells:
            well_positions = self._get_all_well_positions(wellplate_type)
            for well_row, well_column in well_positions:
                try:
                    canvas = self.get_well_canvas(well_row, well_column, wellplate_type, well_padding_mm)
                    initialized_wells.append(f"{well_row}{well_column}")
                except Exception as e:
                    logger.warning(f"Failed to initialize well {well_row}{well_column}: {e}")
        
        return {
            "experiment_name": experiment_name,
            "experiment_path": str(experiment_path),
            "wellplate_type": wellplate_type,
            "initialized_wells": initialized_wells,
            "total_wells": len(initialized_wells) if initialize_all_wells else 0
        }
    
    def set_active_experiment(self, experiment_name: str):
        """
        Set the active experiment.
        
        Args:
            experiment_name: Name of the experiment to activate
            
        Returns:
            dict: Information about the activated experiment
        """
        experiment_path = self.base_path / experiment_name
        
        if not experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Close current well canvases
        for canvas in self.well_canvases.values():
            canvas.close()
        
        # Set new experiment
        self.current_experiment = experiment_name
        self.well_canvases = {}
        
        logger.info(f"Set active experiment to '{experiment_name}'")
        
        return {
            "experiment_name": experiment_name,
            "experiment_path": str(experiment_path),
            "message": f"Activated experiment '{experiment_name}'"
        }
    
    def list_experiments(self):
        """
        List all available experiments.
        
        Returns:
            dict: List of experiments and their information
        """
        experiments = []
        
        try:
            for item in self.base_path.iterdir():
                if item.is_dir():
                    # Count well canvases in this experiment
                    well_count = len([f for f in item.iterdir() if f.is_dir() and f.suffix == '.zarr'])
                    
                    experiments.append({
                        "name": item.name,
                        "path": str(item),
                        "is_active": item.name == self.current_experiment,
                        "well_count": well_count
                    })
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
        
        return {
            "experiments": experiments,
            "active_experiment": self.current_experiment,
            "total_count": len(experiments)
        }
    
    def remove_experiment(self, experiment_name: str):
        """
        Remove an experiment and all its well canvases.
        
        Args:
            experiment_name: Name of the experiment to remove
            
        Returns:
            dict: Information about the removed experiment
        """
        if experiment_name == self.current_experiment:
            raise ValueError(f"Cannot remove active experiment '{experiment_name}'. Please switch to another experiment first.")
        
        experiment_path = self.base_path / experiment_name
        
        if not experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Remove experiment directory and all contents
        import shutil
        shutil.rmtree(experiment_path)
        
        logger.info(f"Removed experiment '{experiment_name}'")
        
        return {
            "experiment_name": experiment_name,
            "message": f"Removed experiment '{experiment_name}'"
        }
    
    def reset_experiment(self, experiment_name: str = None):
        """
        Reset an experiment by removing all well canvases but keeping the folder.
        
        Args:
            experiment_name: Name of the experiment to reset (default: current experiment)
            
        Returns:
            dict: Information about the reset experiment
        """
        if experiment_name is None:
            experiment_name = self.current_experiment
        
        if experiment_name is None:
            raise ValueError("No experiment specified and no active experiment")
        
        experiment_path = self.base_path / experiment_name
        
        if not experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Close well canvases if this is the active experiment
        if experiment_name == self.current_experiment:
            for canvas in self.well_canvases.values():
                canvas.close()
            self.well_canvases = {}
        
        # Remove all .zarr directories in the experiment folder
        removed_count = 0
        for item in experiment_path.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                import shutil
                shutil.rmtree(item)
                removed_count += 1
        
        logger.info(f"Reset experiment '{experiment_name}', removed {removed_count} well canvases")
        
        return {
            "experiment_name": experiment_name,
            "removed_wells": removed_count,
            "message": f"Reset experiment '{experiment_name}'"
        }
    
    def get_well_canvas(self, well_row: str, well_column: int, wellplate_type: str = '96',
                       padding_mm: float = 1.0):
        """
        Get or create a well canvas for the current experiment.
        
        Args:
            well_row: Well row (e.g., 'A', 'B')
            well_column: Well column (e.g., 1, 2, 3)
            wellplate_type: Well plate type ('6', '12', '24', '96', '384')
            padding_mm: Padding around well in mm
            
        Returns:
            WellZarrCanvas: The well-specific canvas
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment. Create or set an experiment first.")
        
        well_id = f"{well_row}{well_column}_{wellplate_type}"
        
        if well_id not in self.well_canvases:
            # Create new well canvas in experiment folder
            experiment_path = self.base_path / self.current_experiment
            
            from squid_control.control.config import ChannelMapper, CONFIG
            all_channels = ChannelMapper.get_all_human_names()
            
            canvas = WellZarrCanvas(
                well_row=well_row,
                well_column=well_column,
                wellplate_type=wellplate_type,
                padding_mm=padding_mm,
                base_path=str(experiment_path),  # Use experiment folder as base
                pixel_size_xy_um=self.pixel_size_xy_um,
                channels=all_channels,
                rotation_angle_deg=CONFIG.STITCHING_ROTATION_ANGLE_DEG,
                initial_timepoints=20,
                timepoint_expansion_chunk=10
            )
            
            self.well_canvases[well_id] = canvas
            logger.info(f"Created well canvas {well_row}{well_column} for experiment '{self.current_experiment}'")
        
        return self.well_canvases[well_id]
    
    def list_well_canvases(self):
        """
        List all well canvases in the current experiment.
        
        Returns:
            dict: Information about well canvases
        """
        if self.current_experiment is None:
            return {
                "well_canvases": [],
                "experiment_name": None,
                "total_count": 0
            }
        
        canvases = []
        
        # List active canvases
        for well_id, canvas in self.well_canvases.items():
            well_info = canvas.get_well_info()
            canvases.append({
                "well_id": well_id,
                "well_row": canvas.well_row,
                "well_column": canvas.well_column,
                "wellplate_type": canvas.wellplate_type,
                "canvas_path": str(canvas.zarr_path),
                "well_center_x_mm": canvas.well_center_x,
                "well_center_y_mm": canvas.well_center_y,
                "padding_mm": canvas.padding_mm,
                "channels": len(canvas.channels),
                "timepoints": len(canvas.available_timepoints),
                "status": "active"
            })
        
        # List canvases on disk (in experiment folder)
        experiment_path = self.base_path / self.current_experiment
        for item in experiment_path.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                well_name = item.stem  # e.g., "well_A1_96"
                if well_name not in [c["well_id"] for c in canvases]:
                    canvases.append({
                        "well_id": well_name,
                        "canvas_path": str(item),
                        "status": "on_disk"
                    })
        
        return {
            "well_canvases": canvases,
            "experiment_name": self.current_experiment,
            "total_count": len(canvases)
        }
    
    def get_experiment_info(self, experiment_name: str = None):
        """
        Get detailed information about an experiment.
        
        Args:
            experiment_name: Name of the experiment (default: current experiment)
            
        Returns:
            dict: Detailed experiment information
        """
        if experiment_name is None:
            experiment_name = self.current_experiment
        
        if experiment_name is None:
            raise ValueError("No experiment specified and no active experiment")
        
        experiment_path = self.base_path / experiment_name
        
        if not experiment_path.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Count well canvases
        well_canvases = []
        total_size_bytes = 0
        
        for item in experiment_path.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                try:
                    # Calculate size
                    size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    total_size_bytes += size_bytes
                    
                    well_canvases.append({
                        "name": item.stem,
                        "path": str(item),
                        "size_bytes": size_bytes,
                        "size_mb": size_bytes / (1024 * 1024)
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for {item}: {e}")
        
        return {
            "experiment_name": experiment_name,
            "experiment_path": str(experiment_path),
            "is_active": experiment_name == self.current_experiment,
            "well_canvases": well_canvases,
            "total_wells": len(well_canvases),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": total_size_bytes / (1024 * 1024)
        }
    
    def _get_all_well_positions(self, wellplate_type: str):
        """Get all well positions for a given plate type."""
        from squid_control.control.config import (
            WELLPLATE_FORMAT_6, WELLPLATE_FORMAT_12, WELLPLATE_FORMAT_24,
            WELLPLATE_FORMAT_96, WELLPLATE_FORMAT_384
        )
        
        if wellplate_type == '6':
            max_rows, max_cols = 2, 3  # A-B, 1-3
        elif wellplate_type == '12':
            max_rows, max_cols = 3, 4  # A-C, 1-4
        elif wellplate_type == '24':
            max_rows, max_cols = 4, 6  # A-D, 1-6
        elif wellplate_type == '96':
            max_rows, max_cols = 8, 12  # A-H, 1-12
        elif wellplate_type == '384':
            max_rows, max_cols = 16, 24  # A-P, 1-24
        else:
            max_rows, max_cols = 8, 12  # Default to 96-well
        
        positions = []
        for row_idx in range(max_rows):
            for col_idx in range(max_cols):
                row_letter = chr(ord('A') + row_idx)
                col_number = col_idx + 1
                positions.append((row_letter, col_number))
        
        return positions
    
    def close(self):
        """Close all well canvases and clean up resources."""
        for canvas in self.well_canvases.values():
            canvas.close()
        self.well_canvases = {}
        logger.info("ExperimentManager closed")


# Alias for backward compatibility
ZarrCanvas = WellZarrCanvasBase
