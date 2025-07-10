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

logger = logging.getLogger(__name__)

class ZarrCanvas:
    """
    Manages an OME-Zarr canvas for live microscope image stitching.
    Creates and maintains a multi-scale pyramid structure for efficient viewing.
    
    Example usage with zarr upload:
        # Create and initialize canvas
        canvas = ZarrCanvas(base_path="/tmp/zarr", pixel_size_xy_um=0.33, stage_limits={...})
        canvas.initialize_canvas()
        
        # Add images during scanning
        canvas.add_image_sync(image, x_mm=10.0, y_mm=15.0, channel_idx=0, timepoint=0)
        
        # Check export feasibility
        export_info = canvas.get_export_info()
        if export_info["export_feasible"]:
            # Export for upload to artifact manager
            zip_content = canvas.export_as_zip()
            # Use the zip_content with the microscope service upload_zarr_dataset API
    """
    
    def __init__(self, base_path: str, pixel_size_xy_um: float, stage_limits: Dict[str, float], 
                 channels: List[str] = None, chunk_size: int = 256, rotation_angle_deg: float = 0.0,
                 initial_timepoints: int = 20, timepoint_expansion_chunk: int = 10, fileset_name: str = "live_stitching"):
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
        
        logger.info(f"ZarrCanvas initialized: {self.canvas_width_px}x{self.canvas_height_px} px, "
                    f"{self.num_scales} scales, chunk_size={chunk_size}, "
                    f"initial_timepoints={self.initial_timepoints}, expansion_chunk={self.timepoint_expansion_chunk}")
    
    def _calculate_num_scales(self) -> int:
        """Calculate the number of pyramid levels needed."""
        min_size = 512  # Minimum size for lowest resolution
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
                array = root.create_dataset(
                    str(scale),
                    shape=(self.initial_timepoints, len(self.channels), 1, height, width),
                    chunks=(1, 1, 1, self.chunk_size, self.chunk_size),
                    dtype='uint8',
                    fill_value=0,
                    overwrite=True
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
        
        # Convert to pixels at scale 0
        x_px = int((x_mm + x_offset_mm) * 1000 / self.pixel_size_xy_um)
        y_px = int((y_mm + y_offset_mm) * 1000 / self.pixel_size_xy_um)
        
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
                
                # Write to zarr at the specified timepoint
                if y_end > y_start and x_end > x_start:
                    try:
                        zarr_array[timepoint, channel_idx, z_idx, y_start:y_end, x_start:x_end] = \
                            scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]
                        logger.debug(f"Successfully wrote image to zarr at scale {scale}, channel {channel_idx}, timepoint {timepoint}")
                    except IndexError as e:
                        logger.error(f"IndexError writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                        logger.error(f"Zarr array shape: {zarr_array.shape}, trying to access timepoint {timepoint}")
                    except Exception as e:
                        logger.error(f"Error writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
    
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
        
        # For quick scan, we skip rotation to reduce computation pressure
        # The image should already be rotated and flipped by the caller
        processed_image = image
        
        with self.zarr_lock:
            # Ensure zarr arrays are sized correctly for this timepoint (lazy expansion)
            self._ensure_timepoint_exists_in_zarr(timepoint)
            
            # Only process scales 1-5 (skip scale 0 for performance)
            for scale in range(1, min(self.num_scales, 6)):  # scales 1-5
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
                
                # Crop image if it extends beyond canvas
                img_y_start = max(0, -y_px + scaled_image.shape[0] // 2)
                img_y_end = img_y_start + (y_end - y_start)
                img_x_start = max(0, -x_px + scaled_image.shape[1] // 2)
                img_x_end = img_x_start + (x_end - x_start)
                
                # Write to zarr at the specified timepoint
                if y_end > y_start and x_end > x_start:
                    try:
                        zarr_array[timepoint, channel_idx, z_idx, y_start:y_end, x_start:x_end] = \
                            scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]
                        logger.debug(f"Successfully wrote image to zarr at scale {scale}, channel {channel_idx}, timepoint {timepoint} (quick scan)")
                    except IndexError as e:
                        logger.error(f"IndexError writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
                        logger.error(f"Zarr array shape: {zarr_array.shape}, trying to access timepoint {timepoint}")
                    except Exception as e:
                        logger.error(f"Error writing to zarr array at scale {scale}, channel {channel_idx}, timepoint {timepoint}: {e}")
    
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
                    # Use normal method that updates all scales
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
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        logger.info("ZarrCanvas closed")
    
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
    
    def export_as_zip(self) -> bytes:
        """
        Export the entire zarr canvas as a zip file for upload to artifact manager.
        
        Returns:
            bytes: The zip file content containing the entire zarr directory structure
        """
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
                # Walk through the entire zarr directory
                for root, dirs, files in os.walk(self.zarr_path):
                    for file in files:
                        file_path = Path(root) / file
                        # Create relative path within the zarr directory
                        relative_path = file_path.relative_to(self.zarr_path)
                        # Build the archive name with fixed "data.zarr" prefix
                        if relative_path.parts:
                            # Join "data.zarr" with the relative path using forward slashes (ZIP standard)
                            arcname = "data.zarr/" + "/".join(relative_path.parts)
                        else:
                            # This shouldn't happen, but handle edge case
                            arcname = "data.zarr/" + file_path.name
                        zip_file.write(file_path, arcname=arcname)
                        logger.debug(f"Added to zip: {arcname}")
                
                # Add a metadata file with canvas information
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
                        "squid_canvas_version": "1.0"
                    }
                }
                
                import json
                metadata_json = json.dumps(metadata, indent=2)
                zip_file.writestr("squid_canvas_metadata.json", metadata_json)
                
            zip_buffer.seek(0)
            zip_content = zip_buffer.getvalue()
            zip_size_mb = len(zip_content) / (1024 * 1024)
            
            logger.info(f"Exported zarr canvas as zip: {zip_size_mb:.2f} MB")
            return zip_content
            
        except Exception as e:
            logger.error(f"Failed to export zarr canvas as zip: {e}")
            raise RuntimeError(f"Cannot export zarr canvas: {e}")
        finally:
            zip_buffer.close()
    
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
                "export_feasible": estimated_zip_size_mb < 1000  # Reasonable limit
            }
            
        except Exception as e:
            logger.error(f"Failed to get export info: {e}")
            return {
                "error": str(e),
                "export_feasible": False
            } 