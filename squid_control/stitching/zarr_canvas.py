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

logger = logging.getLogger(__name__)

class ZarrCanvas:
    """
    Manages an OME-Zarr canvas for live microscope image stitching.
    Creates and maintains a multi-scale pyramid structure for efficient viewing.
    """
    
    def __init__(self, base_path: str, pixel_size_xy_um: float, stage_limits: Dict[str, float], 
                 channels: List[str] = None, chunk_size: int = 256):
        """
        Initialize the Zarr canvas.
        
        Args:
            base_path: Base directory for zarr storage (from ZARR_PATH env variable)
            pixel_size_xy_um: Pixel size in micrometers
            stage_limits: Dictionary with x_positive, x_negative, y_positive, y_negative in mm
            channels: List of channel names
            chunk_size: Size of chunks in pixels (default 256)
        """
        self.base_path = Path(base_path)
        self.pixel_size_xy_um = pixel_size_xy_um
        self.stage_limits = stage_limits
        self.channels = channels or ['BF_LED_matrix_full']
        self.chunk_size = chunk_size
        self.zarr_path = self.base_path / "live_stitching.zarr"
        
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
        
        # Queue for frame stitching
        self.stitch_queue = asyncio.Queue(maxsize=100)
        self.stitching_task = None
        self.is_stitching = False
        
        logger.info(f"ZarrCanvas initialized: {self.canvas_width_px}x{self.canvas_height_px} px, "
                    f"{self.num_scales} scales, chunk_size={chunk_size}")
    
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
                    "name": "live_stitching",
                    "version": "0.4"
                }],
                "omero": {
                    "channels": [{"label": ch, "color": "FFFFFF"} for ch in self.channels]
                }
            }
            
            # Create arrays for each scale level
            for scale in range(self.num_scales):
                scale_factor = 4 ** scale
                width = self.canvas_width_px // scale_factor
                height = self.canvas_height_px // scale_factor
                
                # Create the array (T, C, Z, Y, X)
                # For now: 1 timepoint, len(channels) channels, 1 z-slice
                array = root.create_dataset(
                    str(scale),
                    shape=(1, len(self.channels), 1, height, width),
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
    
    def add_image_sync(self, image: np.ndarray, x_mm: float, y_mm: float, 
                       channel_idx: int = 0, z_idx: int = 0):
        """
        Synchronously add an image to the canvas at the specified position.
        Updates all pyramid levels.
        
        Args:
            image: Image array (2D)
            x_mm: X position in millimeters
            y_mm: Y position in millimeters
            channel_idx: Channel index
            z_idx: Z-slice index (default 0)
        """
        with self.zarr_lock:
            for scale in range(self.num_scales):
                scale_factor = 4 ** scale
                
                # Get pixel coordinates for this scale
                x_px, y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)
                
                # Resize image if needed
                if scale > 0:
                    import cv2
                    new_size = (image.shape[1] // scale_factor, image.shape[0] // scale_factor)
                    scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                else:
                    scaled_image = image
                
                # Get the zarr array for this scale
                zarr_array = self.zarr_arrays[scale]
                
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
                
                # Write to zarr
                if y_end > y_start and x_end > x_start:
                    zarr_array[0, channel_idx, z_idx, y_start:y_end, x_start:x_end] = \
                        scaled_image[img_y_start:img_y_end, img_x_start:img_x_end]
    
    async def add_image_async(self, image: np.ndarray, x_mm: float, y_mm: float,
                              channel_idx: int = 0, z_idx: int = 0):
        """Add image to the stitching queue for asynchronous processing."""
        await self.stitch_queue.put({
            'image': image.copy(),
            'x_mm': x_mm,
            'y_mm': y_mm,
            'channel_idx': channel_idx,
            'z_idx': z_idx,
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
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.add_image_sync,
                    frame_data['image'],
                    frame_data['x_mm'],
                    frame_data['y_mm'],
                    frame_data['channel_idx'],
                    frame_data['z_idx']
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
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.add_image_sync,
                    frame_data['image'],
                    frame_data['x_mm'],
                    frame_data['y_mm'],
                    frame_data['channel_idx'],
                    frame_data['z_idx']
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
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.add_image_sync,
                    frame_data['image'],
                    frame_data['x_mm'],
                    frame_data['y_mm'],
                    frame_data['channel_idx'],
                    frame_data['z_idx']
                )
                final_count += 1
                
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing final image in stitching loop: {e}")
        
        if final_count > 0:
            logger.info(f"Stitching loop processed {final_count} final images before exiting")
    
    def get_canvas_region(self, x_mm: float, y_mm: float, width_mm: float, height_mm: float,
                          scale: int = 0, channel_idx: int = 0) -> np.ndarray:
        """
        Get a region from the canvas.
        
        Args:
            x_mm: Center X position in millimeters
            y_mm: Center Y position in millimeters
            width_mm: Width in millimeters
            height_mm: Height in millimeters
            scale: Scale level to retrieve from
            channel_idx: Channel index
            
        Returns:
            Retrieved image region as numpy array
        """
        with self.zarr_lock:
            # Convert to pixel coordinates
            center_x_px, center_y_px = self.stage_to_pixel_coords(x_mm, y_mm, scale)
            
            scale_factor = 4 ** scale
            width_px = int(width_mm * 1000 / (self.pixel_size_xy_um * scale_factor))
            height_px = int(height_mm * 1000 / (self.pixel_size_xy_um * scale_factor))
            
            # Calculate bounds
            x_start = max(0, center_x_px - width_px // 2)
            x_end = min(self.zarr_arrays[scale].shape[4], x_start + width_px)
            y_start = max(0, center_y_px - height_px // 2)
            y_end = min(self.zarr_arrays[scale].shape[3], y_start + height_px)
            
            # Read from zarr
            region = self.zarr_arrays[scale][0, channel_idx, 0, y_start:y_end, x_start:x_end]
            
            return region
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        logger.info("ZarrCanvas closed")
    
    def save_preview(self, action_ID: str = "canvas_preview"):
        """Save a preview image of the entire canvas from the lowest resolution scale."""
        if not hasattr(self, 'zarr_arrays') or not self.zarr_arrays:
            logger.warning('No zarr arrays available for preview generation')
            return None
        
        try:
            # Use the highest scale level (lowest resolution)
            scale_level = self.num_scales - 1
            
            logger.info(f'Generating preview image from scale{scale_level} for entire canvas')
            
            # Get the entire canvas at the lowest resolution
            with self.zarr_lock:
                zarr_array = self.zarr_arrays[scale_level]
                # Get the entire array for the first channel (T=0, C=0, Z=0, all Y, all X)
                preview_region = zarr_array[0, 0, 0, :, :]
            
            if preview_region.size == 0:
                logger.warning('Preview region is empty, skipping preview generation')
                return None
            
            # Convert to PIL Image
            from PIL import Image
            from datetime import datetime
            
            # Ensure data is uint8
            if preview_region.dtype != np.uint8:
                # Scale to 8-bit
                if preview_region.max() > 0:
                    preview_region = (preview_region / preview_region.max() * 255).astype(np.uint8)
                else:
                    preview_region = preview_region.astype(np.uint8)
            
            # Create PIL image
            preview_img = Image.fromarray(preview_region)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            preview_filename = f'{action_ID}_preview.png'
            
            # Save in the same directory as the zarr canvas
            preview_path = self.base_path / preview_filename
            
            # Save the image
            preview_img.save(preview_path, 'PNG')
            
            # Calculate physical dimensions
            scale_factor = 4 ** scale_level
            pixel_size_at_scale = self.pixel_size_xy_um * scale_factor
            width_mm = preview_region.shape[1] * pixel_size_at_scale / 1000
            height_mm = preview_region.shape[0] * pixel_size_at_scale / 1000
            
            logger.info(f'Saved canvas preview image: {preview_path}')
            logger.info(f'Preview image size: {preview_region.shape[1]}x{preview_region.shape[0]} pixels, '
                        f'covering {width_mm:.1f}x{height_mm:.1f}mm canvas area')
            
            return str(preview_path)
            
        except Exception as e:
            logger.error(f'Error generating canvas preview: {e}')
            return None 