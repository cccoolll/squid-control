# Image Map with Stitching Feature

This feature adds image stitching capabilities to the microscope control system, creating a map of the microscope's field of view as it moves across the sample.

## Overview

The stitching system creates a continuously updated zarr canvas that users can access through chunk-based queries. As the microscope moves and captures video frames, each frame is automatically placed in the correct location on a large canvas based on the stage position.

## Key Features

### 1. Multi-Scale Canvas
- Creates zarr files with pyramid levels (scale0-5)
- scale0: Base resolution
- scale1:  Progressively downsampled, but also base resolution for fast scan (750x750 frame resolution)
- scale2-5: Progressively downsampled versions (4x reduction per level)
- 256x256 pixel chunks for efficient access

### 2. Real-Time Frame Placement & Normal scan
- Extracts stage position from video frames
- Converts physical coordinates (mm) to pixel coordinates
- Places frames directly on canvas without complex blending
- Updates all pyramid levels automatically
For normal scan:
- Extracts stage position, snap an image with full resolution
- Places images directly on canvas without complex blending
- Updates all pyramid levels automatically

### 3. Hypha Service Integration
- New service methods for canvas access:
  - `normal_scan_with_stitching(start_x_mm, start_y_mm, Nx, Ny, dx_mm, dy_mm, illumination_settings, do_contrast_autofocus, do_reflection_af, action_ID)` - Perform a normal scan with stitching
  - `get_stitched_region(center_x_mm, center_y_mm, width_mm, height_mm, scale_level, channel_name, output_format)` - Get region from canvas
  - `reset_stitching_canvas()` - Reset the canvas

## Configuration

### Environment Variables
- `ZARR_PATH`: Base directory for zarr storage (default: `/tmp/zarr_canvas`)

### Stage Limits
The system uses predefined stage limits for canvas calculation:
```python
stage_limits = {
    "x_positive": 120,    # mm
    "x_negative": 0,      # mm  
    "y_positive": 86,     # mm
    "y_negative": 0,      # mm
    "z_positive": 6       # mm
}
```

### Canvas Calculation
- Physical area: 120mm x 86mm
- Pixel size: Can be calculated in @squid_controller.py

## Technical Details

### Thread Safety
- Uses asyncio queues for frame processing
- Thread pool executor for zarr operations
- RLock for thread-safe zarr updates

### Performance
- Non-blocking frame addition to prevent video delays
- Efficient zarr chunking for fast access
- PNG compression with base64 encoding for data transfer
- Automatic RGB to grayscale conversion for reduced storage

## Usage Examples

### 1. Performing a Normal Scan with Stitching

```python
# Via Hypha service
microscope_service = await server.get_service("microscope-service-id")

# Define scan parameters
result = await microscope_service.normal_scan_with_stitching(
    start_x_mm=10.0,
    start_y_mm=10.0,
    Nx=5,  # 5 positions in X
    Ny=5,  # 5 positions in Y
    dx_mm=1.0,  # 1mm spacing in X
    dy_mm=1.0,  # 1mm spacing in Y
    illumination_settings=[
        {'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100},
        {'channel': 'Fluorescence 488 nm Ex', 'intensity': 30, 'exposure_time': 200}
    ],
    do_contrast_autofocus=False,
    do_reflection_af=False,
    action_ID='my_scan_001'
)
```

### 2. Retrieving a Stitched Region

```python
# Get a 5x5mm region starting at (12.5, 12.5)mm (top-left corner)
region_data = await microscope_service.get_stitched_region(
    start_x_mm=12.5,
    start_y_mm=12.5,
    width_mm=5.0,
    height_mm=5.0,
    scale_level=0,  # Full resolution
    channel_name='BF LED matrix full',
    output_format='base64'  # Get as base64 PNG
)

# Display the image
if region_data['success']:
    import base64
    from PIL import Image
    import io
    
    img_data = base64.b64decode(region_data['data'])
    img = Image.open(io.BytesIO(img_data))
    img.show()
```

### 3. Working with Different Scale Levels

```python
# Get a lower resolution overview starting from (10, 10)mm
overview = await microscope_service.get_stitched_region(
    start_x_mm=10.0,  # Top-left corner
    start_y_mm=10.0,
    width_mm=50.0,     # Large area
    height_mm=50.0,
    scale_level=2,     # 1/16 resolution
    channel_name='BF LED matrix full',
    output_format='array'  # Get as numpy array
)
```

### 4. Resetting the Canvas

```python
# Clear all stitched data
result = await microscope_service.reset_stitching_canvas()
```

## Implementation Details

### ZarrCanvas Class
The core stitching functionality is implemented in `squid_control/stitching/zarr_canvas.py`:

- **Initialization**: Creates OME-Zarr structure with proper metadata
- **Coordinate Conversion**: Handles stage (mm) to pixel coordinate mapping
- **Multi-scale Updates**: Automatically updates all pyramid levels
- **Async Processing**: Background thread for non-blocking stitching
- **Thread Safety**: Uses locks for concurrent access

### Integration with SquidController
The `normal_scan_with_stitching` method in `squid_controller.py`:

- **Snake Pattern Scanning**: Efficient bidirectional scanning
- **Autofocus Integration**: Supports both contrast and reflection AF
- **Multi-channel Support**: Acquires multiple channels per position
- **Progress Logging**: Detailed logging of scan progress

## Error Handling

The system includes robust error handling:
- Graceful degradation when stage position is unavailable
- Automatic canvas boundary checking
- Service restart recovery
- Comprehensive logging at all levels

## Future Enhancements

- Quick scan mode with continuous stage movement
- Advanced blending algorithms for seamless stitching
- Real-time preview during scanning
- Export to standard formats (OME-TIFF, etc.)
- ROI-based selective scanning