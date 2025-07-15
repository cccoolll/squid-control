# Feature Introduction

1. **Stage Software Barrier**: The software barrier prevents the stage from moving beyond a certain area, ensuring that the stage does not collide with the microscope hardware.

    We have a function called 'is_point_in_concave_hull' that checks if a point is within the software barrier. The barrier is defined by a concave hull, which is a polygon that encloses the stage area. The file is 'edge_positions.json'. The points are in usteps instead of mm. Take X axis as a example. You can calculate the usteps from mm using this formula:

    ```
    // You can find these values in the configuration file
    CONFIG.STAGE_MOVEMENT_SIGN_X
            * int(
                Distance_mm
                / (
                    CONFIG.SCREW_PITCH_X_MM
                    / (CONFIG.MICROSTEPPING_DEFAULT_X * CONFIG.FULLSTEPS_PER_REV_X)
                )
            )
    ```
    The function 'is_point_in_concave_hull' will return True if the point is within the barrier, and False otherwise.

2. **Simulated Sample**: The simulated sample is a virtual sample that can be used for testing the microscope software without a physical sample. The simulated sample consists of Zarr data stored in ZIP files that contain high-resolution microscopy images. It's handled by the 'Camera_Simulation' class in the 'camera_default.py' file.

    When a user wants to acquire an image, the workflow for the simulated camera is as follows:

    - The user sends a command to the microscope service with position, channel, exposure, and intensity parameters.
    - The microscope service calls the functions in the 'Camera_Simulation' class.
    - The Camera_Simulation uses the ZarrImageManager to retrieve image data from the Zarr archives.
    - The ZarrImageManager first attempts direct access to the Zarr data for better performance.
    - If direct access fails, it falls back to a chunk-based approach, assembling the full region from smaller chunks.
    - The retrieved image is processed with the requested exposure time and intensity settings.
    - The processed image is returned to the user.

    <img style="width:auto;" src="./assets/how_simulated_sample_works.png"> 

    Some areas in the stage don't have sample data. The simulated camera will return a default image in these areas. If you want to know the location of the sample data, you can check the 'docs/coordinates_of_fovs_simulated_sample.csv' file.

    ### Zarr Image Workflow

    The diagram below illustrates the workflow for retrieving images from Zarr archives:

    <img style="width:auto;" src="./assets/zarr_image_workflow.png">

    This workflow shows how the microscope control interface initiates requests for images, how the Camera_Simulation class processes these requests, and how the ZarrImageManager retrieves and processes data from Zarr archives stored in ZIP files.

The simulation mode includes a virtual microscope sample using Zarr data archives. This allows you to test the microscope software without a physical sample. The simulated camera retrieves image data based on the current stage position, applies exposure and intensity adjustments, and returns realistic microscopy images.

#### Simulated Sample Features:
- Supports different imaging channels (brightfield and fluorescence)
- Adjustable exposure time and intensity
- Realistic Z-axis blurring for out-of-focus images
- High-resolution sample data covering the stage area

3. **Zarr Canvas & Image Stitching**: The Zarr Canvas & Image Stitching system enables real-time creation of large field-of-view images from multiple microscope acquisitions. This advanced feature provides comprehensive experiment management with well-based organization and multi-scale data storage.

### Zarr Canvas & Image Stitching Overview

The stitching system creates continuously updated OME-Zarr canvases that users can access through chunk-based queries. As the microscope moves and captures images, each frame is automatically placed in the correct location on a large canvas based on the stage position, creating a comprehensive map of the sample area.

#### **Core Architecture**

**Multi-Scale Canvas Structure:**
- **OME-Zarr Compliance**: Full OME-Zarr 0.4 specification support with proper metadata
- **Pyramid Levels**: Multi-scale pyramid with 4x downsampling between levels
  - scale0: Full resolution (base level)
  - scale1: 1/4 resolution (also base for quick scan)
  - scale2-5: Progressively downsampled versions (1/16, 1/64, etc.)
- **Optimized Chunking**: 256x256 pixel chunks for efficient I/O performance
- **Memory Efficiency**: Lazy loading and background processing for large datasets

**Well-Based Experiment Management:**
- **Individual Well Canvases**: Each well gets its own zarr canvas for precise control
- **Experiment Organization**: Hierarchical structure with experiments containing multiple well canvases
- **Automatic Well Detection**: System automatically determines which well contains the current stage position
- **Well-Relative Coordinates**: Each well canvas uses well-center-relative coordinate system

#### **Scanning Modes**

**Normal Scan with Stitching:**
- Grid-based scanning with configurable spacing (dx_mm, dy_mm)
- Multi-channel support (brightfield and fluorescence)
- Autofocus integration (contrast and reflection-based)
- Snake pattern scanning for efficiency
- Real-time stitching to OME-Zarr format
- Support for multiple wells in a single scan

**Quick Scan with Stitching:**
- High-speed continuous scanning (up to 10fps)
- Brightfield-only mode with exposure ≤ 30ms
- 4-stripe pattern per well for comprehensive coverage
- Optimized for performance with scale1-5 updates only
- Continuous stage movement with synchronized image acquisition

#### **Technical Implementation**

**ZarrCanvas Classes (`squid_control/stitching/zarr_canvas.py`):**

- **`WellZarrCanvasBase`**: Core stitching functionality with OME-Zarr compliance
  - Multi-scale pyramid creation and management
  - Coordinate conversion (stage mm → pixel coordinates)
  - Background stitching with asyncio queues
  - Thread-safe zarr operations with RLock

- **`WellZarrCanvas`**: Well-specific implementation
  - Automatic well center calculation from well plate formats
  - Well-relative coordinate system (0,0 at well center)
  - Canvas size based on well diameter + configurable padding
  - Well-specific fileset naming (well_{row}{column}_{wellplate_type})

- **`ExperimentManager`**: Manages experiment folders and well canvas lifecycle
  - Experiment creation, listing, and management
  - Well canvas lifecycle management
  - Automatic experiment switching and cleanup

**Performance Optimizations:**
- **Background Stitching**: Non-blocking frame processing with asyncio queues
- **Thread Safety**: RLock-based concurrent access to zarr arrays
- **Memory Management**: Automatic cleanup and resource management
- **Quick Scan Mode**: Optimized for high-speed acquisition with selective scale updates
- **Bounds Validation**: Always validate bounds before zarr write operations to prevent zero-size chunks

#### **API Integration**

The system provides comprehensive Hypha service integration through the following endpoints:

**Scanning Operations:**
- `normal_scan_with_stitching()`: Perform grid-based scanning with multi-channel support
- `quick_scan_with_stitching()`: High-speed continuous scanning for brightfield imaging
- `stop_scan_and_stitching()`: Stop ongoing scanning operations

**Data Retrieval:**
- `get_stitched_region()`: Retrieve regions from stitched canvases with multiple output formats
- `get_canvas_chunk()`: Low-level chunk access for custom applications

**Experiment Management:**
- `create_experiment()`: Create new experiments with optional well initialization
- `list_experiments()`: View all available experiments and their status
- `set_active_experiment()`: Switch between experiments for data collection
- `remove_experiment()`: Clean up experiment data
- `reset_experiment()`: Reset experiment while keeping folder structure
- `get_experiment_info()`: Detailed information about experiment size and contents

#### **Usage Examples**

**Normal Scan with Stitching:**
```javascript
// Perform a 5x5 grid scan in well A1
await microscopeService.normal_scan_with_stitching({
    start_x_mm: -2.0,  // Relative to well center
    start_y_mm: -2.0,
    Nx: 5, Ny: 5,
    dx_mm: 0.9, dy_mm: 0.9,
    illumination_settings: [
        {'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100},
        {'channel': 'Fluorescence 488 nm Ex', 'intensity': 30, 'exposure_time': 200}
    ],
    wells_to_scan: ['A1'],
    experiment_name: 'fluorescence_scan',
    do_contrast_autofocus: true
});
```

**Quick Scan with Stitching:**
```javascript
// Perform high-speed brightfield scan
await microscopeService.quick_scan_with_stitching({
    wellplate_type: '96',
    exposure_time: 5,
    intensity: 70,
    fps_target: 10,
    n_stripes: 4,
    stripe_width_mm: 4.0,
    velocity_scan_mm_per_s: 7.0
});
```

**Retrieve Stitched Regions:**
```javascript
// Get a 5x5mm region at full resolution
const region = await microscopeService.get_stitched_region({
    center_x_mm: 15.0,
    center_y_mm: 15.0,
    width_mm: 5.0,
    height_mm: 5.0,
    scale_level: 0,  // Full resolution
    channel_name: 'BF LED matrix full',
    output_format: 'base64'  // PNG encoded as base64
});

// Display the image
if (region.success) {
    const imgData = atob(region.data);
    const img = new Image();
    img.src = 'data:image/png;base64,' + region.data;
    document.body.appendChild(img);
}
```

**Experiment Management:**
```javascript
// Create a new experiment
await microscopeService.create_experiment('my_experiment');

// List all experiments
const experiments = await microscopeService.list_experiments();
console.log(`Found ${experiments.total_count} experiments`);

// Set active experiment
await microscopeService.set_active_experiment('my_experiment');

// Get experiment information
const info = await microscopeService.get_experiment_info('my_experiment');
console.log(`Experiment size: ${info.total_size_mb.toFixed(2)} MB`);
```

#### **Configuration and Setup**

**Environment Variables:**
- `ZARR_PATH`: Base directory for zarr storage (default: `/tmp/zarr_canvas`)

**Well Plate Support:**
- **Supported Formats**: 6, 12, 24, 96, 384 well plates
- **Well Naming**: Row letters (A-H) + Column numbers (1-12)
- **Padding**: Configurable padding around each well (default: 2.0mm)

**Channel Mapping:**
- **Brightfield**: 'BF LED matrix full'
- **Fluorescence**: 405nm, 488nm, 561nm, 638nm, 730nm channels
- **Multi-channel**: Simultaneous acquisition and storage

#### **Data Formats and Storage**

**Input Data:**
- Real-time microscope frames with stage position metadata
- Multi-channel image data with exposure and intensity settings
- Stage position coordinates in millimeters

**Storage Format:**
- OME-Zarr format with multi-scale pyramid structure
- Proper metadata including channel information and coordinate transformations
- Chunked storage for efficient access to large datasets

**Output Formats:**
- **Base64 PNG**: Compressed image data for web applications
- **Numpy Arrays**: Raw array data for scientific processing
- **Metadata**: Comprehensive region information and coordinate data

#### **Error Handling and Robustness**

The system includes comprehensive error handling:
- **Graceful Degradation**: Automatic fallback when stage position is unavailable
- **Canvas Boundary Checking**: Prevents writing outside canvas boundaries
- **Service Restart Recovery**: Maintains data integrity across service restarts
- **Comprehensive Logging**: Detailed logging at all levels for debugging
- **Bounds Validation**: Always validates bounds before zarr write operations

#### **Performance Considerations**

**Memory Management:**
- Lazy loading of zarr resources only when needed
- Background processing to avoid blocking main operations
- Automatic cleanup of temporary resources

**I/O Optimization:**
- 256x256 pixel chunks for optimal I/O performance
- Multi-scale pyramid for efficient access at different resolutions
- Thread-safe operations with proper locking mechanisms

**Network Efficiency:**
- Base64 PNG compression for efficient data transfer
- Selective scale updates for quick scan mode
- Chunked data access for large regions

This Zarr Canvas & Image Stitching system provides a powerful foundation for creating comprehensive microscopy datasets with real-time processing and efficient storage, enabling advanced analysis and visualization of large sample areas.