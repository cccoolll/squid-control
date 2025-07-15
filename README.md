# Squid Control

The Squid Control software is a Python package that provides a simple interface to control the Squid microscope. The software is designed to be used with the Squid microscope (made by Cephla Inc.).

## Installation and Usage

See the [installation guide](./docs/installation.md) for instructions on how to install and use the software.

### Installation Options

Basic installation:
```bash
pip install .
```

For development (recommend):
```bash
pip install .[dev]
```

### Usage

To run the software, use the following command:
```bash
python -m squid_control --config HCS_v2
```

If you want to use a different configuration file, you can specify the path to the configuration file:
```
python -m squid_control --config /home/user/configuration_HCS_v2.ini
```

### Simulation Mode

To start simulation mode, use the following command:
```
python -m squid_control --config HCS_v2 --simulation
```

#### Simulated Sample (Zarr-based Virtual Sample)

The simulation mode includes a **virtual microscope sample** using Zarr data archives. This allows you to test the microscope software without a physical sample. The simulated camera retrieves image data based on the current stage position, applies exposure and intensity adjustments, and returns realistic microscopy images.

- The simulated sample consists of Zarr data stored in ZIP files containing high-resolution microscopy images.
- The `Camera_Simulation` class (in `camera_default.py`) handles simulated image acquisition.
- The `ZarrImageManager` retrieves image data from the Zarr archives, either by direct array access or by assembling the region from smaller chunks.
- The image is processed with the requested exposure time, intensity, and optional Z-blurring, then returned to the user.



#### Simulated Sample Features:
- Supports different imaging channels (brightfield and fluorescence)
- Adjustable exposure time and intensity
- Realistic Z-axis blurring for out-of-focus images
- High-resolution sample data covering the stage area

## Zarr Canvas & Image Stitching

The Squid Control system features advanced **Zarr Canvas & Image Stitching** capabilities that enable real-time creation of large field-of-view images from multiple microscope acquisitions. This system provides both normal scanning and quick scanning modes with automatic well-based organization.

### Key Features

#### **Multi-Scale Canvas Architecture**
- **OME-Zarr Compliance**: Full OME-Zarr 0.4 specification support with proper metadata
- **Pyramid Structure**: Multi-scale pyramid with 4x downsampling between levels (scale0=full, scale1=1/4, scale2=1/16, etc.)
- **Optimized Chunking**: 256x256 pixel chunks for efficient I/O performance
- **Memory Efficiency**: Lazy loading and background processing for large datasets

#### **Well-Based Experiment Management**
- **Individual Well Canvases**: Each well gets its own zarr canvas for precise control
- **Experiment Organization**: Hierarchical structure with experiments containing multiple well canvases
- **Automatic Well Detection**: System automatically determines which well contains the current stage position
- **Well-Relative Coordinates**: Each well canvas uses well-center-relative coordinate system

#### **Scanning Modes**

**Normal Scan with Stitching:**
- Grid-based scanning with configurable spacing
- Multi-channel support (brightfield and fluorescence)
- Autofocus integration (contrast and reflection-based)
- Snake pattern scanning for efficiency
- Real-time stitching to OME-Zarr format

**Quick Scan with Stitching:**
- High-speed continuous scanning (up to 10fps)
- Brightfield-only mode with exposure ≤ 30ms
- 4-stripe pattern per well for comprehensive coverage
- Optimized for performance with scale1-5 updates only

#### **API Integration**
The system provides comprehensive Hypha service integration:

```javascript
// Normal scan with stitching
await microscopeService.normal_scan_with_stitching({
    start_x_mm: 10.0,
    start_y_mm: 10.0,
    Nx: 5, Ny: 5,
    dx_mm: 0.9, dy_mm: 0.9,
    illumination_settings: [
        {'channel': 'BF LED matrix full', 'intensity': 50, 'exposure_time': 100}
    ],
    wells_to_scan: ['A1', 'B2', 'C3'],
    experiment_name: 'my_experiment'
});

// Quick scan with stitching
await microscopeService.quick_scan_with_stitching({
    wellplate_type: '96',
    exposure_time: 5,
    intensity: 70,
    fps_target: 10
});

// Retrieve stitched regions
const region = await microscopeService.get_stitched_region({
    center_x_mm: 15.0,
    center_y_mm: 15.0,
    width_mm: 5.0,
    height_mm: 5.0,
    scale_level: 0,
    channel_name: 'BF LED matrix full',
    output_format: 'base64'
});
```

#### **Experiment Management**
- **Create Experiments**: Organize scans into named experiments
- **List Experiments**: View all available experiments and their status
- **Set Active Experiment**: Switch between experiments for data collection
- **Remove/Reset Experiments**: Clean up or reset experiment data
- **Experiment Info**: Detailed information about experiment size and contents

### Technical Architecture

#### **ZarrCanvas Classes**
- **`WellZarrCanvasBase`**: Core stitching functionality with OME-Zarr compliance
- **`WellZarrCanvas`**: Well-specific implementation with automatic coordinate conversion
- **`ExperimentManager`**: Manages experiment folders and well canvas lifecycle

#### **Performance Optimizations**
- **Background Stitching**: Non-blocking frame processing with asyncio queues
- **Thread Safety**: RLock-based concurrent access to zarr arrays
- **Memory Management**: Automatic cleanup and resource management
- **Quick Scan Mode**: Optimized for high-speed acquisition with selective scale updates

#### **Data Formats**
- **Input**: Real-time microscope frames with stage position metadata
- **Storage**: OME-Zarr format with multi-scale pyramid structure
- **Output**: Base64 PNG or numpy arrays for flexible integration
- **Metadata**: Comprehensive channel mapping and coordinate transformation data

### Configuration

#### **Environment Variables**
- `ZARR_PATH`: Base directory for zarr storage (default: `/tmp/zarr_canvas`)

#### **Well Plate Support**
- **Supported Formats**: 6, 12, 24, 96, 384 well plates
- **Well Naming**: Row letters (A-H) + Column numbers (1-12)
- **Padding**: Configurable padding around each well (default: 2.0mm)

For detailed usage examples and API documentation, see the [Feature Introduction](./docs/feature_introduction.md) and [Hypha Tutorial](./docs/hypha_tutorial.md).

---

## About

<img style="width:60px;" src="./docs/assets/cephla_logo.svg"> Cephla Inc. 

---

## Note

The current branch is a fork from https://github.com/hongquanli/octopi-research/ at the following commit:
```
commit dbb49fc314d82d8099d5e509c0e1ad9a919245c9 (HEAD -> master, origin/master, origin/HEAD)
Author: Hongquan Li <hqlisu@gmail.com>
Date:   Thu Apr 4 18:07:51 2024 -0700

    add laser af characterization mode for saving images from laser af camera
```

How to make pypi work:
 - Register on pypi.org
 - Create a new token in the account settings
 - In the repository setting, create a new secret called `PYPI_API_TOKEN` and paste the token in the value field
 - Then, if you want to manually publish a new pypi package, go to actions, select the `Publish to PyPi` workflow, and click on `Run workflow`.

---

**Tip:** For more details on the simulated sample and the Zarr workflow, see [Feature Introduction](./docs/feature_introduction.md).

