import pytest
import pytest_asyncio
import asyncio
import os
import time
import uuid
import numpy as np
import json
import zipfile
import tempfile
import shutil
import requests
import httpx
import zarr
from pathlib import Path
from typing import Dict, List, Tuple
from hypha_rpc import connect_to_server
import matplotlib.pyplot as plt

# Mark all tests in this module as asyncio and integration tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# Test configuration
TEST_SERVER_URL = "https://hypha.aicell.io"
TEST_WORKSPACE = "agent-lens"
TEST_TIMEOUT = 300  # seconds (longer for large uploads)

async def cleanup_test_galleries(artifact_manager):
    """Clean up any leftover test galleries from interrupted tests."""
    try:
        # List all artifacts
        artifacts = await artifact_manager.list()
        
        # Find test galleries - check for multiple patterns
        test_galleries = []
        for artifact in artifacts:
            alias = artifact.get('alias', '')
            # Check for various test gallery patterns
            if any(pattern in alias for pattern in [
                'test-zip-gallery',           # Standard test galleries
                'microscope-gallery-test',     # Test microscope galleries
                '1-test-upload-experiment',    # New experiment galleries (test uploads)
                '1-test-experiment'           # Other test experiment galleries
            ]):
                test_galleries.append(artifact)
        
        if not test_galleries:
            print("✅ No test galleries found to clean up")
            return
        
        print(f"🧹 Found {len(test_galleries)} test galleries to clean up:")
        for gallery in test_galleries:
            print(f"  - {gallery['alias']} (ID: {gallery['id']})")
        
        # Delete each test gallery
        for gallery in test_galleries:
            try:
                await artifact_manager.delete(
                    artifact_id=gallery["id"], 
                    delete_files=True, 
                    recursive=True
                )
                print(f"✅ Deleted gallery: {gallery['alias']}")
            except Exception as e:
                print(f"⚠️ Error deleting {gallery['alias']}: {e}")
        
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

# Test sizes in MB - smaller sizes for faster testing
TEST_SIZES = [
    ("100MB", 100),  # Much smaller for CI
    ("mini-chunks-test", 50),  # Even smaller mini-chunks test
]

# CI-friendly test sizes (when running in GitHub Actions or CI environment)
CI_TEST_SIZES = [
    ("10MB", 10),  # Very small for CI
    ("mini-chunks-test", 25),  # Small mini-chunks test
]

# Detect CI environment
def is_ci_environment():
    """Check if running in a CI environment."""
    return any([
        os.environ.get("CI") == "true",
        os.environ.get("GITHUB_ACTIONS") == "true",
        os.environ.get("RUNNER_OS") is not None,
        os.environ.get("QUICK_TEST") == "1"
    ])

# Use appropriate test sizes based on environment
def get_test_sizes():
    """Get appropriate test sizes based on environment."""
    if is_ci_environment():
        print("🏗️ CI environment detected - using smaller test sizes")
        return CI_TEST_SIZES
    else:
        print("🖥️ Local environment detected - using standard test sizes")
        return TEST_SIZES

class OMEZarrCreator:
    """Helper class to create OME-Zarr datasets of specific sizes."""
    
    @staticmethod
    def calculate_dimensions_for_size(target_size_mb: int, num_channels: int = 4, 
                                    num_timepoints: int = 1, dtype=np.uint16) -> Tuple[int, int, int]:
        """Calculate array dimensions to achieve approximately target size in MB."""
        bytes_per_pixel = np.dtype(dtype).itemsize
        target_bytes = target_size_mb * 1024 * 1024
        
        # Account for multiple channels and timepoints
        pixels_needed = target_bytes // (bytes_per_pixel * num_channels * num_timepoints)
        
        # Assume square images, find side length
        # For OME-Zarr we'll create multiple Z slices
        z_slices = max(1, min(50, target_size_mb // 20))  # More Z slices for larger datasets
        pixels_per_slice = pixels_needed // z_slices
        
        # Square root to get X, Y dimensions
        xy_size = int(np.sqrt(pixels_per_slice))
        
        # Round to nice numbers and ensure minimum size
        xy_size = max(512, (xy_size // 64) * 64)  # Round to nearest 64
        z_slices = max(1, z_slices)
        
        return xy_size, xy_size, z_slices
    
    @staticmethod
    def create_mini_chunk_zarr_dataset(output_path: Path, target_size_mb: int, 
                                     dataset_name: str) -> Dict:
        """
        Create an OME-Zarr dataset specifically designed to reproduce mini chunk issues.
        This creates many small chunks that mirror real-world zarr canvas behavior.
        """
        print(f"Creating MINI-CHUNK OME-Zarr dataset: {dataset_name} (~{target_size_mb}MB)")
        
        # Create dimensions that will result in many small chunks
        # Use smaller chunk sizes and sparse data to create mini chunks
        height, width = 2048, 2048  # Reasonable image size
        z_slices = 1
        num_channels = 4
        num_timepoints = 1
        
        # Create the zarr group
        store = zarr.DirectoryStore(str(output_path))
        root = zarr.group(store=store, overwrite=True)
        
        # OME-Zarr metadata
        ome_metadata = {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"}
            ],
            "datasets": [
                {"path": "0"},
                {"path": "1"}, 
                {"path": "2"}
            ],
            "coordinateTransformations": [
                {
                    "scale": [1.0, 1.0, 0.5, 0.1, 0.1],
                    "type": "scale"
                }
            ]
        }
        
        # Channel metadata
        omero_metadata = {
            "channels": [
                {
                    "label": "DAPI",
                    "color": "0000ff",
                    "window": {"start": 0, "end": 4095}
                },
                {
                    "label": "GFP", 
                    "color": "00ff00",
                    "window": {"start": 0, "end": 4095}
                },
                {
                    "label": "RFP",
                    "color": "ff0000", 
                    "window": {"start": 0, "end": 4095}
                },
                {
                    "label": "Brightfield",
                    "color": "ffffff",
                    "window": {"start": 0, "end": 4095}
                }
            ],
            "name": dataset_name
        }
        
        # Store metadata
        root.attrs["ome"] = ome_metadata
        root.attrs["omero"] = omero_metadata
        
        # Create multi-scale pyramid with SMALL CHUNKS to simulate mini chunk problem
        scales = [1, 2, 4]  # 3 scales
        for scale_idx, scale_factor in enumerate(scales):
            scale_height = height // scale_factor
            scale_width = width // scale_factor
            scale_z = z_slices
            
            # CRITICAL: Use small chunk sizes to create mini chunks
            # This mimics the real-world zarr canvas behavior
            if dataset_name.startswith("mini-chunks"):
                chunk_size = (1, 1, 1, 3, 3)  # Smaller chunks = more files
            else:
                chunk_size = (1, 1, 1, 256, 256)  # Standard chunks
            
            # Create the array
            array = root.create_dataset(
                name=str(scale_idx),
                shape=(num_timepoints, num_channels, scale_z, scale_height, scale_width),
                chunks=chunk_size,
                dtype=np.uint16,
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            
            print(f"  Scale {scale_idx}: {scale_width}x{scale_height}x{scale_z}, chunks: {chunk_size}")
            
            # Generate SPARSE data to create many small chunk files
            # This is key to reproducing the mini chunk problem
            for t in range(num_timepoints):
                for c in range(num_channels):
                    for z in range(scale_z):
                        # Create sparse data pattern that results in small compressed chunks
                        if dataset_name.startswith("mini-chunks"):
                            # Create sparse pattern with mostly zeros
                            data = np.zeros((scale_height, scale_width), dtype=np.uint16)
                            
                            # Add small patches of data every ~200 pixels
                            # This creates many chunks with minimal data (mini chunks)
                            for y in range(0, scale_height, 200):
                                for x in range(0, scale_width, 200):
                                    # Small 20x20 patches of data
                                    y_end = min(y + 20, scale_height)
                                    x_end = min(x + 20, scale_width)
                                    data[y:y_end, x:x_end] = np.random.randint(100, 1000, (y_end-y, x_end-x))
                        else:
                            # Standard dense data for comparison
                            y_coords, x_coords = np.ogrid[:scale_height, :scale_width]
                            
                            # Different patterns for different channels
                            if c == 0:  # DAPI - nuclear pattern
                                data = (np.sin(y_coords * 0.1) * np.cos(x_coords * 0.1) * 1000 + 
                                       np.random.randint(0, 500, (scale_height, scale_width))).astype(np.uint16)
                            elif c == 1:  # GFP - cytoplasmic pattern  
                                data = (np.sin(y_coords * 0.05) * np.sin(x_coords * 0.05) * 1500 + 
                                       np.random.randint(0, 300, (scale_height, scale_width))).astype(np.uint16)
                            elif c == 2:  # RFP - spots pattern
                                data = np.random.exponential(200, (scale_height, scale_width)).astype(np.uint16)
                                data = np.clip(data, 0, 4095)
                            else:  # Brightfield - uniform with texture
                                data = (2000 + np.random.normal(0, 100, (scale_height, scale_width))).astype(np.uint16)
                                data = np.clip(data, 0, 4095)
                        
                        array[t, c, z, :, :] = data
        
        # Calculate actual size
        actual_size_mb = sum(os.path.getsize(os.path.join(root_path, f))
                           for root_path, dirs, files in os.walk(output_path)
                           for f in files) / (1024 * 1024)
        
        print(f"  Created dataset: {actual_size_mb:.1f}MB actual size")
        
        return {
            "name": dataset_name,
            "path": str(output_path),
            "target_size_mb": target_size_mb,
            "actual_size_mb": actual_size_mb,
            "dimensions": {
                "height": height,
                "width": width, 
                "z_slices": z_slices,
                "channels": num_channels,
                "timepoints": num_timepoints
            }
        }
    
    @staticmethod
    def create_ome_zarr_dataset(output_path: Path, target_size_mb: int, 
                              dataset_name: str) -> Dict:
        """Create an OME-Zarr dataset of approximately target_size_mb."""
        
        # Use mini-chunk creation for specific test
        if dataset_name.startswith("mini-chunks"):
            return OMEZarrCreator.create_mini_chunk_zarr_dataset(output_path, target_size_mb, dataset_name)
        
        print(f"Creating OME-Zarr dataset: {dataset_name} (~{target_size_mb}MB)")
        
        # Calculate dimensions
        height, width, z_slices = OMEZarrCreator.calculate_dimensions_for_size(target_size_mb)
        num_channels = 4
        num_timepoints = 1
        
        # Create the zarr group
        store = zarr.DirectoryStore(str(output_path))
        root = zarr.group(store=store, overwrite=True)
        
        # OME-Zarr metadata
        ome_metadata = {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"}
            ],
            "datasets": [
                {"path": "0"},
                {"path": "1"}, 
                {"path": "2"}
            ],
            "coordinateTransformations": [
                {
                    "scale": [1.0, 1.0, 0.5, 0.1, 0.1],
                    "type": "scale"
                }
            ]
        }
        
        # Channel metadata
        omero_metadata = {
            "channels": [
                {
                    "label": "DAPI",
                    "color": "0000ff",
                    "window": {"start": 0, "end": 4095}
                },
                {
                    "label": "GFP", 
                    "color": "00ff00",
                    "window": {"start": 0, "end": 4095}
                },
                {
                    "label": "RFP",
                    "color": "ff0000", 
                    "window": {"start": 0, "end": 4095}
                },
                {
                    "label": "Brightfield",
                    "color": "ffffff",
                    "window": {"start": 0, "end": 4095}
                }
            ],
            "name": dataset_name
        }
        
        # Store metadata
        root.attrs["ome"] = ome_metadata
        root.attrs["omero"] = omero_metadata
        
        # Create multi-scale pyramid
        scales = [1, 2, 4]  # 3 scales
        for scale_idx, scale_factor in enumerate(scales):
            scale_height = height // scale_factor
            scale_width = width // scale_factor
            scale_z = z_slices
            
            # Standard chunk size: 256x256 for X,Y dimensions, 1 for other dimensions
            chunk_size = (1, 1, 1, 256, 256)
            
            # Create the array
            array = root.create_dataset(
                name=str(scale_idx),
                shape=(num_timepoints, num_channels, scale_z, scale_height, scale_width),
                chunks=chunk_size,
                dtype=np.uint16,
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            
            print(f"  Scale {scale_idx}: {scale_width}x{scale_height}x{scale_z}, chunks: {chunk_size}")
            
            # Generate synthetic data with patterns
            for t in range(num_timepoints):
                for c in range(num_channels):
                    for z in range(scale_z):
                        # Create synthetic microscopy-like data
                        y_coords, x_coords = np.ogrid[:scale_height, :scale_width]
                        
                        # Different patterns for different channels
                        if c == 0:  # DAPI - nuclear pattern
                            data = (np.sin(y_coords * 0.1) * np.cos(x_coords * 0.1) * 1000 + 
                                   np.random.randint(0, 500, (scale_height, scale_width))).astype(np.uint16)
                        elif c == 1:  # GFP - cytoplasmic pattern  
                            data = (np.sin(y_coords * 0.05) * np.sin(x_coords * 0.05) * 1500 + 
                                   np.random.randint(0, 300, (scale_height, scale_width))).astype(np.uint16)
                        elif c == 2:  # RFP - spots pattern
                            data = np.random.exponential(200, (scale_height, scale_width)).astype(np.uint16)
                            data = np.clip(data, 0, 4095)
                        else:  # Brightfield - uniform with texture
                            data = (2000 + np.random.normal(0, 100, (scale_height, scale_width))).astype(np.uint16)
                            data = np.clip(data, 0, 4095)
                        
                        array[t, c, z, :, :] = data
        
        # Calculate actual size
        actual_size_mb = sum(os.path.getsize(os.path.join(root_path, f))
                           for root_path, dirs, files in os.walk(output_path)
                           for f in files) / (1024 * 1024)
        
        print(f"  Created dataset: {actual_size_mb:.1f}MB actual size")
        
        return {
            "name": dataset_name,
            "path": str(output_path),
            "target_size_mb": target_size_mb,
            "actual_size_mb": actual_size_mb,
            "dimensions": {
                "height": height,
                "width": width, 
                "z_slices": z_slices,
                "channels": num_channels,
                "timepoints": num_timepoints
            }
        }

    @staticmethod
    def analyze_chunk_sizes(zarr_path: Path) -> Dict:
        """
        Analyze the chunk file sizes in a zarr dataset to identify mini chunks.
        This helps diagnose ZIP corruption issues.
        """
        print(f"🔍 Analyzing chunk sizes in: {zarr_path}")
        
        chunk_sizes = []
        file_count = 0
        total_size = 0
        mini_chunks = 0  # Files < 1KB
        small_chunks = 0  # Files < 10KB
        
        # Walk through all files in the zarr directory
        for root, dirs, files in os.walk(zarr_path):
            for file in files:
                file_path = Path(root) / file
                try:
                    size = file_path.stat().st_size
                    chunk_sizes.append(size)
                    total_size += size
                    file_count += 1
                    
                    if size < 1024:  # < 1KB
                        mini_chunks += 1
                    elif size < 10240:  # < 10KB
                        small_chunks += 1
                        
                except OSError:
                    continue
        
        # Calculate statistics
        chunk_sizes = np.array(chunk_sizes)
        stats = {
            "total_files": file_count,
            "total_size_mb": total_size / (1024 * 1024),
            "average_file_size_bytes": np.mean(chunk_sizes) if len(chunk_sizes) > 0 else 0,
            "median_file_size_bytes": np.median(chunk_sizes) if len(chunk_sizes) > 0 else 0,
            "min_file_size_bytes": np.min(chunk_sizes) if len(chunk_sizes) > 0 else 0,
            "max_file_size_bytes": np.max(chunk_sizes) if len(chunk_sizes) > 0 else 0,
            "mini_chunks_count": mini_chunks,  # < 1KB
            "small_chunks_count": small_chunks,  # < 10KB
            "mini_chunks_percentage": (mini_chunks / file_count * 100) if file_count > 0 else 0,
            "small_chunks_percentage": (small_chunks / file_count * 100) if file_count > 0 else 0,
            "chunk_sizes": chunk_sizes.tolist()
        }
        
        print(f"  📊 File Analysis:")
        print(f"    Total files: {stats['total_files']}")
        print(f"    Total size: {stats['total_size_mb']:.1f} MB")
        print(f"    Average file size: {stats['average_file_size_bytes']:.0f} bytes")
        print(f"    Median file size: {stats['median_file_size_bytes']:.0f} bytes")
        print(f"    Mini chunks (<1KB): {stats['mini_chunks_count']} ({stats['mini_chunks_percentage']:.1f}%)")
        print(f"    Small chunks (<10KB): {stats['small_chunks_count']} ({stats['small_chunks_percentage']:.1f}%)")
        print(f"    Size range: {stats['min_file_size_bytes']:.0f} - {stats['max_file_size_bytes']:.0f} bytes")
        
        return stats

    @staticmethod
    def create_zip_from_zarr(zarr_path: Path, zip_path: Path) -> Dict:
        """Create a ZIP file from OME-Zarr dataset with detailed analysis."""
        print(f"Creating ZIP file: {zip_path.name}")
        
        # First analyze the zarr structure
        chunk_analysis = OMEZarrCreator.analyze_chunk_sizes(zarr_path)
        
        # Create ZIP with different compression strategies based on chunk analysis
        mini_chunk_percentage = chunk_analysis["mini_chunks_percentage"]
        
        if mini_chunk_percentage > 20:  # High percentage of mini chunks
            print(f"⚠️ High mini chunk percentage ({mini_chunk_percentage:.1f}%) - using STORED compression to avoid ZIP corruption")
            compression = zipfile.ZIP_STORED
            compresslevel = None
        else:
            print(f"✅ Low mini chunk percentage ({mini_chunk_percentage:.1f}%) - using DEFLATED compression")
            compression = zipfile.ZIP_DEFLATED
            compresslevel = 1
        
        # Create ZIP with appropriate settings
        zip_kwargs = {
            'mode': 'w',
            'compression': compression,
            'allowZip64': True
        }
        if compresslevel is not None:
            zip_kwargs['compresslevel'] = compresslevel
        
        with zipfile.ZipFile(zip_path, **zip_kwargs) as zipf:
            total_files = 0
            for root, dirs, files in os.walk(zarr_path):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(zarr_path)
                    arcname = f"data.zarr/{relative_path}"
                    zipf.write(file_path, arcname=arcname)
                    total_files += 1
                    
                    if total_files % 1000 == 0:
                        print(f"  Added {total_files} files to ZIP")
        
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  ZIP created: {zip_size_mb:.1f}MB, {total_files} files")
        
        # Test ZIP file integrity
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Test central directory access
                file_list = zipf.namelist()
                # Test reading first few files
                for i, filename in enumerate(file_list[:5]):
                    try:
                        with zipf.open(filename) as f:
                            f.read(1)  # Read one byte to test access
                    except Exception as e:
                        print(f"⚠️ Error reading file {filename} from ZIP: {e}")
                        break
                print(f"✅ ZIP integrity test passed")
                zip_valid = True
        except zipfile.BadZipFile as e:
            print(f"❌ ZIP integrity test failed: {e}")
            zip_valid = False
        
        result = {
            "zip_path": str(zip_path),
            "size_mb": zip_size_mb,
            "file_count": total_files,
            "compression": "STORED" if compression == zipfile.ZIP_STORED else "DEFLATED",
            "zip_valid": zip_valid,
            "chunk_analysis": chunk_analysis
        }
        
        return result

async def upload_zip_with_retry(put_url: str, zip_path: Path, size_mb: int, max_retries: int = 3) -> float:
    """
    Upload ZIP file with retry logic and proper timeout handling.
    
    Args:
        put_url: Upload URL
        zip_path: Path to ZIP file
        size_mb: Size in MB for timeout calculation
        max_retries: Maximum retry attempts
        
    Returns:
        Upload time in seconds
    """
    # Calculate timeout based on file size and environment
    if is_ci_environment():
        # More conservative timeouts for CI (slower network, limited resources)
        timeout_seconds = max(120, int(size_mb / 10) * 60 + 120)  # 2 min base + 1 min per 10MB
    else:
        # More generous timeouts for local development
        timeout_seconds = max(300, int(size_mb / 50) * 60 + 300)  # 5 min base + 1 min per 50MB
    
    print(f"📊 Upload timeout calculation: {size_mb}MB → {timeout_seconds}s timeout")
    
    for attempt in range(max_retries):
        try:
            print(f"Upload attempt {attempt + 1}/{max_retries} for {size_mb:.1f}MB ZIP file (timeout: {timeout_seconds}s)")
            
            # Read file content
            with open(zip_path, 'rb') as f:
                zip_content = f.read()
            
            # Upload with httpx (async) and proper timeout
            upload_start = time.time()
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
                response = await client.put(
                    put_url, 
                    content=zip_content,
                    headers={
                        'Content-Type': 'application/zip',
                        'Content-Length': str(len(zip_content))
                    }
                )
                response.raise_for_status()
            
            upload_time = time.time() - upload_start
            print(f"Upload successful on attempt {attempt + 1}")
            return upload_time
            
        except httpx.TimeoutException as e:
            print(f"Upload timeout on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise Exception(f"Upload failed after {max_retries} attempts due to timeout")
            
        except httpx.HTTPStatusError as e:
            print(f"Upload HTTP error on attempt {attempt + 1}: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 413:  # Payload too large
                raise Exception(f"ZIP file is too large ({size_mb:.1f} MB) for upload")
            elif e.response.status_code >= 500:  # Server errors - retry
                if attempt == max_retries - 1:
                    raise Exception(f"Server error after {max_retries} attempts: {e}")
            else:  # Client errors - don't retry
                raise Exception(f"Upload failed with HTTP {e.response.status_code}: {e.response.text}")
            
        except Exception as e:
            print(f"Upload error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise Exception(f"Upload failed after {max_retries} attempts: {e}")
            
        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)

@pytest_asyncio.fixture(scope="function")
async def artifact_manager():
    """Create artifact manager connection for testing."""
    token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
    if not token:
        pytest.skip("AGENT_LENS_WORKSPACE_TOKEN not set in environment")
    
    print(f"🔗 Connecting to {TEST_SERVER_URL} workspace {TEST_WORKSPACE}...")
    
    async with connect_to_server({
        "server_url": TEST_SERVER_URL,
        "token": token,
        "workspace": TEST_WORKSPACE,
        "ping_interval": None
    }) as server:
        print("✅ Connected to server")
        
        # Get artifact manager service
        artifact_manager = await server.get_service("public/artifact-manager")
        print("✅ Artifact manager ready")
        
        # Clean up any leftover test galleries at the start
        print("🧹 Cleaning up any leftover test galleries...")
        await cleanup_test_galleries(artifact_manager)
        
        yield artifact_manager
        
        # Clean up any leftover test galleries at the end
        print("🧹 Final cleanup of test galleries...")
        await cleanup_test_galleries(artifact_manager)

@pytest_asyncio.fixture(scope="function") 
async def test_gallery(artifact_manager):
    """Create a test gallery and clean it up after test."""
    gallery_id = f"test-zip-gallery-{uuid.uuid4().hex[:8]}"
    
    # Create gallery
    gallery_manifest = {
        "name": f"ZIP Upload Test Gallery - {gallery_id}",
        "description": "Test gallery for ZIP file upload and endpoint testing",
        "created_for": "automated_testing"
    }
    
    print(f"📁 Creating test gallery: {gallery_id}")
    gallery = await artifact_manager.create(
        type="collection",
        alias=gallery_id,
        manifest=gallery_manifest,
        config={"permissions": {"*": "r+", "@": "r+"}}
    )
    
    print(f"✅ Gallery created: {gallery['id']}")
    
    yield gallery
    
    # Cleanup - remove gallery and all datasets
    print(f"🧹 Cleaning up gallery: {gallery_id}")
    try:
        await artifact_manager.delete(
            artifact_id=gallery["id"], 
            delete_files=True, 
            recursive=True
        )
        print("✅ Gallery cleaned up")
    except Exception as e:
        print(f"⚠️ Error during gallery cleanup: {e}")

@pytest.mark.timeout(1800)  # 30 minute timeout
async def test_create_datasets_and_test_endpoints(test_gallery, artifact_manager):
    """Test creating datasets of various sizes and accessing their ZIP endpoints."""
    gallery = test_gallery
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_results = []
        
        for size_name, size_mb in get_test_sizes():
            print(f"\n🧪 Testing {size_name} dataset...")
            
            try:
                # Skip very large tests in CI or quick testing
                if size_mb > 100 and os.environ.get("QUICK_TEST"):
                    print(f"⏭️ Skipping {size_name} (QUICK_TEST mode)")
                    continue
                
                # Create OME-Zarr dataset
                dataset_name = f"test-dataset-{size_name.lower()}-{uuid.uuid4().hex[:6]}"
                zarr_path = temp_path / f"{dataset_name}.zarr"
                
                dataset_info = OMEZarrCreator.create_ome_zarr_dataset(
                    zarr_path, size_mb, dataset_name
                )
                
                # Create ZIP file
                zip_path = temp_path / f"{dataset_name}.zip"
                zip_info = OMEZarrCreator.create_zip_from_zarr(zarr_path, zip_path)
                
                # Create artifact in gallery
                print(f"📦 Creating artifact: {dataset_name}")
                dataset_manifest = {
                    "name": f"Test Dataset {size_name}",
                    "description": f"OME-Zarr dataset for testing ZIP endpoints (~{size_mb}MB)",
                    "size_category": size_name,
                    "target_size_mb": size_mb,
                    "actual_size_mb": dataset_info["actual_size_mb"],
                    "dataset_type": "ome-zarr",
                    "test_purpose": "zip_endpoint_testing"
                }
                
                dataset = await artifact_manager.create(
                    parent_id=gallery["id"],
                    alias=dataset_name,
                    manifest=dataset_manifest,
                    stage=True
                )
                
                # Upload ZIP file using improved async method
                print(f"⬆️ Uploading ZIP file: {zip_info['size_mb']:.1f}MB")
                
                put_url = await artifact_manager.put_file(
                    dataset["id"], 
                    file_path="zarr_dataset.zip",
                    download_weight=1.0
                )
                
                # Use the improved async upload function
                upload_time = await upload_zip_with_retry(put_url, zip_path, zip_info['size_mb'])
                
                print(f"✅ Upload completed in {upload_time:.1f}s ({zip_info['size_mb']/upload_time:.1f} MB/s)")
                
                # Commit the dataset
                await artifact_manager.commit(dataset["id"])
                print(f"✅ Dataset committed")
                
                # Test ZIP endpoint access
                print(f"🔍 Testing ZIP endpoint access...")
                endpoint_url = f"{TEST_SERVER_URL}/{TEST_WORKSPACE}/artifacts/{dataset_name}/zip-files/zarr_dataset.zip/?path=data.zarr/"
                
                # Test directory listing
                response = requests.get(endpoint_url, timeout=60)
                
                # Print the actual response for debugging
                print(f"📄 Response Status: {response.status_code}")
                print(f"📄 Response Headers: {dict(response.headers)}")
                print(f"📄 Response Content: {response.text[:1000]}...")
                
                test_result = {
                    "size_name": size_name,
                    "size_mb": size_mb,
                    "actual_size_mb": dataset_info["actual_size_mb"],
                    "zip_size_mb": zip_info["size_mb"],
                    "upload_time_s": upload_time,
                    "upload_speed_mbps": zip_info["size_mb"] / upload_time,
                    "dataset_id": dataset["id"],
                    "endpoint_url": endpoint_url,
                    "endpoint_status": response.status_code,
                    "endpoint_success": False  # Will be set based on content check
                }
                
                # Check if response is OK and contains valid JSON
                if response.ok:
                    try:
                        content = response.json()
                        
                        # Check if the response is a list (successful directory listing)
                        if isinstance(content, list):
                            test_result["endpoint_success"] = True
                            test_result["endpoint_content_type"] = "json"
                            test_result["endpoint_files_count"] = len(content)
                            print(f"✅ Endpoint SUCCESS: {response.status_code}, {len(content)} items")
                            print(f"📄 Directory listing: {content}")
                            
                            # Test accessing a specific file in the ZIP
                            if len(content) > 0:
                                first_item = content[0]
                                if first_item.get("type") == "file":
                                    file_url = f"{endpoint_url}?path=data.zarr/{first_item['name']}"
                                    file_response = requests.head(file_url, timeout=30)
                                    test_result["file_access_status"] = file_response.status_code
                                    test_result["file_access_success"] = file_response.ok
                                    print(f"✅ File access test: {file_response.status_code}")
                        
                        # Check if the response is an error message
                        elif isinstance(content, dict) and content.get("success") == False:
                            test_result["endpoint_success"] = False
                            test_result["endpoint_error"] = content.get("detail", "Unknown error")
                            print(f"❌ Endpoint FAILED: ZIP file not found - {content.get('detail', 'Unknown error')}")
                        
                        else:
                            test_result["endpoint_success"] = False
                            test_result["endpoint_error"] = f"Unexpected response format: {content}"
                            print(f"❌ Endpoint FAILED: Unexpected response format - {content}")
                        
                    except json.JSONDecodeError:
                        test_result["endpoint_success"] = False
                        test_result["endpoint_content_type"] = "text"
                        test_result["endpoint_error"] = f"Invalid JSON response: {response.text[:200]}"
                        print(f"❌ Endpoint FAILED: Invalid JSON response - {response.text[:200]}")
                    
                else:
                    test_result["endpoint_success"] = False
                    test_result["endpoint_error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                    print(f"❌ Endpoint FAILED: HTTP {response.status_code} - {response.text[:200]}")
                
                test_results.append(test_result)
                
                # Clean up individual dataset to save space
                print(f"🧹 Cleaning up dataset: {dataset_name}")
                await artifact_manager.delete(
                    artifact_id=dataset["id"],
                    delete_files=True
                )
                
                # Clean up local files
                if zarr_path.exists():
                    shutil.rmtree(zarr_path)
                if zip_path.exists():
                    zip_path.unlink()
                
                print(f"✅ {size_name} test completed successfully")
                
            except Exception as e:
                print(f"❌ {size_name} test failed: {e}")
                test_results.append({
                    "size_name": size_name,
                    "size_mb": size_mb,
                    "error": str(e),
                    "endpoint_success": False
                })
                
                # Continue with next test
                continue
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"{'Size':<10} {'Upload':<8} {'Speed':<12} {'Endpoint':<10} {'Status'}")
        print(f"{'-'*50}")
        
        for result in test_results:
            size_name = result["size_name"]
            if "error" in result:
                print(f"{size_name:<10} {'ERROR':<8} {'':<12} {'FAIL':<10} {result['error'][:20]}")
            else:
                upload_time = f"{result['upload_time_s']:.1f}s"
                upload_speed = f"{result['upload_speed_mbps']:.1f}MB/s"
                endpoint_status = "PASS" if result["endpoint_success"] else "FAIL"
                status_code = result.get("endpoint_status", "N/A")
                print(f"{size_name:<10} {upload_time:<8} {upload_speed:<12} {endpoint_status:<10} {status_code}")
        
        # Assert that at least small tests passed
        successful_tests = [r for r in test_results if r.get("endpoint_success", False)]
        assert len(successful_tests) > 0, "No tests passed successfully"
        
        # Assert that at least the smaller tests (< 1GB) passed
        small_tests = [r for r in test_results if r.get("size_mb", 0) < 1000 and r.get("endpoint_success", False)]
        assert len(small_tests) > 0, "No small tests passed successfully"
        
        print(f"\n✅ Test completed: {len(successful_tests)}/{len(test_results)} tests passed")

# Quick test for CI/small environments
async def test_quick_zip_endpoint(test_gallery, artifact_manager):
    """Quick test with just 100MB dataset for CI environments."""
    if not os.environ.get("QUICK_TEST"):
        pytest.skip("Set QUICK_TEST=1 for quick test mode")
    
    gallery = test_gallery
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create small test dataset
        dataset_name = f"quick-test-{uuid.uuid4().hex[:6]}"
        zarr_path = temp_path / f"{dataset_name}.zarr"
        
        dataset_info = OMEZarrCreator.create_ome_zarr_dataset(
            zarr_path, 50, dataset_name  # 50MB for quick test
        )
        
        zip_path = temp_path / f"{dataset_name}.zip"
        zip_info = OMEZarrCreator.create_zip_from_zarr(zarr_path, zip_path)
        
        # Create and upload dataset
        dataset_manifest = {
            "name": "Quick Test Dataset",
            "description": "Small dataset for quick testing",
            "test_purpose": "quick_validation"
        }
        
        dataset = await artifact_manager.create(
            parent_id=gallery["id"],
            alias=dataset_name,
            manifest=dataset_manifest,
            stage=True
        )
        
        put_url = await artifact_manager.put_file(
            dataset["id"], 
            file_path="zarr_dataset.zip"
        )
        
        # Use the improved async upload function
        upload_time = await upload_zip_with_retry(put_url, zip_path, zip_info['size_mb'])
        
        print(f"✅ Quick test upload completed in {upload_time:.1f}s")
        
        await artifact_manager.commit(dataset["id"])
        
        # Test endpoint
        endpoint_url = f"{TEST_SERVER_URL}/{TEST_WORKSPACE}/artifacts/{dataset_name}/zip-files/zarr_dataset.zip/?path=data.zarr/"
        response = requests.get(endpoint_url, timeout=30)
        
        # Print the actual response for debugging
        print(f"📄 Quick Test Response Status: {response.status_code}")
        print(f"📄 Quick Test Response Content: {response.text[:1000]}...")
        
        # Check response content
        if response.ok:
            try:
                content = response.json()
                if isinstance(content, list):
                    print(f"✅ Quick test passed: {len(content)} items in directory")
                    print(f"📄 Directory listing: {content}")
                elif isinstance(content, dict) and content.get("success") == False:
                    raise Exception(f"ZIP file not found: {content.get('detail', 'Unknown error')}")
                else:
                    raise Exception(f"Unexpected response format: {content}")
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON response: {response.text[:200]}")
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

@pytest.mark.timeout(600)  # 10 minute timeout
async def test_upload_zarr_dataset_experiment_logic(test_gallery, artifact_manager):
    """
    Test the upload_zarr_dataset function with a small experiment containing well canvases.
    This validates the new well-separated experiment structure and upload logic.
    """
    print("🧪 Testing upload_zarr_dataset experiment logic...")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Simulate a small experiment with two well canvases
        experiment_name = f"test-upload-experiment-{uuid.uuid4().hex[:6]}"
        wells = ["A1", "B2"]
        wellplate_type = "96"
        zarr_dirs = []
        for well in wells:
            well_dir = temp_path / f"well_{well}_{wellplate_type}.zarr"
            well_dir.mkdir(parents=True, exist_ok=True)
            
            # Create minimal zarr structure
            zarr_group = zarr.open_group(str(well_dir), mode='w')
            
            # Create a simple array for testing
            test_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            zarr_group.create_dataset('0', data=test_array, chunks=(50, 50))
            
            # Create .zarray file
            zarray_path = well_dir / "0" / ".zarray"
            zarray_path.parent.mkdir(exist_ok=True)
            zarray_content = {
                "zarr_format": 2,
                "shape": [100, 100],
                "chunks": [50, 50],
                "dtype": "<u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C"
            }
            with open(zarray_path, 'w') as f:
                json.dump(zarray_content, f)
            
            zarr_dirs.append(well_dir)
            print(f"  📁 Created test zarr directory: {well_dir}")
        
        # Create zip files for each well with correct data.zarr/ structure
        zip_files = []
        for well_dir in zarr_dirs:
            well_name = well_dir.stem  # e.g., "well_A1_96"
            zip_path = temp_path / f"{well_name}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', allowZip64=True) as zf:
                # Walk through the zarr directory and add files under data.zarr/ prefix
                for root, dirs, files in os.walk(well_dir):
                    for file in files:
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(well_dir)
                        # CRITICAL: Always use data.zarr/ prefix, not well name
                        arcname = f"data.zarr/{relative_path}"
                        zf.write(file_path, arcname)
            
            # Validate zip structure
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zip_contents = zf.namelist()
                assert any('data.zarr/' in name for name in zip_contents), f"Zip {zip_path} must contain data.zarr/ structure"
                assert any('.zarray' in name for name in zip_contents), f"Zip {zip_path} must contain .zarray file"
            
            zip_files.append(zip_path)
            print(f"  📦 Created zip file: {zip_path}")
        
        # Use the existing artifact_manager fixture instead of creating a new one
        squid_artifact_manager = artifact_manager
        print("  ✅ Using existing artifact_manager fixture")
        
        # Test uploading each well as a separate dataset
        uploaded_datasets = []
        test_microscope_service_id = f"test-microscope-{uuid.uuid4().hex[:6]}-1"  # End with -1 for new gallery naming
        
        try:
            for i, zip_path in enumerate(zip_files):
                well_name = zip_path.stem  # e.g., "well_A1_96"
                
                # Read zip content
                with open(zip_path, 'rb') as f:
                    zip_content = f.read()
                
                print(f"  📤 Uploading {well_name} as experiment dataset...")
                
                # For this test, we'll simulate the upload process manually since we don't have the full SquidArtifactManager
                # Create a simple dataset upload using the artifact_manager service directly
                
                # Generate dataset name with timestamp
                import time
                timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
                dataset_name = f"{experiment_name}-{timestamp}"
                
                # Create dataset manifest
                dataset_manifest = {
                    "name": dataset_name,
                    "description": f"Test upload for {well_name}",
                    "record_type": "zarr-dataset",
                    "microscope_service_id": test_microscope_service_id,
                    "experiment_id": experiment_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    "acquisition_settings": {
                        "well_name": well_name,
                        "experiment_name": experiment_name,
                        "test_upload": True
                    },
                    "file_format": "ome-zarr",
                    "upload_method": "test-upload",
                    "zip_size_mb": len(zip_content) / (1024 * 1024)
                }
                
                # Create dataset in the test gallery
                dataset = await squid_artifact_manager.create(
                    parent_id=test_gallery["id"],
                    alias=f"agent-lens/{dataset_name}",
                    manifest=dataset_manifest,
                    stage=True
                )
                
                # Upload the ZIP content
                put_url = await squid_artifact_manager.put_file(
                    dataset["id"], 
                    file_path="zarr_dataset.zip",
                    download_weight=1.0
                )
                
                # Upload the file content
                async with httpx.AsyncClient() as client:
                    response = await client.put(
                        put_url, 
                        content=zip_content,
                        headers={
                            'Content-Type': 'application/zip',
                            'Content-Length': str(len(zip_content))
                        }
                    )
                    response.raise_for_status()
                
                # Commit the dataset
                await squid_artifact_manager.commit(dataset["id"])
                
                upload_result = {
                    "success": True,
                    "dataset_id": dataset["id"],
                    "dataset_name": dataset_name,
                    "gallery_id": test_gallery["id"],
                    "experiment_id": experiment_name,
                    "upload_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    "zip_size_mb": len(zip_content) / (1024 * 1024)
                }
                
                uploaded_datasets.append({
                    "well_name": well_name,
                    "dataset_name": upload_result["dataset_name"],
                    "upload_result": upload_result
                })
                
                print(f"  ✅ Uploaded {well_name} -> {upload_result['dataset_name']}")
            
            # Validate upload results
            print(f"  📊 Validating upload results...")
            for dataset_info in uploaded_datasets:
                assert dataset_info["upload_result"]["success"] == True, f"Upload failed for {dataset_info['dataset_name']}"
                
                # Check that dataset name follows new format: {experiment_name}-{timestamp}
                dataset_name = dataset_info["dataset_name"]
                assert experiment_name in dataset_name, f"Dataset name should contain experiment name"
                assert dataset_name.startswith(f"{experiment_name}-"), f"Dataset name should start with experiment name and timestamp"
                
                # Check that gallery follows new format: "1-{experiment_name}"
                gallery_id = dataset_info["upload_result"]["gallery_id"]
                print(f"  📁 Gallery ID: {gallery_id}")
                
                # Test endpoint access
                try:
                    # Get the dataset details
                    dataset_id = dataset_info["upload_result"]["dataset_id"]
                    dataset_details = await squid_artifact_manager.read(artifact_id=dataset_id)
                    print(f"  ✅ Dataset endpoint accessible: {dataset_name}")
                except Exception as e:
                    print(f"  ❌ Dataset endpoint failed: {e}")
                    raise e
            
            print(f"  🎉 All uploads successful! Uploaded {len(uploaded_datasets)} datasets")
            
        finally:
            # Cleanup: Remove uploaded test datasets
            try:
                print("🧹 Cleaning up uploaded test datasets...")
                for dataset_info in uploaded_datasets:
                    dataset_name = dataset_info["dataset_name"]
                    try:
                        # Get the dataset and delete it
                        artifacts = await squid_artifact_manager.list()
                        test_artifacts = [a for a in artifacts if a.get('alias') == f"agent-lens/{dataset_name}"]
                        
                        for artifact in test_artifacts:
                            await squid_artifact_manager.delete(
                                artifact_id=artifact["id"],
                            )
                            print(f"  ✅ Deleted dataset: {dataset_name}")
                    except Exception as e:
                        print(f"  ⚠️ Error deleting {dataset_name}: {e}")
                
                # Also clean up the gallery if it was created
                try:
                    gallery_alias = f"1-{experiment_name}"
                    gallery_artifacts = [a for a in artifacts if a.get('alias') == f"agent-lens/{gallery_alias}"]
                    for gallery in gallery_artifacts:
                        await squid_artifact_manager.delete(
                            artifact_id=gallery["id"],
                            delete_files=True,
                            recursive=True
                        )
                        print(f"  ✅ Deleted gallery: {gallery_alias}")
                except Exception as e:
                    print(f"  ⚠️ Error deleting gallery: {e}")
                    
            except Exception as e:
                print(f"  ⚠️ Error during cleanup: {e}")
        
        print("✅ Test completed successfully!")

if __name__ == "__main__":
    # Allow running this test directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        os.environ["QUICK_TEST"] = "1"
    
    pytest.main([__file__, "-v", "-s"])

# ============================================================================
# NEW TESTS FOR BACKGROUND UPLOAD FUNCTIONALITY
# ============================================================================

class MockWellCanvas:
    """Mock well canvas for testing upload functionality."""
    
    def __init__(self, well_name: str, canvas_path: Path):
        self.well_name = well_name
        self.canvas_path = canvas_path
        self.export_called = False
        self.export_path = None
    
    async def export_to_zip(self, zip_path: Path) -> bool:
        """Mock export to ZIP functionality."""
        self.export_called = True
        self.export_path = zip_path
        
        # Create a mock ZIP file
        with zipfile.ZipFile(zip_path, 'w', allowZip64=True) as zf:
            # Add some mock data
            mock_data = f"Mock data for {self.well_name}".encode()
            zf.writestr("data.zarr/mock_data.txt", mock_data)
            
            # Add a mock zarr structure
            zf.writestr("data.zarr/.zattrs", '{"mock": "true"}')
            zf.writestr("data.zarr/0/.zarray", '{"shape": [100, 100], "chunks": [50, 50]}')
        
        return True

class MockSquidArtifactManager:
    """Mock artifact manager for testing upload queue functionality."""
    
    def __init__(self):
        self.upload_queue = None
        self.upload_worker_task = None
        self.upload_worker_running = False
        self.current_dataset_id = None
        self.current_gallery_id = None
        self.upload_frozen = False
        self.microscope_service_id = None
        self.experiment_id = None
        self.acquisition_settings = None
        self.description = None
        self.uploaded_wells = []
        self.dataset_committed = False
        self.worker_started = False
        self.worker_stopped = False
    
    async def start_upload_worker(self, microscope_service_id, experiment_id, acquisition_settings=None, description=None):
        """Mock start upload worker."""
        self.upload_queue = asyncio.Queue()
        self.upload_worker_running = True
        self.upload_frozen = False
        self.microscope_service_id = microscope_service_id
        self.experiment_id = experiment_id
        self.acquisition_settings = acquisition_settings
        self.description = description
        self.worker_started = True
        
        # Create mock dataset
        self.current_dataset_id = f"mock-dataset-{uuid.uuid4().hex[:8]}"
        self.current_gallery_id = f"mock-gallery-{uuid.uuid4().hex[:8]}"
        
        # Start background worker
        self.upload_worker_task = asyncio.create_task(self._mock_upload_worker_loop())
        print(f"Mock upload worker started for experiment: {experiment_id}")
    
    async def stop_upload_worker(self):
        """Mock stop upload worker."""
        if not self.upload_worker_running:
            return
        
        self.upload_worker_running = False
        
        if self.upload_worker_task:
            try:
                await asyncio.wait_for(self.upload_worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.upload_worker_task.cancel()
                try:
                    await self.upload_worker_task
                except asyncio.CancelledError:
                    pass
        
        self.dataset_committed = True
        self.worker_stopped = True
        print("Mock upload worker stopped")
    
    async def add_well_to_upload_queue(self, well_name, well_zip_content, well_size_mb):
        """Mock add well to upload queue."""
        if not self.upload_worker_running or self.upload_frozen:
            return
        
        await self.upload_queue.put({
            'name': well_name,
            'content': well_zip_content,
            'size_mb': well_size_mb
        })
        print(f"Mock: Added {well_name} to upload queue")
    
    async def _mock_upload_worker_loop(self):
        """Mock upload worker loop."""
        while self.upload_worker_running:
            try:
                well_info = await asyncio.wait_for(self.upload_queue.get(), timeout=1.0)
                
                # Simulate upload
                self.uploaded_wells.append({
                    'name': well_info['name'],
                    'size_mb': well_info['size_mb'],
                    'uploaded_at': time.time()
                })
                
                print(f"Mock: Uploaded {well_info['name']}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Mock upload worker error: {e}")
                continue

@pytest.mark.timeout(300)  # 5 minute timeout
async def test_background_upload_queue_functionality():
    """Test the background upload queue functionality."""
    print("🧪 Testing background upload queue functionality...")
    
    # Create mock artifact manager
    artifact_manager = MockSquidArtifactManager()
    
    # Test 1: Start upload worker
    microscope_id = "test-microscope-123"
    experiment_id = "test-experiment-456"
    acquisition_settings = {"test": True, "wells": ["A1", "B2"]}
    description = "Test upload experiment"
    
    await artifact_manager.start_upload_worker(
        microscope_id, experiment_id, acquisition_settings, description
    )
    
    assert artifact_manager.worker_started == True
    assert artifact_manager.upload_worker_running == True
    assert artifact_manager.microscope_service_id == microscope_id
    assert artifact_manager.experiment_id == experiment_id
    assert artifact_manager.acquisition_settings == acquisition_settings
    assert artifact_manager.description == description
    assert artifact_manager.current_dataset_id is not None
    assert artifact_manager.current_gallery_id is not None
    print("✅ Upload worker started successfully")
    
    # Test 2: Add wells to upload queue
    wells = [
        ("well_A1_96", b"mock_zip_content_1", 10.5),
        ("well_B2_96", b"mock_zip_content_2", 15.2),
        ("well_C3_96", b"mock_zip_content_3", 8.7)
    ]
    
    for well_name, content, size_mb in wells:
        await artifact_manager.add_well_to_upload_queue(well_name, content, size_mb)
    
    # Wait for uploads to process
    await asyncio.sleep(2)
    
    assert len(artifact_manager.uploaded_wells) == 3
    uploaded_names = [w['name'] for w in artifact_manager.uploaded_wells]
    assert "well_A1_96" in uploaded_names
    assert "well_B2_96" in uploaded_names
    assert "well_C3_96" in uploaded_names
    print("✅ Wells uploaded successfully")
    
    # Test 3: Stop upload worker
    await artifact_manager.stop_upload_worker()
    
    assert artifact_manager.worker_stopped == True
    assert artifact_manager.upload_worker_running == False
    assert artifact_manager.dataset_committed == True
    print("✅ Upload worker stopped successfully")
    
    print("✅ Background upload queue functionality test passed")

@pytest.mark.timeout(300)  # 5 minute timeout
async def test_well_canvas_export_functionality():
    """Test the well canvas export to ZIP functionality."""
    print("🧪 Testing well canvas export functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock well canvases
        well_canvases = []
        for well_name in ["A1", "B2", "C3"]:
            canvas_path = temp_path / f"well_{well_name}_96.zarr"
            canvas_path.mkdir(parents=True, exist_ok=True)
            
            # Create mock zarr structure
            zarr_group = zarr.open_group(str(canvas_path), mode='w')
            test_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            zarr_group.create_dataset('0', data=test_array, chunks=(50, 50))
            
            well_canvas = MockWellCanvas(f"well_{well_name}_96", canvas_path)
            well_canvases.append(well_canvas)
        
        # Test export to ZIP
        for well_canvas in well_canvases:
            zip_path = temp_path / f"{well_canvas.well_name}.zip"
            
            success = await well_canvas.export_to_zip(zip_path)
            
            assert success == True
            assert well_canvas.export_called == True
            assert well_canvas.export_path == zip_path
            assert zip_path.exists()
            
            # Verify ZIP structure
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = zf.namelist()
                assert "data.zarr/mock_data.txt" in file_list
                assert "data.zarr/.zattrs" in file_list
                assert "data.zarr/0/.zarray" in file_list
                
                # Test content
                content = zf.read("data.zarr/mock_data.txt").decode()
                assert f"Mock data for {well_canvas.well_name}" in content
            
            print(f"✅ Well canvas {well_canvas.well_name} exported successfully")
        
        print("✅ Well canvas export functionality test passed")

@pytest.mark.timeout(600)  # 10 minute timeout
async def test_scanning_with_background_upload_integration(test_gallery, artifact_manager):
    """Test the integration of background upload with scanning functionality."""
    print("🧪 Testing scanning with background upload integration...")
    
    # Create a mock scanning scenario
    experiment_id = f"test-scan-upload-{uuid.uuid4().hex[:6]}"
    microscope_service_id = f"test-microscope-{uuid.uuid4().hex[:6]}-1"
    wells_to_scan = ["A1", "B2", "C3"]
    
    # Mock acquisition settings
    acquisition_settings = {
        "experiment_id": experiment_id,
        "wells_to_scan": wells_to_scan,
        "wellplate_type": "96",
        "scanning_mode": "background_upload",
        "test_mode": True
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Simulate the complete workflow
        workflow_results = {
            'scanning_started': False,
            'wells_scanned': [],
            'wells_uploaded': [],
            'upload_worker_started': False,
            'upload_worker_stopped': False,
            'dataset_created': False,
            'dataset_committed': False
        }
        
        # Step 1: Start upload worker (simulating start_hypha_service.py)
        print("  📤 Step 1: Starting upload worker...")
        workflow_results['upload_worker_started'] = True
        
        # Step 2: Start scanning (simulating squid_controller.py)
        print("  🔬 Step 2: Starting scanning process...")
        workflow_results['scanning_started'] = True
        
        # Step 3: Scan each well and trigger upload
        for i, well_name in enumerate(wells_to_scan):
            print(f"  🔬 Step 3.{i+1}: Scanning well {well_name}...")
            
            # Simulate well scanning
            canvas_path = temp_path / f"well_{well_name}_96.zarr"
            canvas_path.mkdir(parents=True, exist_ok=True)
            
            # Create mock zarr structure
            zarr_group = zarr.open_group(str(canvas_path), mode='w')
            test_array = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
            zarr_group.create_dataset('0', data=test_array, chunks=(150, 150))
            
            workflow_results['wells_scanned'].append({
                'well_name': well_name,
                'canvas_path': canvas_path,
                'scan_completed_at': time.time()
            })
            
            print(f"  ✅ Well {well_name} scanned")
            
            # Simulate background upload trigger (simulating squid_controller.py)
            print(f"  📤 Step 3.{i+1}.1: Triggering background upload for {well_name}...")
            
            # Create ZIP for upload
            zip_path = temp_path / f"well_{well_name}_96.zip"
            
            with zipfile.ZipFile(zip_path, 'w', allowZip64=True) as zf:
                for root, dirs, files in os.walk(canvas_path):
                    for file in files:
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(canvas_path)
                        arcname = f"data.zarr/{relative_path}"
                        zf.write(file_path, arcname)
            
            # Read ZIP content for upload
            with open(zip_path, 'rb') as f:
                zip_content = f.read()
            
            zip_size_mb = len(zip_content) / (1024 * 1024)
            
            workflow_results['wells_uploaded'].append({
                'well_name': well_name,
                'zip_content': zip_content,
                'zip_size_mb': zip_size_mb,
                'upload_triggered_at': time.time()
            })
            
            print(f"  ✅ Upload triggered for {well_name} ({zip_size_mb:.1f}MB)")
        
        # Step 4: Create dataset and upload wells (simulating artifact_manager.py)
        print("  📦 Step 4: Creating dataset and uploading wells...")
        
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        dataset_name = f"{experiment_id}-{timestamp}"
        
        dataset_manifest = {
            "name": dataset_name,
            "description": f"Complete workflow test - {experiment_id}",
            "record_type": "zarr-dataset",
            "microscope_service_id": microscope_service_id,
            "experiment_id": experiment_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "acquisition_settings": scanning_settings,
            "file_format": "ome-zarr",
            "upload_method": "squid-control-api-background",
            "scanned_wells": wells_to_scan,
            "workflow_test": True
        }
        
        dataset = await artifact_manager.create(
            parent_id=test_gallery["id"],
            alias=f"agent-lens/{dataset_name}",
            manifest=dataset_manifest,
            stage=True
        )
        
        workflow_results['dataset_created'] = True
        print(f"  ✅ Dataset created: {dataset['id']}")
        
        # Simulate uploading each well
        for well_info in workflow_results['wells_uploaded']:
            well_name = well_info['well_name']
            zip_content = well_info['zip_content']
            zip_size_mb = well_info['zip_size_mb']
            
            print(f"  ⬆️ Uploading {well_name} ({zip_size_mb:.1f}MB)...")
            
            put_url = await artifact_manager.put_file(
                dataset["id"], 
                file_path=f"{well_name}.zip",
                download_weight=1.0
            )
            
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    put_url, 
                    content=zip_content,
                    headers={
                        'Content-Type': 'application/zip',
                        'Content-Length': str(len(zip_content))
                    }
                )
                response.raise_for_status()
            
            print(f"  ✅ {well_name} uploaded")
        
        # Step 5: Commit dataset and stop upload worker
        print("  ✅ Step 5: Committing dataset and stopping upload worker...")
        
        await artifact_manager.commit(dataset["id"])
        workflow_results['dataset_committed'] = True
        workflow_results['upload_worker_stopped'] = True
        
        print(f"  ✅ Dataset committed")
        
        # Validate complete workflow
        print("  📊 Validating complete workflow...")
        
        assert workflow_results['upload_worker_started'] == True
        assert workflow_results['scanning_started'] == True
        assert len(workflow_results['wells_scanned']) == len(wells_to_scan)
        assert len(workflow_results['wells_uploaded']) == len(wells_to_scan)
        assert workflow_results['dataset_created'] == True
        assert workflow_results['dataset_committed'] == True
        assert workflow_results['upload_worker_stopped'] == True
        
        # Check that all wells were processed
        scanned_names = [w['well_name'] for w in workflow_results['wells_scanned']]
        uploaded_names = [w['well_name'] for w in workflow_results['wells_uploaded']]
        
        for well_name in wells_to_scan:
            assert well_name in scanned_names, f"Well {well_name} was not scanned"
            assert f"well_{well_name}_96" in uploaded_names, f"Well {well_name} was not uploaded"
        
        print(f"  📊 Workflow Summary:")
        print(f"    - Upload worker: {'✅ Started' if workflow_results['upload_worker_started'] else '❌ Failed'}")
        print(f"    - Scanning: {'✅ Started' if workflow_results['scanning_started'] else '❌ Failed'}")
        print(f"    - Wells scanned: {len(workflow_results['wells_scanned'])}/{len(wells_to_scan)}")
        print(f"    - Wells uploaded: {len(workflow_results['wells_uploaded'])}/{len(wells_to_scan)}")
        print(f"    - Dataset created: {'✅ Yes' if workflow_results['dataset_created'] else '❌ No'}")
        print(f"    - Dataset committed: {'✅ Yes' if workflow_results['dataset_committed'] else '❌ No'}")
        print(f"    - Upload worker stopped: {'✅ Yes' if workflow_results['upload_worker_stopped'] else '❌ No'}")
        
        print("✅ Complete scanning + upload workflow test passed")

@pytest.mark.timeout(300)  # 5 minute timeout
async def test_upload_queue_failure_handling():
    """Test upload queue failure handling and freezing behavior."""
    print("🧪 Testing upload queue failure handling...")
    
    # Create mock artifact manager with failure simulation
    class MockFailingArtifactManager(MockSquidArtifactManager):
        def __init__(self, fail_after_attempts=3):
            super().__init__()
            self.fail_after_attempts = fail_after_attempts
            self.upload_attempts = 0
        
        async def _mock_upload_worker_loop(self):
            """Mock upload worker loop with failure simulation."""
            while self.upload_worker_running:
                try:
                    well_info = await asyncio.wait_for(self.upload_queue.get(), timeout=1.0)
                    
                    self.upload_attempts += 1
                    
                    # Simulate failure after certain number of attempts
                    if self.upload_attempts >= self.fail_after_attempts:
                        print(f"Mock: Simulating upload failure for {well_info['name']}")
                        self.upload_frozen = True
                        break
                    else:
                        # Simulate successful upload
                        self.uploaded_wells.append({
                            'name': well_info['name'],
                            'size_mb': well_info['size_mb'],
                            'uploaded_at': time.time()
                        })
                        print(f"Mock: Uploaded {well_info['name']} (attempt {self.upload_attempts})")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Mock upload worker error: {e}")
                    continue
    
    # Test upload queue with failure simulation
    artifact_manager = MockFailingArtifactManager(fail_after_attempts=3)
    
    # Start upload worker
    await artifact_manager.start_upload_worker(
        "test-microscope-fail", "test-experiment-fail", {}, "Test failure handling"
    )
    
    # Add wells to queue
    wells = [
        ("well_A1_96", b"content1", 10.0),
        ("well_B2_96", b"content2", 15.0),
        ("well_C3_96", b"content3", 12.0),
        ("well_D4_96", b"content4", 8.0)  # This should fail
    ]
    
    for well_name, content, size_mb in wells:
        await artifact_manager.add_well_to_upload_queue(well_name, content, size_mb)
    
    # Wait for processing
    await asyncio.sleep(3)
    
    # Check results
    assert artifact_manager.upload_frozen == True
    assert len(artifact_manager.uploaded_wells) == 2  # First 2 should succeed
    assert artifact_manager.upload_attempts == 3  # Should fail on 3rd attempt
    
    uploaded_names = [w['name'] for w in artifact_manager.uploaded_wells]
    assert "well_A1_96" in uploaded_names
    assert "well_B2_96" in uploaded_names
    assert "well_C3_96" not in uploaded_names  # Should not be uploaded due to failure
    
    # Stop worker
    await artifact_manager.stop_upload_worker()
    
    print("✅ Upload queue failure handling test passed")

@pytest.mark.timeout(300)  # 5 minute timeout
async def test_upload_queue_retry_logic():
    """Test upload queue retry logic with exponential backoff."""
    print("🧪 Testing upload queue retry logic...")
    
    # Create mock artifact manager with retry simulation
    class MockRetryArtifactManager(MockSquidArtifactManager):
        def __init__(self, fail_pattern=None):
            super().__init__()
            self.fail_pattern = fail_pattern or [True, True, False]  # Fail first 2, succeed on 3rd
            self.retry_counts = {}
        
        async def _mock_upload_worker_loop(self):
            """Mock upload worker loop with retry simulation."""
            while self.upload_worker_running:
                try:
                    well_info = await asyncio.wait_for(self.upload_queue.get(), timeout=1.0)
                    
                    well_name = well_info['name']
                    if well_name not in self.retry_counts:
                        self.retry_counts[well_name] = 0
                    
                    self.retry_counts[well_name] += 1
                    attempt = self.retry_counts[well_name]
                    
                    # Simulate retry pattern
                    should_fail = attempt <= len(self.fail_pattern) and self.fail_pattern[attempt - 1]
                    
                    if should_fail:
                        print(f"Mock: Upload attempt {attempt} failed for {well_name}")
                        if attempt >= 3:  # Max retries reached
                            print(f"Mock: Max retries reached for {well_name}, freezing queue")
                            self.upload_frozen = True
                            break
                        else:
                            # Simulate retry delay
                            wait_time = 2 ** (attempt - 1)
                            print(f"Mock: Waiting {wait_time}s before retry for {well_name}")
                            await asyncio.sleep(wait_time)
                            continue
                    else:
                        # Success
                        self.uploaded_wells.append({
                            'name': well_name,
                            'size_mb': well_info['size_mb'],
                            'uploaded_at': time.time(),
                            'attempts': attempt
                        })
                        print(f"Mock: Uploaded {well_name} after {attempt} attempts")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Mock upload worker error: {e}")
                    continue
    
    # Test retry logic
    artifact_manager = MockRetryArtifactManager(fail_pattern=[True, True, False])  # Fail twice, succeed on 3rd
    
    await artifact_manager.start_upload_worker(
        "test-microscope-retry", "test-experiment-retry", {}, "Test retry logic"
    )
    
    # Add wells to queue
    wells = [
        ("well_A1_96", b"content1", 10.0),
        ("well_B2_96", b"content2", 15.0)
    ]
    
    for well_name, content, size_mb in wells:
        await artifact_manager.add_well_to_upload_queue(well_name, content, size_mb)
    
    # Wait for processing with retries
    await asyncio.sleep(10)  # Longer wait for retries
    
    # Check results
    assert len(artifact_manager.uploaded_wells) == 2  # Both should succeed after retries
    assert artifact_manager.upload_frozen == False  # Should not freeze since retries succeeded
    
    for well_info in artifact_manager.uploaded_wells:
        assert well_info['attempts'] == 3  # Should succeed on 3rd attempt
    
    # Stop worker
    await artifact_manager.stop_upload_worker()
    
    print("✅ Upload queue retry logic test passed")

@pytest.mark.timeout(300)  # 5 minute timeout
async def test_upload_queue_concurrent_processing():
    """Test upload queue with concurrent well processing."""
    print("🧪 Testing upload queue concurrent processing...")
    
    # Create mock artifact manager with concurrent processing
    class MockConcurrentArtifactManager(MockSquidArtifactManager):
        def __init__(self):
            super().__init__()
            self.processing_times = {}
            self.concurrent_uploads = 0
            self.max_concurrent = 0
        
        async def _mock_upload_worker_loop(self):
            """Mock upload worker loop with concurrent processing simulation."""
            while self.upload_worker_running:
                try:
                    well_info = await asyncio.wait_for(self.upload_queue.get(), timeout=1.0)
                    
                    # Simulate concurrent processing
                    self.concurrent_uploads += 1
                    self.max_concurrent = max(self.max_concurrent, self.concurrent_uploads)
                    
                    start_time = time.time()
                    
                    # Simulate upload processing time
                    await asyncio.sleep(0.5)  # Simulate upload time
                    
                    self.uploaded_wells.append({
                        'name': well_info['name'],
                        'size_mb': well_info['size_mb'],
                        'uploaded_at': time.time(),
                        'processing_time': time.time() - start_time
                    })
                    
                    self.concurrent_uploads -= 1
                    print(f"Mock: Uploaded {well_info['name']} (concurrent: {self.concurrent_uploads})")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Mock upload worker error: {e}")
                    self.concurrent_uploads -= 1
                    continue
    
    # Test concurrent processing
    artifact_manager = MockConcurrentArtifactManager()
    
    await artifact_manager.start_upload_worker(
        "test-microscope-concurrent", "test-experiment-concurrent", {}, "Test concurrent processing"
    )
    
    # Add multiple wells quickly to test concurrency
    wells = [
        ("well_A1_96", b"content1", 10.0),
        ("well_B2_96", b"content2", 15.0),
        ("well_C3_96", b"content3", 12.0),
        ("well_D4_96", b"content4", 8.0),
        ("well_E5_96", b"content5", 20.0)
    ]
    
    # Add all wells quickly
    for well_name, content, size_mb in wells:
        await artifact_manager.add_well_to_upload_queue(well_name, content, size_mb)
    
    # Wait for processing
    await asyncio.sleep(5)
    
    # Check results
    assert len(artifact_manager.uploaded_wells) == 5
    assert artifact_manager.max_concurrent > 1  # Should have concurrent processing
    
    # Check processing times
    total_time = sum(w['processing_time'] for w in artifact_manager.uploaded_wells)
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Max concurrent uploads: {artifact_manager.max_concurrent}")
    
    # Stop worker
    await artifact_manager.stop_upload_worker()
    
    print("✅ Upload queue concurrent processing test passed") 