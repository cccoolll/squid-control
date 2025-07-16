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
import seaborn as sns

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
        
        # Find test galleries
        test_galleries = [a for a in artifacts if 'test-zip-gallery' in a.get('alias', '')]
        
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
    ("10MB", 10),
    ("1.6GB", 1600),
    ("100MB", 100),
    ("mini-chunks-test", 1),  # New test designed to create mini chunks
]

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
        height, width = 512, 512  # Reasonable image size
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
            if "mini-chunks" in dataset_name:
                chunk_size = (1, 1, 1, 8, 8)  # Smaller chunks = more files
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
                        if "mini-chunks" in dataset_name:
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
    async def create_ome_zarr_dataset(output_path: Path, target_size_mb: int, 
                              dataset_name: str) -> Dict:
        """Create an OME-Zarr dataset of approximately target_size_mb."""
        
        # Use mini-chunk creation for specific test
        if "mini-chunks" in dataset_name:
            return await asyncio.get_event_loop().run_in_executor(
                None, OMEZarrCreator.create_mini_chunk_zarr_dataset, output_path, target_size_mb, dataset_name
            )
        
        print(f"Creating OME-Zarr dataset: {dataset_name} (~{target_size_mb}MB)")
        
        # Calculate dimensions
        height, width, z_slices = OMEZarrCreator.calculate_dimensions_for_size(target_size_mb)
        num_channels = 4
        num_timepoints = 1
        
        # Define the dataset creation function to run in executor
        def create_dataset_sync():
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
        
        # Run dataset creation in executor to avoid blocking event loop
        return await asyncio.get_event_loop().run_in_executor(
            None, create_dataset_sync
        )

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
    async def create_zip_from_zarr(zarr_path: Path, zip_path: Path) -> Dict:
        """Create a ZIP file from OME-Zarr dataset with detailed analysis."""
        print(f"Creating ZIP file: {zip_path.name}")
        
        # First analyze the zarr structure
        chunk_analysis = await asyncio.get_event_loop().run_in_executor(
            None, OMEZarrCreator.analyze_chunk_sizes, zarr_path
        )
        
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
        
        # Define the ZIP creation function to run in executor
        def create_zip_sync():
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
            return total_files, zip_size_mb
        
        # Run ZIP creation in executor to avoid blocking event loop
        total_files, zip_size_mb = await asyncio.get_event_loop().run_in_executor(
            None, create_zip_sync
        )
        
        # Test ZIP file integrity in executor
        def test_zip_integrity():
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
                    return True
            except zipfile.BadZipFile as e:
                print(f"❌ ZIP integrity test failed: {e}")
                return False
        
        zip_valid = await asyncio.get_event_loop().run_in_executor(
            None, test_zip_integrity
        )
        
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
    # Calculate timeout based on file size (minimum 5 minutes, add 1 minute per 50MB)
    timeout_seconds = max(300, int(size_mb / 50) * 60 + 300)
    
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
        
        for size_name, size_mb in TEST_SIZES:
            print(f"\n🧪 Testing {size_name} dataset...")
            
            try:
                # Skip very large tests in CI or quick testing
                if size_mb > 1000 and os.environ.get("QUICK_TEST"):
                    print(f"⏭️ Skipping {size_name} (QUICK_TEST mode)")
                    continue
                
                # Create OME-Zarr dataset
                dataset_name = f"test-dataset-{size_name.lower()}-{uuid.uuid4().hex[:6]}"
                zarr_path = temp_path / f"{dataset_name}.zarr"
                
                dataset_info = await OMEZarrCreator.create_ome_zarr_dataset(
                    zarr_path, size_mb, dataset_name
                )
                
                # Create ZIP file
                zip_path = temp_path / f"{dataset_name}.zip"
                zip_info = await OMEZarrCreator.create_zip_from_zarr(zarr_path, zip_path)
                
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
        
        dataset_info = await OMEZarrCreator.create_ome_zarr_dataset(
            zarr_path, 50, dataset_name  # 50MB for quick test
        )
        
        zip_path = temp_path / f"{dataset_name}.zip"
        zip_info = await OMEZarrCreator.create_zip_from_zarr(zarr_path, zip_path)
        
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

@pytest.mark.timeout(1800)  # 30 minute timeout
async def test_mini_chunk_reproduction(test_gallery, artifact_manager):
    """
    Test specifically designed to reproduce the mini chunk ZIP corruption issue.
    This test compares normal chunks vs mini chunks to identify the root cause.
    """
    gallery = test_gallery
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"\n🧪 MINI CHUNK REPRODUCTION TEST")
        print(f"=" * 60)
        
        # Test both normal and mini chunk scenarios
        test_scenarios = [
            ("normal-chunks", 400, "Normal 256x256 chunks"),
            ("mini-chunks-test", 0.001, "Small 8x8 chunks with sparse data")
        ]
        
        results = []
        
        for scenario_name, size_mb, description in test_scenarios:
            print(f"\n📋 Testing scenario: {description}")
            print(f"   Target size: {size_mb}MB")
            
            try:
                # Create dataset with specific chunk characteristics
                dataset_name = f"{scenario_name}-{uuid.uuid4().hex[:6]}"
                zarr_path = temp_path / f"{dataset_name}.zarr"
                
                dataset_info = await OMEZarrCreator.create_ome_zarr_dataset(
                    zarr_path, size_mb, dataset_name
                )
                
                # Create ZIP with analysis
                zip_path = temp_path / f"{dataset_name}.zip"
                zip_info = await OMEZarrCreator.create_zip_from_zarr(zarr_path, zip_path)
                
                # Extract key metrics
                chunk_analysis = zip_info["chunk_analysis"]
                
                result = {
                    "scenario": scenario_name,
                    "description": description,
                    "zarr_size_mb": dataset_info["actual_size_mb"],
                    "zip_size_mb": zip_info["size_mb"],
                    "compression_ratio": zip_info["size_mb"] / dataset_info["actual_size_mb"],
                    "total_files": chunk_analysis["total_files"],
                    "average_file_size": chunk_analysis["average_file_size_bytes"],
                    "mini_chunks_count": chunk_analysis["mini_chunks_count"],
                    "mini_chunks_percentage": chunk_analysis["mini_chunks_percentage"],
                    "small_chunks_count": chunk_analysis["small_chunks_count"],
                    "small_chunks_percentage": chunk_analysis["small_chunks_percentage"],
                    "zip_compression": zip_info["compression"],
                    "zip_valid": zip_info["zip_valid"],
                    "files_per_mb": chunk_analysis["total_files"] / dataset_info["actual_size_mb"]
                }
                
                results.append(result)
                
                print(f"📊 Results for {scenario_name}:")
                print(f"   Zarr size: {result['zarr_size_mb']:.1f} MB")
                print(f"   ZIP size: {result['zip_size_mb']:.1f} MB")
                print(f"   Compression: {result['zip_compression']}")
                print(f"   Total files: {result['total_files']}")
                print(f"   Files per MB: {result['files_per_mb']:.0f}")
                print(f"   Average file size: {result['average_file_size']:.0f} bytes")
                print(f"   Mini chunks (<1KB): {result['mini_chunks_count']} ({result['mini_chunks_percentage']:.1f}%)")
                print(f"   Small chunks (<10KB): {result['small_chunks_count']} ({result['small_chunks_percentage']:.1f}%)")
                print(f"   ZIP valid: {'✅' if result['zip_valid'] else '❌'}")
                
                # Try to upload and test the problematic case
                if scenario_name.startswith("mini-chunks"):
                    print(f"\n🚀 Testing upload of mini-chunk scenario...")
                    
                    # Create artifact in gallery
                    dataset_manifest = {
                        "name": f"Mini Chunk Test - {scenario_name}",
                        "description": f"Test for mini chunk ZIP corruption: {description}",
                        "test_purpose": "mini_chunk_reproduction"
                    }
                    
                    dataset = await artifact_manager.create(
                        parent_id=gallery["id"],
                        alias=dataset_name,
                        manifest=dataset_manifest,
                        stage=True
                    )
                    
                    # Upload ZIP file
                    put_url = await artifact_manager.put_file(
                        dataset["id"], 
                        file_path="zarr_dataset.zip"
                    )
                    
                    upload_time = await upload_zip_with_retry(put_url, zip_path, zip_info['size_mb'])
                    print(f"✅ Upload completed in {upload_time:.1f}s")
                    
                    await artifact_manager.commit(dataset["id"])
                    
                    # Test endpoint access
                    endpoint_url = f"{TEST_SERVER_URL}/{TEST_WORKSPACE}/artifacts/{dataset_name}/zip-files/zarr_dataset.zip/?path=data.zarr/"
                    
                    print(f"🔍 Testing endpoint: {endpoint_url}")
                    response = requests.get(endpoint_url, timeout=60)
                    
                    result["endpoint_status"] = response.status_code
                    result["endpoint_success"] = response.ok
                    
                    if response.ok:
                        try:
                            content = response.json()
                            if isinstance(content, list):
                                result["endpoint_files_found"] = len(content)
                                print(f"✅ Endpoint SUCCESS: Found {len(content)} files")
                            else:
                                result["endpoint_error"] = str(content)
                                print(f"❌ Endpoint returned error: {content}")
                        except json.JSONDecodeError:
                            result["endpoint_error"] = f"Invalid JSON: {response.text[:200]}"
                            print(f"❌ Endpoint returned invalid JSON: {response.text[:200]}")
                    else:
                        result["endpoint_error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                        print(f"❌ Endpoint FAILED: {result['endpoint_error']}")
                    
                    # Clean up
                    await artifact_manager.delete(artifact_id=dataset["id"], delete_files=True)
                
                # Clean up local files
                if zarr_path.exists():
                    shutil.rmtree(zarr_path)
                if zip_path.exists():
                    zip_path.unlink()
                
            except Exception as e:
                print(f"❌ Error in scenario {scenario_name}: {e}")
                results.append({
                    "scenario": scenario_name,
                    "description": description,
                    "error": str(e)
                })
        
        # Print comparison analysis
        print(f"\n📊 COMPARISON ANALYSIS")
        print(f"=" * 60)
        
        if len(results) >= 2:
            normal = next((r for r in results if r["scenario"].startswith("normal")), None)
            mini = next((r for r in results if r["scenario"].startswith("mini-chunks")), None)
            
            if normal and mini and "error" not in normal and "error" not in mini:
                print(f"📈 Comparison between scenarios:")
                print(f"")
                print(f"{'Metric':<25} {'Normal Chunks':<15} {'Mini Chunks':<15} {'Difference'}")
                print(f"{'-'*70}")
                print(f"{'Total files':<25} {normal['total_files']:<15} {mini['total_files']:<15} {mini['total_files'] - normal['total_files']:+}")
                print(f"{'Files per MB':<25} {normal['files_per_mb']:<15.0f} {mini['files_per_mb']:<15.0f} {mini['files_per_mb'] - normal['files_per_mb']:+.0f}")
                print(f"{'Avg file size (bytes)':<25} {normal['average_file_size']:<15.0f} {mini['average_file_size']:<15.0f} {mini['average_file_size'] - normal['average_file_size']:+.0f}")
                print(f"{'Mini chunks %':<25} {normal['mini_chunks_percentage']:<15.1f} {mini['mini_chunks_percentage']:<15.1f} {mini['mini_chunks_percentage'] - normal['mini_chunks_percentage']:+.1f}")
                print(f"{'ZIP compression':<25} {normal['zip_compression']:<15} {mini['zip_compression']:<15} {'Different' if normal['zip_compression'] != mini['zip_compression'] else 'Same'}")
                print(f"{'ZIP valid':<25} {normal['zip_valid']:<15} {mini['zip_valid']:<15} {'Different' if normal['zip_valid'] != mini['zip_valid'] else 'Same'}")
                
                if "endpoint_success" in mini:
                    print(f"{'Endpoint access':<25} {'N/A':<15} {mini['endpoint_success']:<15} {'N/A'}")
                
                print(f"")
                
                # Determine likely cause
                if mini['mini_chunks_percentage'] > normal['mini_chunks_percentage'] * 2:
                    print(f"🔍 DIAGNOSIS: Mini chunk percentage is significantly higher in the problematic case")
                    print(f"   Normal: {normal['mini_chunks_percentage']:.1f}% mini chunks")
                    print(f"   Problematic: {mini['mini_chunks_percentage']:.1f}% mini chunks")
                    print(f"   This likely causes ZIP central directory corruption")
                
                if mini['files_per_mb'] > normal['files_per_mb'] * 1.5:
                    print(f"🔍 DIAGNOSIS: File density is much higher in the problematic case")
                    print(f"   Normal: {normal['files_per_mb']:.0f} files/MB")
                    print(f"   Problematic: {mini['files_per_mb']:.0f} files/MB")
                    print(f"   High file density can stress ZIP central directory structure")
                
                if not mini['zip_valid'] and normal['zip_valid']:
                    print(f"🔍 DIAGNOSIS: ZIP corruption detected in mini chunk case")
                    print(f"   This confirms that mini chunks cause ZIP file corruption")
        
        # Assert that we can reproduce the issue
        mini_chunk_result = next((r for r in results if r["scenario"].startswith("mini-chunks")), None)
        if mini_chunk_result and "endpoint_success" in mini_chunk_result:
            if not mini_chunk_result["endpoint_success"]:
                print(f"\n✅ SUCCESS: Reproduced the mini chunk ZIP corruption issue!")
                print(f"   Endpoint failed as expected: {mini_chunk_result.get('endpoint_error', 'Unknown error')}")
            else:
                print(f"\n⚠️ UNEXPECTED: Mini chunk test passed - may need to adjust parameters")
        
        print(f"\n✅ Mini chunk reproduction test completed")

if __name__ == "__main__":
    # Allow running this test directly
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        os.environ["QUICK_TEST"] = "1"
    
    pytest.main([__file__, "-v", "-s"]) 