"""
Test script for ZIP64 upload and endpoint access functionality.

This test verifies that large ZIP files (including ZIP64 format) are correctly
handled by the upload and endpoint access systems. It tests various file sizes
from 100MB to 10GB to ensure proper ZIP64 support.

Test Cases:
- 100MB, 200MB, 400MB, 800MB, 1.6GB, 3.2GB, 10GB
- Creates OME-Zarr datasets of specified sizes
- Uploads via artifact manager
- Tests endpoint access via HTTP
- Ensures proper cleanup
"""

import pytest
import pytest_asyncio
import asyncio
import os
import time
import uuid
import numpy as np
import json
import tempfile
import shutil
import httpx
from pathlib import Path
from typing import List, Dict, Tuple
import zarr
from numcodecs import Blosc

from hypha_rpc import connect_to_server
from squid_control.hypha_tools.artifact_manager.artifact_manager import SquidArtifactManager
from squid_control.stitching.zarr_canvas import ZarrCanvas

# Mark all tests as integration tests requiring external services
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# Test configuration
TEST_SERVER_URL = "https://hypha.aicell.io"
TEST_WORKSPACE = "agent-lens"
TEST_TIMEOUT = 600  # 10 minutes for large uploads


class TestZip64UploadEndpoint:
    """Test class for ZIP64 upload and endpoint functionality."""
    
    @pytest_asyncio.fixture(autouse=True)
    async def setup_and_cleanup(self):
        """Setup and cleanup fixture that runs for each test method."""
        # Setup
        self.temp_dirs = []
        self.created_datasets = []
        self.artifact_manager = None
        self.gallery_id = None
        
        # Check for required token
        token = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
        if not token:
            pytest.skip("AGENT_LENS_WORKSPACE_TOKEN not set in environment")
        
        try:
            # Initialize artifact manager
            self.artifact_manager = SquidArtifactManager(
                server_url=TEST_SERVER_URL,
                workspace=TEST_WORKSPACE,
                token=token
            )
            await self.artifact_manager.setup()
            
            # Create test gallery
            gallery_name = f"test-zip64-gallery-{uuid.uuid4().hex[:8]}"
            print(f"Creating test gallery: {gallery_name}")
            self.gallery_id = await self.artifact_manager.create_or_get_microscope_gallery(
                microscope_service_id=gallery_name
            )
            print(f"Test gallery created: {self.gallery_id}")
            
            yield  # Run the test
            
        finally:
            # Cleanup
            await self._cleanup()
    
    async def _cleanup(self):
        """Comprehensive cleanup of all test resources."""
        print("\n🧹 Starting comprehensive cleanup...")
        
        # Remove all created datasets
        if self.artifact_manager and self.created_datasets:
            for dataset_info in self.created_datasets:
                try:
                    print(f"Removing dataset: {dataset_info['name']}")
                    await self.artifact_manager.remove_dataset(dataset_info['id'])
                    print(f"✅ Removed dataset: {dataset_info['name']}")
                except Exception as e:
                    print(f"⚠️ Failed to remove dataset {dataset_info['name']}: {e}")
        
        # Remove test gallery
        if self.artifact_manager and self.gallery_id:
            try:
                print(f"Removing test gallery: {self.gallery_id}")
                await self.artifact_manager.remove_gallery(self.gallery_id)
                print(f"✅ Removed test gallery")
            except Exception as e:
                print(f"⚠️ Failed to remove gallery: {e}")
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"✅ Cleaned up temp dir: {temp_dir}")
            except Exception as e:
                print(f"⚠️ Failed to clean up {temp_dir}: {e}")
        
        print("✅ Cleanup completed")
    
    def create_test_zarr_dataset(self, target_size_mb: float, temp_dir: str) -> str:
        """
        Create a test OME-Zarr dataset of approximately the target size.
        
        Args:
            target_size_mb: Target size in megabytes
            temp_dir: Temporary directory for the dataset
            
        Returns:
            str: Path to the created zarr directory
        """
        zarr_path = os.path.join(temp_dir, "data.zarr")
        
        # Calculate chunk and array dimensions to reach target size
        target_bytes = int(target_size_mb * 1024 * 1024)
        
        # Use smaller chunks for efficiency, more chunks for larger datasets
        if target_size_mb < 500:
            chunk_size = 256
            dtype = np.uint8  # 1 byte per pixel
        elif target_size_mb < 2000:
            chunk_size = 512
            dtype = np.uint8
        else:
            chunk_size = 1024
            dtype = np.uint8
        
        bytes_per_pixel = np.dtype(dtype).itemsize
        pixels_per_chunk = chunk_size * chunk_size * bytes_per_pixel
        
        # Calculate how many chunks we need
        total_chunks_needed = max(1, target_bytes // pixels_per_chunk)
        
        # Distribute chunks across timepoints, channels, and Z-slices
        if target_size_mb < 200:
            timepoints = 1
            channels = 2
            z_slices = 1
        elif target_size_mb < 1000:
            timepoints = 2
            channels = 3
            z_slices = 2
        else:
            timepoints = 3
            channels = 4
            z_slices = 3
        
        chunks_per_layer = total_chunks_needed // (timepoints * channels * z_slices)
        chunks_per_layer = max(1, chunks_per_layer)
        
        # Calculate grid dimensions
        grid_size = int(np.sqrt(chunks_per_layer)) + 1
        height = width = grid_size * chunk_size
        
        print(f"Creating {target_size_mb}MB zarr dataset:")
        print(f"  Dimensions: {timepoints}T x {channels}C x {z_slices}Z x {height}Y x {width}X")
        print(f"  Chunk size: {chunk_size}x{chunk_size}")
        print(f"  Data type: {dtype}")
        print(f"  Estimated chunks: {total_chunks_needed}")
        
        # Create zarr group with OME structure
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)
        
        # Create OME metadata
        ome_metadata = {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "multiscales": [
                {
                    "version": "0.4",
                    "name": "test_dataset",
                    "axes": [
                        {"name": "t", "type": "time", "unit": "second"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"}
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.5, 0.5]}
                            ]
                        },
                        {
                            "path": "1",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}
                            ]
                        }
                    ]
                }
            ]
        }
        
        root.attrs["omero"] = ome_metadata
        
        # Create scale 0 (full resolution)
        compressor = Blosc(cname='zstd', clevel=1)
        scale0 = root.create_dataset(
            "0",
            shape=(timepoints, channels, z_slices, height, width),
            chunks=(1, 1, 1, chunk_size, chunk_size),
            dtype=dtype,
            compressor=compressor,
            fill_value=0
        )
        
        # Create scale 1 (half resolution)
        scale1 = root.create_dataset(
            "1",
            shape=(timepoints, channels, z_slices, height//2, width//2),
            chunks=(1, 1, 1, chunk_size//2, chunk_size//2),
            dtype=dtype,
            compressor=compressor,
            fill_value=0
        )
        
        # Generate test data patterns
        print(f"Generating test data...")
        total_pixels_written = 0
        
        for t in range(timepoints):
            for c in range(channels):
                for z in range(z_slices):
                    # Create a test pattern
                    layer_data = np.zeros((height, width), dtype=dtype)
                    
                    # Add some pattern - gradients and shapes
                    y_indices, x_indices = np.meshgrid(
                        np.arange(height), np.arange(width), indexing='ij'
                    )
                    
                    # Different patterns for different channels
                    if c == 0:  # Gradient pattern
                        pattern = ((x_indices + y_indices) % 256).astype(dtype)
                    elif c == 1:  # Checkerboard
                        pattern = (((x_indices // 32) + (y_indices // 32)) % 2 * 255).astype(dtype)
                    elif c == 2:  # Radial pattern
                        center_y, center_x = height // 2, width // 2
                        distance = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                        pattern = (distance % 256).astype(dtype)
                    else:  # Random-like pattern
                        pattern = ((x_indices * y_indices + t * 100 + z * 50) % 256).astype(dtype)
                    
                    # Add timepoint and z variation
                    pattern = ((pattern + t * 20 + z * 10) % 256).astype(dtype)
                    
                    # Write to scale 0
                    scale0[t, c, z, :, :] = pattern
                    total_pixels_written += pattern.size
                    
                    # Write to scale 1 (downsampled)
                    downsampled = pattern[::2, ::2]
                    scale1[t, c, z, :, :] = downsampled
                    
                    if (t * channels * z_slices + c * z_slices + z + 1) % 5 == 0:
                        progress = (t * channels * z_slices + c * z_slices + z + 1) / (timepoints * channels * z_slices)
                        print(f"  Progress: {progress*100:.1f}%")
        
        # Verify the created dataset size
        dataset_size = self._get_directory_size(zarr_path)
        actual_size_mb = dataset_size / (1024 * 1024)
        
        print(f"✅ Created zarr dataset: {actual_size_mb:.1f}MB (target: {target_size_mb}MB)")
        print(f"  Total pixels written: {total_pixels_written:,}")
        print(f"  Path: {zarr_path}")
        
        return zarr_path
    
    def _get_directory_size(self, path: str) -> int:
        """Get the total size of a directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    async def create_and_upload_dataset(self, size_mb: float) -> Dict:
        """Create and upload a dataset of the specified size."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"zarr_test_{int(size_mb)}mb_")
        self.temp_dirs.append(temp_dir)
        
        try:
            # Create zarr dataset
            print(f"\n📦 Creating {size_mb}MB OME-Zarr dataset...")
            zarr_path = self.create_test_zarr_dataset(size_mb, temp_dir)
            
            # Create ZarrCanvas and export as ZIP
            print(f"🗜️ Creating ZIP archive...")
            canvas = ZarrCanvas(
                base_path=temp_dir,
                pixel_size_xy_um=0.33,
                stage_limits={"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
                channels=["BF", "DAPI", "FITC", "TRITC"],
                initialize_new=False
            )
            
            # Export as ZIP
            zip_start_time = time.time()
            zip_content = canvas.export_as_zip()
            zip_time = time.time() - zip_start_time
            zip_size_mb = len(zip_content) / (1024 * 1024)
            
            print(f"✅ ZIP created: {zip_size_mb:.1f}MB in {zip_time:.1f}s")
            
            # Upload to artifact manager
            dataset_name = f"test-zip64-{int(size_mb)}mb-{uuid.uuid4().hex[:8]}"
            print(f"⬆️ Uploading dataset: {dataset_name}")
            
            upload_start_time = time.time()
            upload_result = await self.artifact_manager.upload_zarr_dataset(
                microscope_service_id=self.gallery_id["microscope_service_id"],
                dataset_name=dataset_name,
                zarr_zip_content=zip_content,
                description=f"Test dataset for ZIP64 functionality - {size_mb}MB"
            )
            upload_time = time.time() - upload_start_time
            
            print(f"✅ Upload completed in {upload_time:.1f}s")
            
            # Track for cleanup
            dataset_info = {
                "id": upload_result["dataset"]["id"],
                "name": dataset_name,
                "size_mb": zip_size_mb,
                "artifact_id": upload_result["dataset"]["artifact_id"]
            }
            self.created_datasets.append(dataset_info)
            
            return {
                "dataset_info": dataset_info,
                "upload_result": upload_result,
                "zip_size_mb": zip_size_mb,
                "upload_time": upload_time,
                "zip_creation_time": zip_time
            }
            
        except Exception as e:
            print(f"❌ Failed to create/upload {size_mb}MB dataset: {e}")
            raise
    
    async def test_endpoint_access(self, dataset_info: Dict) -> Dict:
        """Test accessing the dataset via the HTTP endpoint."""
        artifact_id = dataset_info["artifact_id"]
        endpoint_url = f"https://hypha.aicell.io/agent-lens/artifacts/{artifact_id}/zip-files/zarr_dataset.zip/?path=data.zarr/"
        
        print(f"🌐 Testing endpoint access: {endpoint_url}")
        
        # Test with timeout appropriate for large files
        timeout_seconds = max(60, int(dataset_info["size_mb"] / 10))
        
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            try:
                start_time = time.time()
                response = await client.get(endpoint_url)
                access_time = time.time() - start_time
                
                print(f"📊 Response: {response.status_code} in {access_time:.1f}s")
                
                if response.status_code == 200:
                    try:
                        content = response.json()
                        if isinstance(content, list):
                            print(f"✅ Directory listing received: {len(content)} items")
                            return {
                                "success": True,
                                "status_code": response.status_code,
                                "access_time": access_time,
                                "content_type": "directory_listing",
                                "item_count": len(content)
                            }
                        else:
                            print(f"✅ JSON response received: {type(content)}")
                            return {
                                "success": True,
                                "status_code": response.status_code,
                                "access_time": access_time,
                                "content_type": "json",
                                "content": content
                            }
                    except json.JSONDecodeError:
                        # Not JSON, might be binary data
                        content_length = len(response.content)
                        print(f"✅ Binary response received: {content_length} bytes")
                        return {
                            "success": True,
                            "status_code": response.status_code,
                            "access_time": access_time,
                            "content_type": "binary",
                            "content_length": content_length
                        }
                else:
                    error_detail = "Unknown error"
                    try:
                        error_content = response.json()
                        error_detail = error_content.get("detail", str(error_content))
                    except:
                        error_detail = response.text[:200]
                    
                    print(f"❌ Endpoint access failed: {response.status_code} - {error_detail}")
                    return {
                        "success": False,
                        "status_code": response.status_code,
                        "access_time": access_time,
                        "error": error_detail
                    }
                    
            except httpx.TimeoutException:
                print(f"⏰ Endpoint access timed out after {timeout_seconds}s")
                return {
                    "success": False,
                    "status_code": None,
                    "access_time": timeout_seconds,
                    "error": "Timeout"
                }
            except Exception as e:
                print(f"❌ Endpoint access error: {e}")
                return {
                    "success": False,
                    "status_code": None,
                    "access_time": 0,
                    "error": str(e)
                }
    
    @pytest.mark.parametrize("size_mb", [100, 200, 400, 800, 1600, 3200])
    async def test_zip64_upload_and_access(self, size_mb: float):
        """Test uploading and accessing datasets of various sizes."""
        print(f"\n{'='*60}")
        print(f"🧪 Testing {size_mb}MB dataset upload and access")
        print(f"{'='*60}")
        
        # Create and upload dataset
        upload_result = await self.create_and_upload_dataset(size_mb)
        
        # Verify upload was successful
        assert upload_result["upload_result"]["success"] == True
        assert upload_result["zip_size_mb"] > 0
        
        # Test endpoint access
        access_result = await self.test_endpoint_access(upload_result["dataset_info"])
        
        # Verify endpoint access
        if not access_result["success"]:
            # For large files, we might need to investigate the specific error
            if size_mb >= 1600 and "Bad offset for central directory" in str(access_result.get("error", "")):
                pytest.xfail(f"Known ZIP64 central directory issue with {size_mb}MB files")
            else:
                pytest.fail(f"Endpoint access failed: {access_result}")
        
        assert access_result["success"] == True
        assert access_result["status_code"] == 200
        
        # Print summary
        print(f"\n📋 Test Summary for {size_mb}MB:")
        print(f"  ZIP Size: {upload_result['zip_size_mb']:.1f}MB")
        print(f"  ZIP Creation Time: {upload_result['zip_creation_time']:.1f}s")
        print(f"  Upload Time: {upload_result['upload_time']:.1f}s")
        print(f"  Endpoint Access Time: {access_result['access_time']:.1f}s")
        print(f"  Content Type: {access_result.get('content_type', 'unknown')}")
        
        if access_result.get("item_count"):
            print(f"  Directory Items: {access_result['item_count']}")
        
        print(f"✅ {size_mb}MB test completed successfully")
    
    @pytest.mark.slow
    async def test_very_large_zip64_upload_and_access(self):
        """Test very large datasets (10GB) - marked as slow test."""
        size_mb = 10 * 1024  # 10GB
        
        print(f"\n{'='*60}")
        print(f"🧪 Testing VERY LARGE {size_mb}MB (10GB) dataset upload and access")
        print(f"⚠️ This test will take significant time and resources")
        print(f"{'='*60}")
        
        # Create and upload dataset
        upload_result = await self.create_and_upload_dataset(size_mb)
        
        # Verify upload was successful
        assert upload_result["upload_result"]["success"] == True
        assert upload_result["zip_size_mb"] > 0
        
        # Test endpoint access
        access_result = await self.test_endpoint_access(upload_result["dataset_info"])
        
        # For very large files, we expect this might fail with current ZIP64 implementation
        if not access_result["success"]:
            if "Bad offset for central directory" in str(access_result.get("error", "")):
                pytest.xfail("Expected ZIP64 central directory issue with 10GB files - needs server-side fix")
            else:
                pytest.fail(f"Unexpected endpoint access failure: {access_result}")
        
        # If it succeeds, that's great!
        assert access_result["success"] == True
        assert access_result["status_code"] == 200
        
        print(f"\n📋 Test Summary for 10GB:")
        print(f"  ZIP Size: {upload_result['zip_size_mb']:.1f}MB")
        print(f"  ZIP Creation Time: {upload_result['zip_creation_time']:.1f}s")
        print(f"  Upload Time: {upload_result['upload_time']:.1f}s")
        print(f"  Endpoint Access Time: {access_result['access_time']:.1f}s")
        print(f"✅ 10GB test completed successfully - ZIP64 implementation is working!")


if __name__ == "__main__":
    """Run tests directly if executed as a script."""
    pytest.main([__file__, "-v", "-s"]) 