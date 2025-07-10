"""
This module provides the ArtifactManager class, which manages artifacts for the application.
It includes methods for creating vector collections, adding vectors, searching vectors,
and handling file uploads and downloads.
"""

import httpx
from hypha_rpc.rpc import RemoteException
import asyncio
import os
import io
import dotenv
from hypha_rpc import connect_to_server
from PIL import Image
import numpy as np
import base64
import numcodecs
import blosc
import aiohttp
from collections import deque
import zarr
import time
import json
import uuid

dotenv.load_dotenv()  
ENV_FILE = dotenv.find_dotenv()  
if ENV_FILE:  
    dotenv.load_dotenv(ENV_FILE)  

class SquidArtifactManager:
    """
    Manages artifacts for the application.
    """

    def __init__(self):
        self._svc = None
        self.server = None

    async def connect_server(self, server):
        """
        Connect to the server.

        Args:
            server (Server): The server instance.
        """
        self.server = server
        self._svc = await server.get_service("public/artifact-manager")


    def _artifact_id(self, workspace, name):
        """
        Generate the artifact ID.

        Args:
            workspace (str): The workspace.
            name (str): The artifact name.

        Returns:
            str: The artifact ID.
        """
        return f"{workspace}/{name}"

    async def create_vector_collection(
        self, workspace, name, manifest, config, overwrite=False, exists_ok=False
    ):
        """
        Create a vector collection.

        Args:
            workspace (str): The workspace.
            name (str): The collection name.
            manifest (dict): The collection manifest.
            config (dict): The collection configuration.
            overwrite (bool, optional): Whether to overwrite the existing collection.
        """
        art_id = self._artifact_id(workspace, name)
        try:
            await self._svc.create(
                alias=art_id,
                type="vector-collection",
                manifest=manifest,
                config=config,
                overwrite=overwrite,
            )
        except RemoteException as e:
            if not exists_ok:
                raise e

    async def add_vectors(self, workspace, coll_name, vectors):
        """
        Add vectors to the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            vectors (list): The vectors to add.
        """
        art_id = self._artifact_id(workspace, coll_name)
        await self._svc.add_vectors(artifact_id=art_id, vectors=vectors)

    async def search_vectors(self, workspace, coll_name, vector, top_k=None):
        """
        Search for vectors in the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            vector (ndarray): The query vector.
            top_k (int, optional): The number of top results to return.

        Returns:
            list: The search results.
        """
        art_id = self._artifact_id(workspace, coll_name)
        return await self._svc.search_vectors(
            artifact_id=art_id, query={"cell_image_vector": vector}, limit=top_k
        )

    async def add_file(self, workspace, coll_name, file_content, file_path):
        """
        Add a file to the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            file_content (bytes): The file content.
            file_path (str): The file path.
        """
        art_id = self._artifact_id(workspace, coll_name)
        await self._svc.edit(artifact_id=art_id, version="stage")
        put_url = await self._svc.put_file(art_id, file_path, download_weight=1.0)
        async with httpx.AsyncClient() as client:
            response = await client.put(put_url, data=file_content, timeout=500)
        response.raise_for_status()
        await self._svc.commit(art_id)

    async def get_file(self, workspace, coll_name, file_path):
        """
        Retrieve a file from the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
            file_path (str): The file path.

        Returns:
            bytes: The file content.
        """
        art_id = self._artifact_id(workspace, coll_name)
        get_url = await self._svc.get_file(art_id, file_path)

        async with httpx.AsyncClient() as client:
            response = await client.get(get_url, timeout=500)
        response.raise_for_status()

        return response.content

    async def remove_vectors(self, workspace, coll_name, vector_ids=None):
        """
        Clear the vectors in the collection.

        Args:
            workspace (str): The workspace.
            coll_name (str): The collection name.
        """
        art_id = self._artifact_id(workspace, coll_name)
        if vector_ids is None:
            all_vectors = await self._svc.list_vectors(art_id)
            while len(all_vectors) > 0:
                vector_ids = [vector["id"] for vector in all_vectors]
                await self._svc.remove_vectors(art_id, vector_ids)
                all_vectors = await self._svc.list_vectors(art_id)
        else:
            await self._svc.remove_vectors(art_id, vector_ids)

    async def list_files_in_dataset(self, dataset_id):
        """
        List all files in a dataset.

        Args:
            dataset_id (str): The ID of the dataset.

        Returns:
            list: A list of files in the dataset.
        """
        files = await self._svc.list_files(dataset_id)
        return files

    async def navigate_collections(self, parent_id=None):
        """
        Navigate through collections and datasets.

        Args:
            parent_id (str, optional): The ID of the parent collection. Defaults to None for top-level collections.

        Returns:
            list: A list of collections and datasets under the specified parent.
        """
        collections = await self._svc.list(artifact_id=parent_id)
        return collections

    async def get_file_details(self, dataset_id, file_path):
        """
        Get details of a specific file in a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            file_path (str): The path to the file in the dataset.

        Returns:
            dict: Details of the file, including size, type, and last modified date.
        """
        files = await self._svc.list_files(dataset_id)
        for file in files:
            if file['name'] == file_path:
                return file
        return None

    async def download_file(self, dataset_id, file_path, local_path):
        """
        Download a file from a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            file_path (str): The path to the file in the dataset.
            local_path (str): The local path to save the downloaded file.
        """
        get_url = await self._svc.get_file(dataset_id, file_path)
        async with httpx.AsyncClient() as client:
            response = await client.get(get_url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)

    async def search_datasets(self, keywords=None, filters=None):
        """
        Search and filter datasets based on keywords and filters.

        Args:
            keywords (list, optional): A list of keywords for searching datasets.
            filters (dict, optional): A dictionary of filters to apply.

        Returns:
            list: A list of datasets matching the search criteria.
        """
        datasets = await self._svc.list(keywords=keywords, filters=filters)
        return datasets

    async def list_subfolders(self, dataset_id, dir_path=None):
        """
        List all subfolders in a specified directory within a dataset.

        Args:
            dataset_id (str): The ID of the dataset.
            dir_path (str, optional): The directory path within the dataset to list subfolders. Defaults to None for the root directory.

        Returns:
            list: A list of subfolders in the specified directory.
        """
        try:
            print(f"Listing files for dataset_id={dataset_id}, dir_path={dir_path}")
            files = await self._svc.list_files(dataset_id, dir_path=dir_path)
            print(f"Files received, length: {len(files)}")
            subfolders = [file for file in files if file.get('type') == 'directory']
            print(f"Subfolders filtered, length: {len(subfolders)}")
            return subfolders
        except Exception as e:
            print(f"Error listing subfolders for {dataset_id}: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    async def create_or_get_microscope_gallery(self, microscope_service_id):
        """
        Create or get a gallery for a specific microscope in the agent-lens workspace.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            
        Returns:
            dict: The gallery artifact information
        """
        workspace = "agent-lens"
        gallery_alias = f"microscope-gallery-{microscope_service_id}"
        
        try:
            # Try to get existing gallery
            gallery = await self._svc.read(artifact_id=f"{workspace}/{gallery_alias}")
            print(f"Found existing gallery: {gallery_alias}")
            return gallery
        except Exception as e:
            # Handle both RemoteException and RemoteError, and check for various error patterns
            error_str = str(e).lower()
            if ("not found" in error_str or 
                "does not exist" in error_str or 
                "keyerror" in error_str or
                "artifact with id" in error_str):
                # Gallery doesn't exist, create it
                print(f"Creating new gallery: {gallery_alias}")
                gallery_manifest = {
                    "name": f"Microscope Gallery - {microscope_service_id}",
                    "description": f"Dataset collection for microscope service {microscope_service_id}",
                    "microscope_service_id": microscope_service_id,
                    "created_by": "squid-control-system",
                    "type": "microscope-gallery"
                }
                
                gallery_config = {
                    "permissions": {"*": "r", "@": "r+"},
                    "collection_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "record_type": {"type": "string", "enum": ["zarr-dataset"]},
                            "microscope_service_id": {"type": "string"},
                            "acquisition_settings": {"type": "object"},
                            "timestamp": {"type": "string"}
                        },
                        "required": ["name", "description", "record_type"]
                    }
                }
                
                gallery = await self._svc.create(
                    alias=f"{workspace}/{gallery_alias}",
                    type="collection",
                    manifest=gallery_manifest,
                    config=gallery_config
                )
                print(f"Created gallery: {gallery_alias}")
                return gallery
            else:
                raise e
    
    async def check_dataset_name_availability(self, microscope_service_id, dataset_name):
        """
        Check if a dataset name is available in the microscope's gallery.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            dataset_name (str): The proposed dataset name
            
        Returns:
            dict: Information about name availability and suggestions
        """
        workspace = "agent-lens"
        gallery_alias = f"microscope-gallery-{microscope_service_id}"
        dataset_alias = f"{workspace}/{dataset_name}"
        
        # Validate dataset name format
        import re
        if not re.match(r'^[a-z0-9][a-z0-9\-:]*[a-z0-9]$', dataset_name):
            return {
                "available": False,
                "reason": "Invalid name format. Use lowercase letters, numbers, hyphens, and colons only. Must start and end with alphanumeric character.",
                "suggestions": []
            }
        
        try:
            # Check if dataset already exists
            existing_dataset = await self._svc.read(artifact_id=dataset_alias)
            
            # Generate alternative suggestions
            suggestions = []
            import time
            timestamp = int(time.time())
            base_suggestions = [
                f"{dataset_name}-v2",
                f"{dataset_name}-{timestamp}",
                f"{dataset_name}-copy",
                f"{dataset_name}-new"
            ]
            
            for suggestion in base_suggestions:
                try:
                    await self._svc.read(artifact_id=f"{workspace}/{suggestion}")
                except Exception:
                    suggestions.append(suggestion)
                    if len(suggestions) >= 3:
                        break
            
            return {
                "available": False,
                "reason": "Dataset name already exists",
                "existing_dataset": existing_dataset,
                "suggestions": suggestions
            }
            
        except Exception as e:
            # Handle both RemoteException and RemoteError, and check for various error patterns
            error_str = str(e).lower()
            if ("not found" in error_str or 
                "does not exist" in error_str or 
                "keyerror" in error_str or
                "artifact with id" in error_str):
                return {
                    "available": True,
                    "reason": "Name is available",
                    "suggestions": []
                }
            else:
                raise e
    
    async def upload_zarr_dataset(self, microscope_service_id, dataset_name, zarr_zip_content, 
                                 acquisition_settings=None, description=None):
        """
        Upload a zarr fileset as a zip file to the microscope's gallery.
        
        Args:
            microscope_service_id (str): The hypha service ID of the microscope
            dataset_name (str): The name for the dataset
            zarr_zip_content (bytes): The zip file content containing the zarr fileset
            acquisition_settings (dict, optional): Acquisition settings metadata
            description (str, optional): Description of the dataset
            
        Returns:
            dict: Information about the uploaded dataset
        """
        workspace = "agent-lens"
        
        # Validate ZIP file before upload
        await self._validate_zarr_zip_content(zarr_zip_content)
        
        # Run detailed ZIP integrity test
        zip_test_results = await self.test_zip_file_integrity(zarr_zip_content, f"Upload: {dataset_name}")
        if not zip_test_results["valid"]:
            raise ValueError(f"ZIP file integrity test failed: {', '.join(zip_test_results['issues'])}")
        
        # Ensure gallery exists
        gallery = await self.create_or_get_microscope_gallery(microscope_service_id)
        
        # Check name availability
        name_check = await self.check_dataset_name_availability(microscope_service_id, dataset_name)
        if not name_check["available"]:
            raise ValueError(f"Dataset name '{dataset_name}' is not available: {name_check['reason']}")
        
        # Create dataset manifest
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        zip_size_mb = len(zarr_zip_content) / (1024 * 1024)
        
        dataset_manifest = {
            "name": dataset_name,
            "description": description or f"Zarr dataset from microscope {microscope_service_id}",
            "record_type": "zarr-dataset",
            "microscope_service_id": microscope_service_id,
            "timestamp": timestamp,
            "acquisition_settings": acquisition_settings or {},
            "file_format": "ome-zarr",
            "upload_method": "squid-control-api",
            "zip_size_mb": zip_size_mb,
            "zip_format": "ZIP64-compatible"
        }
        
        # Create dataset in staging mode
        dataset_alias = f"{workspace}/{dataset_name}"
        dataset = await self._svc.create(
            parent_id=gallery["id"],
            alias=dataset_alias,
            manifest=dataset_manifest,
            stage=True
        )
        
        try:
            # Upload zarr zip file with retry logic
            await self._upload_large_zip_with_retry(dataset["id"], zarr_zip_content, max_retries=3)
            
            # Commit the dataset
            await self._svc.commit(dataset["id"])
            
            # Test the uploaded ZIP file to ensure it's accessible
            print("Testing uploaded ZIP file accessibility...")
            upload_test_results = await self.test_uploaded_zip_file(dataset["id"])
            
            if not upload_test_results["download_success"]:
                raise Exception(f"Uploaded ZIP file is not accessible: {upload_test_results.get('error', 'Unknown error')}")
            
            if not upload_test_results["zip_integrity"]["valid"]:
                issues = upload_test_results["zip_integrity"]["issues"]
                print(f"WARNING: Uploaded ZIP file has integrity issues: {', '.join(issues)}")
            else:
                print("Uploaded ZIP file verified successfully")
            
            print(f"Successfully uploaded zarr dataset: {dataset_name} ({zip_size_mb:.2f} MB)")
            return {
                "success": True,
                "dataset_id": dataset["id"],
                "dataset_name": dataset_name,
                "gallery_id": gallery["id"],
                "upload_timestamp": timestamp,
                "zip_size_mb": zip_size_mb,
                "zip_test_results": upload_test_results
            }
            
        except Exception as e:
            # If upload fails, try to clean up the staged dataset
            try:
                await self._svc.discard(dataset["id"])
            except:
                pass
            raise e

    async def _validate_zarr_zip_content(self, zarr_zip_content: bytes) -> None:
        """
        Validate that the ZIP content is properly formatted and not corrupted.
        
        Args:
            zarr_zip_content (bytes): The ZIP file content to validate
            
        Raises:
            ValueError: If ZIP file is invalid or corrupted
        """
        try:
            import zipfile
            import io
            
            zip_buffer = io.BytesIO(zarr_zip_content)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                # Test the ZIP file structure
                file_list = zip_file.namelist()
                if not file_list:
                    raise ValueError("ZIP file is empty")
                
                # Check for expected zarr structure
                zarr_files = [f for f in file_list if f.startswith('data.zarr/')]
                if not zarr_files:
                    raise ValueError("ZIP file does not contain expected 'data.zarr/' structure")
                
                # Test that we can read the ZIP file entries
                test_count = min(5, len(file_list))
                for i in range(test_count):
                    try:
                        with zip_file.open(file_list[i]) as f:
                            f.read(1024)  # Read a small chunk
                    except Exception as e:
                        raise ValueError(f"Cannot read file {file_list[i]} from ZIP: {e}")
                
                print(f"ZIP validation passed: {len(file_list)} files, {len(zarr_zip_content) / (1024*1024):.2f} MB")
                
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid ZIP file format: {e}")
        except Exception as e:
            raise ValueError(f"ZIP file validation failed: {e}")

    async def _upload_large_zip_with_retry(self, dataset_id: str, zarr_zip_content: bytes, max_retries: int = 3) -> None:
        """
        Upload large ZIP file with retry logic and appropriate timeouts.
        
        Args:
            dataset_id (str): The dataset ID
            zarr_zip_content (bytes): The ZIP file content
            max_retries (int): Maximum number of retry attempts
            
        Raises:
            Exception: If upload fails after all retries
        """
        zip_size_mb = len(zarr_zip_content) / (1024 * 1024)
        
        # Calculate timeout based on file size (minimum 5 minutes, add 1 minute per 50MB)
        timeout_seconds = max(300, int(zip_size_mb / 50) * 60 + 300)
        
        for attempt in range(max_retries):
            try:
                print(f"Upload attempt {attempt + 1}/{max_retries} for {zip_size_mb:.2f} MB ZIP file (timeout: {timeout_seconds}s)")
                
                # Get upload URL
                put_url = await self._svc.put_file(
                    dataset_id, 
                    file_path="zarr_dataset.zip", 
                    download_weight=1.0
                )
                
                # Upload with appropriate timeout and chunked transfer
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
                    response = await client.put(
                        put_url, 
                        content=zarr_zip_content,
                        headers={
                            'Content-Type': 'application/zip',
                            'Content-Length': str(len(zarr_zip_content))
                        }
                    )
                    response.raise_for_status()
                
                print(f"Upload successful on attempt {attempt + 1}")
                return
                
            except httpx.TimeoutException as e:
                print(f"Upload timeout on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Upload failed after {max_retries} attempts due to timeout")
                
            except httpx.HTTPStatusError as e:
                print(f"Upload HTTP error on attempt {attempt + 1}: {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 413:  # Payload too large
                    raise Exception(f"ZIP file is too large ({zip_size_mb:.2f} MB) for upload")
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

    async def test_zip_file_integrity(self, zip_content: bytes, description: str = "ZIP file") -> dict:
        """
        Test ZIP file integrity and provide detailed diagnostics.
        
        Args:
            zip_content (bytes): The ZIP file content to test
            description (str): Description of the ZIP file for logging
            
        Returns:
            dict: Detailed test results including file count, size, and any issues
        """
        import zipfile
        import io
        
        results = {
            "valid": False,
            "size_mb": len(zip_content) / (1024 * 1024),
            "file_count": 0,
            "zip64_required": False,
            "zip64_enabled": False,
            "compression_method": None,
            "issues": [],
            "sample_files": []
        }
        
        try:
            zip_buffer = io.BytesIO(zip_content)
            
            # Test basic ZIP file opening
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                file_list = zip_file.namelist()
                results["file_count"] = len(file_list)
                
                if not file_list:
                    results["issues"].append("ZIP file is empty")
                    return results
                
                # Check if ZIP64 is required/enabled
                results["zip64_required"] = results["size_mb"] > 200 or results["file_count"] > 65535
                
                # Test central directory access
                try:
                    info_list = zip_file.infolist()
                    
                    # More comprehensive ZIP64 detection
                    # Check for ZIP64 indicators: large file sizes, large archive, or ZIP64 extra fields
                    zip64_indicators = []
                    
                    # Check for large individual files
                    large_files = any(info.file_size >= 0xFFFFFFFF or info.compress_size >= 0xFFFFFFFF for info in info_list)
                    if large_files:
                        zip64_indicators.append("large_files")
                    
                    # Check for ZIP64 extra fields (more reliable indicator)
                    zip64_extra_fields = any(
                        any(extra[0] == 0x0001 for extra in getattr(info, 'extra', []) if len(extra) >= 2) 
                        for info in info_list[:10]  # Check first 10 files
                    )
                    if zip64_extra_fields:
                        zip64_indicators.append("extra_fields")
                    
                    # Check total archive size vs ZIP32 limits
                    if results["size_mb"] * 1024 * 1024 >= 0xFFFFFFFF:  # 4GB limit
                        zip64_indicators.append("archive_size")
                    
                    # Check file count vs ZIP32 limits
                    if results["file_count"] >= 0xFFFF:  # 65535 file limit
                        zip64_indicators.append("file_count")
                    
                    results["zip64_enabled"] = len(zip64_indicators) > 0
                    results["zip64_indicators"] = zip64_indicators
                    
                    # Get compression method from first file
                    if info_list:
                        results["compression_method"] = info_list[0].compress_type
                        
                except Exception as e:
                    results["issues"].append(f"Cannot access central directory: {e}")
                    return results
                
                # Test reading sample files
                sample_count = min(5, len(file_list))
                for i in range(sample_count):
                    try:
                        file_info = zip_file.getinfo(file_list[i])
                        with zip_file.open(file_list[i]) as f:
                            data = f.read(1024)  # Read first 1KB
                            results["sample_files"].append({
                                "name": file_list[i],
                                "size": file_info.file_size,
                                "compressed_size": file_info.compress_size,
                                "readable": True
                            })
                    except Exception as e:
                        results["issues"].append(f"Cannot read file {file_list[i]}: {e}")
                        results["sample_files"].append({
                            "name": file_list[i],
                            "readable": False,
                            "error": str(e)
                        })
                
                # Additional ZIP64 validation for large files
                if results["zip64_required"] and not results["zip64_enabled"]:
                    results["issues"].append("Large file requires ZIP64 format but ZIP64 is not enabled")
                elif results["zip64_required"] and results["zip64_enabled"]:
                    # Test ZIP64 central directory
                    try:
                        for info in info_list[:3]:  # Test first 3 files
                            if info.file_size > 0:
                                with zip_file.open(info) as f:
                                    f.read(1)
                    except Exception as e:
                        results["issues"].append(f"ZIP64 central directory test failed: {e}")
                
                # Mark as valid if no issues found
                results["valid"] = len(results["issues"]) == 0
                
        except zipfile.BadZipFile as e:
            results["issues"].append(f"Invalid ZIP file format: {e}")
        except Exception as e:
            results["issues"].append(f"ZIP file test failed: {e}")
        
        # Log results
        status = "VALID" if results["valid"] else "INVALID"
        print(f"ZIP Test [{description}]: {status}")
        print(f"  Size: {results['size_mb']:.2f} MB, Files: {results['file_count']}")
        print(f"  ZIP64 Required: {results['zip64_required']}, Enabled: {results['zip64_enabled']}")
        if "zip64_indicators" in results and results["zip64_indicators"]:
            print(f"  ZIP64 Indicators: {', '.join(results['zip64_indicators'])}")
        print(f"  Compression: {results['compression_method']}")
        
        if results["issues"]:
            print(f"  Issues: {', '.join(results['issues'])}")
        
        return results

    async def test_uploaded_zip_file(self, dataset_id: str, file_path: str = "zarr_dataset.zip") -> dict:
        """
        Test an uploaded ZIP file by downloading and verifying its integrity.
        
        Args:
            dataset_id (str): The dataset ID containing the ZIP file
            file_path (str): Path to the ZIP file within the dataset
            
        Returns:
            dict: Test results including download success and ZIP integrity
        """
        try:
            # Download the ZIP file
            get_url = await self._svc.get_file(dataset_id, file_path, silent=True)
            
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.get(get_url)
                response.raise_for_status()
                
                zip_content = response.content
                
                # Test the downloaded ZIP file
                zip_test_results = await self.test_zip_file_integrity(
                    zip_content, 
                    f"Downloaded: {dataset_id}/{file_path}"
                )
                
                return {
                    "download_success": True,
                    "download_size_mb": len(zip_content) / (1024 * 1024),
                    "zip_integrity": zip_test_results
                }
                
        except Exception as e:
            return {
                "download_success": False,
                "error": str(e),
                "zip_integrity": {"valid": False, "issues": [f"Download failed: {e}"]}
            }

# Constants
SERVER_URL = "https://hypha.aicell.io"
ARTIFACT_ALIAS = "20250506-scan-time-lapse-2025-05-06_17-56-38"
DEFAULT_CHANNEL = "BF_LED_matrix_full"

# New class to replace TileManager using Zarr for efficient access
class ZarrImageManager:
    def __init__(self):
        self.artifact_manager = None
        self.artifact_manager_server = None
        self.workspace = "agent-lens"  # Default workspace
        self.chunk_size = 256  # Default chunk size for Zarr
        self.channels = [
            "BF_LED_matrix_full",
            "Fluorescence_405_nm_Ex",
            "Fluorescence_488_nm_Ex",
            "Fluorescence_561_nm_Ex",
            "Fluorescence_638_nm_Ex"
        ]
        self.is_running = True
        self.session = None
        self.default_timestamp = "20250506-scan-time-lapse-2025-05-06_17-56-38"  # Set a default timestamp
        self.scale_key = 'scale0'
        
        # New attributes for HTTP-based access
        self.metadata_cache = {}  # Cache for .zarray and .zgroup metadata
        self.metadata_cache_lock = asyncio.Lock()
        self.processed_tile_cache = {}  # Cache for processed tiles
        self.processed_tile_ttl = 40 * 60  # 40 minutes in seconds
        self.processed_tile_cache_size = 1000  # Max number of tiles to cache
        self.empty_regions_cache = {}  # Cache for known empty regions
        self.empty_regions_cache_size = 500  # Max number of empty regions to cache
        self.http_session_lock = asyncio.Lock()
        self.server_url = "https://hypha.aicell.io"

    async def _get_http_session(self):
        """Get or create an aiohttp.ClientSession with increased connection pool."""
        async with self.http_session_lock:
            if self.session is None or self.session.closed:
                connector = aiohttp.TCPConnector(
                    limit_per_host=50,  # Max connections per host
                    limit=100,          # Total max connections
                    ssl=False if "localhost" in self.server_url else True
                )
                self.session = aiohttp.ClientSession(connector=connector)
            return self.session

    async def _fetch_zarr_metadata(self, dataset_alias, metadata_path_in_dataset, use_cache=True):
        """
        Fetch and cache Zarr metadata (.zgroup or .zarray) for a given dataset alias.
        Args:
            dataset_alias (str): The alias of the dataset (e.g., "agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38")
            metadata_path_in_dataset (str): Path within the dataset (e.g., "Channel/scaleN/.zarray")
            use_cache (bool): Whether to use cached metadata. Defaults to True.
        """
        cache_key = (dataset_alias, metadata_path_in_dataset)
        if use_cache:
            async with self.metadata_cache_lock:
                if cache_key in self.metadata_cache:
                    print(f"Using cached metadata for {cache_key}")
                    return self.metadata_cache[cache_key]

        if not self.artifact_manager:
            print("Artifact manager not available in ZarrImageManager for metadata fetch.")
            # Attempt to connect if not already
            await self.connect()
            if not self.artifact_manager:
                raise ConnectionError("Artifact manager connection failed.")

        try:
            print(f"Fetching metadata: dataset_alias='{dataset_alias}', path='{metadata_path_in_dataset}'")
            
            metadata_content_bytes = await self.artifact_manager.get_file(
                self.workspace,
                dataset_alias.split('/')[-1],  # Extract artifact name from full path
                metadata_path_in_dataset
            )
            metadata_str = metadata_content_bytes.decode('utf-8')
            import json
            metadata = json.loads(metadata_str)
            
            async with self.metadata_cache_lock:
                self.metadata_cache[cache_key] = metadata
            print(f"Fetched and cached metadata for {cache_key}")
            return metadata
        except Exception as e:
            print(f"Error fetching metadata for {dataset_alias} / {metadata_path_in_dataset}: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    async def connect(self,server_url="https://hypha.aicell.io"):
        """Connect to the Artifact Manager service and initialize http session."""
        try:
            self.server_url = server_url.rstrip('/') # Ensure no trailing slash

            self.artifact_manager_server = await connect_to_server({
                "client_id": f"zarr-image-client-for-squid-{uuid.uuid4()}",
                "server_url": server_url,
            })
            
            self.artifact_manager = SquidArtifactManager()
            await self.artifact_manager.connect_server(self.artifact_manager_server)
            
            # Initialize aiohttp session
            await self._get_http_session()  # Ensures session is created
            
            # Prime metadata for a default dataset if needed, or remove if priming is dynamic
            # Example: await self.prime_metadata("agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38", self.channels[0], scale=0)
            
            print("ZarrImageManager connected successfully")
            return True
        except Exception as e:
            print(f"Error connecting to artifact manager: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    async def close(self):
        """Close the image manager and cleanup resources"""
        self.is_running = False
        
        # Clear all caches
        self.processed_tile_cache.clear()
        async with self.metadata_cache_lock:
            self.metadata_cache.clear()
        self.empty_regions_cache.clear()
        
        # Close the aiohttp session
        async with self.http_session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
        
        # Disconnect from the server
        if self.artifact_manager_server:
            await self.artifact_manager_server.disconnect()
            self.artifact_manager_server = None
            self.artifact_manager = None

    def _add_to_empty_regions_cache(self, key):
        """Add a region key to the empty regions cache."""
        # Add to cache
        self.empty_regions_cache[key] = True # Store True instead of expiry_time
        
        # Basic FIFO size control if cache exceeds max size
        if len(self.empty_regions_cache) > self.empty_regions_cache_size:
            try:
                # Get the first key inserted (FIFO)
                first_key = next(iter(self.empty_regions_cache))
                del self.empty_regions_cache[first_key]
                print(f"Cleaned up oldest entry {first_key} from empty regions cache due to size limit.")
            except StopIteration: # pragma: no cover
                pass # Cache might have been cleared by another operation concurrently

    async def get_chunk_np_data(self, dataset_id, channel, scale, x, y):
        """
        Get a chunk as numpy array using new HTTP chunk access.
        Args:
            dataset_id (str): The alias of the dataset.
            channel (str): Channel name
            scale (int): Scale level
            x (int): X coordinate of the chunk for this scale.
            y (int): Y coordinate of the chunk for this scale.
        Returns:
            np.ndarray or None: Chunk data as numpy array, or None if not found/empty/error.
        """
        start_time = time.time()
        # Key for processed_tile_cache and empty_regions_cache
        tile_cache_key = f"{dataset_id}:{channel}:{scale}:{x}:{y}"

        # 1. Check processed tile cache
        if tile_cache_key in self.processed_tile_cache:
            cached_data = self.processed_tile_cache[tile_cache_key]
            if time.time() - cached_data['timestamp'] < self.processed_tile_ttl:
                print(f"Using cached processed tile data for {tile_cache_key}")
                return cached_data['data']
            else:
                del self.processed_tile_cache[tile_cache_key]

        # 2. Check empty regions cache
        if tile_cache_key in self.empty_regions_cache:
            # No TTL check, if it's in the cache, it's considered empty
            print(f"Skipping known empty tile: {tile_cache_key}")
            return None

        # Construct path to .zarray metadata
        zarray_path_in_dataset = f"{channel}/scale{scale}/.zarray"
        zarray_metadata = await self._fetch_zarr_metadata(dataset_id, zarray_path_in_dataset)

        if not zarray_metadata:
            print(f"Failed to get .zarray metadata for {dataset_id}/{zarray_path_in_dataset}")
            self._add_to_empty_regions_cache(tile_cache_key)
            return None

        try:
            z_shape = zarray_metadata["shape"]         # [total_height, total_width]
            z_chunks = zarray_metadata["chunks"]       # [chunk_height, chunk_width]
            z_dtype_str = zarray_metadata["dtype"]
            z_dtype = np.dtype(z_dtype_str)
            z_compressor_meta = zarray_metadata["compressor"]  # Can be null
            z_fill_value = zarray_metadata.get("fill_value")  # Important for empty/partial chunks

        except KeyError as e:
            print(f"Incomplete .zarray metadata for {dataset_id}/{zarray_path_in_dataset}: Missing key {e}")
            return None

        # Check chunk coordinates are within bounds of the scale array
        num_chunks_y_total = (z_shape[0] + z_chunks[0] - 1) // z_chunks[0]
        num_chunks_x_total = (z_shape[1] + z_chunks[1] - 1) // z_chunks[1]

        if not (0 <= y < num_chunks_y_total and 0 <= x < num_chunks_x_total):
            print(f"Chunk coordinates ({x}, {y}) out of bounds for {dataset_id}/{channel}/scale{scale} (max: {num_chunks_x_total-1}, {num_chunks_y_total-1})")
            self._add_to_empty_regions_cache(tile_cache_key)
            return None
        
        # Determine path to the zip file and the chunk name within that zip
        # Interpretation: {y}.zip contains a row of chunks, chunk file is named {x}
        zip_file_path_in_dataset = f"{channel}/scale{scale}/{y}.zip"
        chunk_name_in_zip = str(x)

        # Construct the full chunk download URL
        # dataset_id is the full path like "agent-lens/artifact-name"
        # self.workspace is "agent-lens"
        artifact_name_only = dataset_id.split('/')[-1]
        chunk_download_url = f"{self.server_url}/{self.workspace}/artifacts/{artifact_name_only}/zip-files/{zip_file_path_in_dataset}?path={chunk_name_in_zip}"
        
        print(f"Attempting to fetch chunk: {chunk_download_url}")
        
        http_session = await self._get_http_session()
        raw_chunk_bytes = None
        try:
            async with http_session.get(chunk_download_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    raw_chunk_bytes = await response.read()
                elif response.status == 404:
                    print(f"Chunk not found (404) at {chunk_download_url}. Treating as empty.")
                    self._add_to_empty_regions_cache(tile_cache_key)
                    # Create an empty tile using fill_value if available
                    empty_tile_data = np.full(z_chunks, z_fill_value if z_fill_value is not None else 0, dtype=z_dtype)
                    return empty_tile_data[:self.chunk_size, :self.chunk_size]  # Ensure correct output size
                else:
                    error_text = await response.text()
                    print(f"Error fetching chunk {chunk_download_url}: HTTP {response.status} - {error_text}")
                    return None  # Indicate error
        except asyncio.TimeoutError:
            print(f"Timeout fetching chunk: {chunk_download_url}")
            return None
        except aiohttp.ClientError as e:  # More specific aiohttp errors
            print(f"ClientError fetching chunk {chunk_download_url}: {e}")
            return None
        except Exception as e:  # Catch-all for other unexpected errors during fetch
            print(f"Unexpected error fetching chunk {chunk_download_url}: {e}")
            import traceback
            print(traceback.format_exc())
            return None

        if not raw_chunk_bytes:  # Should be caught by 404 or other errors, but as a safeguard
            print(f"No data received for chunk: {chunk_download_url}, though HTTP status was not an error.")
            self._add_to_empty_regions_cache(tile_cache_key)
            empty_tile_data = np.full(z_chunks, z_fill_value if z_fill_value is not None else 0, dtype=z_dtype)
            return empty_tile_data[:self.chunk_size, :self.chunk_size]

        # 4. Decompress and decode chunk data
        try:
            if z_compressor_meta is None:  # Raw, uncompressed data
                decompressed_data = raw_chunk_bytes
            else:
                codec = numcodecs.get_codec(z_compressor_meta)  # Handles filters too if defined in compressor object
                decompressed_data = codec.decode(raw_chunk_bytes)
            
            # Convert to NumPy array and reshape. Chunk shape from .zarray is [height, width]
            chunk_data = np.frombuffer(decompressed_data, dtype=z_dtype).reshape(z_chunks)
            
            # The Zarr chunk might be smaller than self.chunk_size if it's a partial edge chunk.
            # Or it could be larger if .zarray chunks are not self.chunk_size.
            # We need to return a tile of self.chunk_size.
            
            final_tile_data = np.full((self.chunk_size, self.chunk_size), 
                                       z_fill_value if z_fill_value is not None else 0, 
                                       dtype=z_dtype)
            
            # Determine the slice to copy from chunk_data and where to place it in final_tile_data
            copy_height = min(chunk_data.shape[0], self.chunk_size)
            copy_width = min(chunk_data.shape[1], self.chunk_size)
            
            final_tile_data[:copy_height, :copy_width] = chunk_data[:copy_height, :copy_width]

        except Exception as e:
            print(f"Error decompressing/decoding chunk from {chunk_download_url}: {e}")
            print(f"Metadata: dtype={z_dtype_str}, compressor={z_compressor_meta}, chunk_shape={z_chunks}")
            import traceback
            print(traceback.format_exc())
            return None  # Indicate error

        # 5. Check if tile is effectively empty (e.g., all fill_value or zeros)
        # Use a small threshold for non-zero values if fill_value is 0 or not defined
        is_empty_threshold = 10 
        if z_fill_value is not None:
            if np.all(final_tile_data == z_fill_value):
                print(f"Tile data is all fill_value ({z_fill_value}), treating as empty: {tile_cache_key}")
                self._add_to_empty_regions_cache(tile_cache_key)  # Cache as empty
                return None  # Return None for empty tiles based on fill_value
        elif np.count_nonzero(final_tile_data) < is_empty_threshold:
            print(f"Tile data is effectively empty (few non-zeros), treating as empty: {tile_cache_key}")
            self._add_to_empty_regions_cache(tile_cache_key)  # Cache as empty
            return None

        # 6. Cache the processed tile
        self.processed_tile_cache[tile_cache_key] = {
            'data': final_tile_data,
            'timestamp': time.time()
        }
        
        total_time = time.time() - start_time
        print(f"Total tile processing time for {tile_cache_key}: {total_time:.3f}s, size: {final_tile_data.nbytes/1024:.1f}KB")
        
        return final_tile_data

    # Legacy methods for backward compatibility - now use chunk-based access
    async def get_zarr_group(self, dataset_id, channel):
        """Legacy method - now returns None as we use direct chunk access instead. Timestamp is ignored."""
        print("Warning: get_zarr_group is deprecated, using direct chunk access instead. Timestamp parameter is ignored.")
        return None

    async def prime_metadata(self, dataset_alias, channel_name, scale, use_cache=True):
        """Pre-fetch .zarray metadata for a given dataset, channel, and scale."""
        print(f"Priming metadata for {dataset_alias}/{channel_name}/scale{scale} (use_cache={use_cache})")
        try:
            zarray_path = f"{channel_name}/scale{scale}/.zarray"
            await self._fetch_zarr_metadata(dataset_alias, zarray_path, use_cache=use_cache)

            zgroup_channel_path = f"{channel_name}/.zgroup"
            await self._fetch_zarr_metadata(dataset_alias, zgroup_channel_path, use_cache=use_cache)

            zgroup_root_path = ".zgroup"
            await self._fetch_zarr_metadata(dataset_alias, zgroup_root_path, use_cache=use_cache)
            print(f"Metadata priming complete for {dataset_alias}/{channel_name}/scale{scale}")
            return True
        except Exception as e:
            print(f"Error priming metadata: {e}")
            return False

    async def get_region_np_data(self, dataset_id, channel, scale, x, y, direct_region=None, width=None, height=None):
        """
        Get a region as numpy array using new HTTP chunk access
        
        Args:
            dataset_id (str): The dataset ID (e.g., "agent-lens/20250506-scan-time-lapse-...")
            channel (str): Channel name
            scale (int): Scale level
            x (int): X coordinate (chunk coordinates)
            y (int): Y coordinate (chunk coordinates)
            direct_region (tuple, optional): A tuple of (y_start, y_end, x_start, x_end) for direct region extraction.
                If provided, x and y are ignored and this region is used directly.
            width (int, optional): Desired width of the output image. If specified, the output will be resized/padded to this width.
            height (int, optional): Desired height of the output image. If specified, the output will be resized/padded to this height.
            
        Returns:
            np.ndarray: Region data as numpy array
        """
        try:
            # Determine the output dimensions
            output_width = width if width is not None else self.chunk_size
            output_height = height if height is not None else self.chunk_size
            
            # For direct region access, we need to fetch multiple chunks and stitch them together
            if direct_region is not None:
                y_start, y_end, x_start, x_end = direct_region
                
                # Get metadata to determine chunk size
                # dataset_id is now the full path like "agent-lens/20250506-scan-time-lapse-..."
                zarray_path_in_dataset = f"{channel}/scale{scale}/.zarray"
                zarray_metadata = await self._fetch_zarr_metadata(dataset_id, zarray_path_in_dataset)
                
                if not zarray_metadata:
                    print(f"Failed to get .zarray metadata for direct region access")
                    return np.zeros((output_height, output_width), dtype=np.uint8)
                
                z_chunks = zarray_metadata["chunks"]  # [chunk_height, chunk_width]
                z_dtype = np.dtype(zarray_metadata["dtype"])
                
                # Calculate which chunks we need
                chunk_y_start = y_start // z_chunks[0]
                chunk_y_end = (y_end - 1) // z_chunks[0] + 1
                chunk_x_start = x_start // z_chunks[1]
                chunk_x_end = (x_end - 1) // z_chunks[1] + 1
                
                # Create result array
                result_height = y_end - y_start
                result_width = x_end - x_start
                result = np.zeros((result_height, result_width), dtype=z_dtype)
                
                # Fetch and stitch chunks
                for chunk_y in range(chunk_y_start, chunk_y_end):
                    for chunk_x in range(chunk_x_start, chunk_x_end):
                        chunk_data = await self.get_chunk_np_data(dataset_id, channel, scale, chunk_x, chunk_y)
                        
                        if chunk_data is not None:
                            # Calculate where this chunk fits in the result
                            chunk_y_offset = chunk_y * z_chunks[0]
                            chunk_x_offset = chunk_x * z_chunks[1]
                            
                            # Calculate the slice within the chunk
                            chunk_y_slice_start = max(0, y_start - chunk_y_offset)
                            chunk_y_slice_end = min(z_chunks[0], y_end - chunk_y_offset)
                            chunk_x_slice_start = max(0, x_start - chunk_x_offset)
                            chunk_x_slice_end = min(z_chunks[1], x_end - chunk_x_offset)
                            
                            # Calculate the slice within the result
                            result_y_slice_start = max(0, chunk_y_offset - y_start + chunk_y_slice_start)
                            result_y_slice_end = result_y_slice_start + (chunk_y_slice_end - chunk_y_slice_start)
                            result_x_slice_start = max(0, chunk_x_offset - x_start + chunk_x_slice_start)
                            result_x_slice_end = result_x_slice_start + (chunk_x_slice_end - chunk_x_slice_start)
                            
                            # Copy the data
                            if (chunk_y_slice_end > chunk_y_slice_start and chunk_x_slice_end > chunk_x_slice_start and
                                result_y_slice_end > result_y_slice_start and result_x_slice_end > result_x_slice_start):
                                result[result_y_slice_start:result_y_slice_end, result_x_slice_start:result_x_slice_end] = \
                                    chunk_data[chunk_y_slice_start:chunk_y_slice_end, chunk_x_slice_start:chunk_x_slice_end]
                
                # Resize to requested dimensions if needed
                if width is not None or height is not None:
                    final_result = np.zeros((output_height, output_width), dtype=result.dtype)
                    copy_height = min(result.shape[0], output_height)
                    copy_width = min(result.shape[1], output_width)
                    final_result[:copy_height, :copy_width] = result[:copy_height, :copy_width]
                    result = final_result
                
                # Ensure data is in the right format (uint8)
                if result.dtype != np.uint8:
                    if result.dtype == np.float32 or result.dtype == np.float64:
                        # Normalize floating point data
                        if result.max() > 0:
                            result = (result / result.max() * 255).astype(np.uint8)
                        else:
                            result = np.zeros(result.shape, dtype=np.uint8)
                    else:
                        # For other integer types, scale appropriately
                        result = result.astype(np.uint8)
                
                return result
            
            else:
                # Single chunk access
                # dataset_id is the full path like "agent-lens/20250506-scan-time-lapse-..."
                chunk_data = await self.get_chunk_np_data(dataset_id, channel, scale, x, y)
                
                if chunk_data is None:
                    return np.zeros((output_height, output_width), dtype=np.uint8)
                
                # Resize to requested dimensions if needed
                if width is not None or height is not None:
                    result = np.zeros((output_height, output_width), dtype=chunk_data.dtype)
                    copy_height = min(chunk_data.shape[0], output_height)
                    copy_width = min(chunk_data.shape[1], output_width)
                    result[:copy_height, :copy_width] = chunk_data[:copy_height, :copy_width]
                    chunk_data = result
                
                # Ensure data is in the right format (uint8)
                if chunk_data.dtype != np.uint8:
                    if chunk_data.dtype == np.float32 or chunk_data.dtype == np.float64:
                        # Normalize floating point data
                        if chunk_data.max() > 0:
                            chunk_data = (chunk_data / chunk_data.max() * 255).astype(np.uint8)
                        else:
                            chunk_data = np.zeros(chunk_data.shape, dtype=np.uint8)
                    else:
                        # For other integer types, scale appropriately
                        chunk_data = chunk_data.astype(np.uint8)
                
                return chunk_data
                
        except Exception as e:
            print(f"Error getting region data: {e}")
            import traceback
            print(traceback.format_exc())
            return np.zeros((output_height, output_width), dtype=np.uint8)

    async def get_region_bytes(self, dataset_id, channel, scale, x, y):
        """Serve a region as PNG bytes. Timestamp is ignored."""
        try:
            # Get region data as numpy array
            region_data = await self.get_region_np_data(dataset_id, channel, scale, x, y)
            
            if region_data is None:
                print(f"No numpy data for region {dataset_id}/{channel}/{scale}/{x}/{y}, returning blank image.")
                # Create a blank image
                pil_image = Image.new("L", (self.chunk_size, self.chunk_size), color=0) 
            else:
                try:
                    # Ensure data is in a suitable range for image conversion if necessary
                    if region_data.dtype == np.uint16:
                        # Basic windowing for uint16: scale to uint8
                        scaled_data = (region_data / 256).astype(np.uint8)
                        pil_image = Image.fromarray(scaled_data)
                    elif region_data.dtype == np.float32 or region_data.dtype == np.float64:
                        # Handle float data: normalize to 0-255 for PNG
                        min_val, max_val = np.min(region_data), np.max(region_data)
                        if max_val > min_val:
                            normalized_data = ((region_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else: # Flat data
                            normalized_data = np.zeros_like(region_data, dtype=np.uint8)
                        pil_image = Image.fromarray(normalized_data)
                    else: # Assume uint8 or other directly compatible types
                        pil_image = Image.fromarray(region_data)
                except Exception as e:
                    print(f"Error converting numpy region to PIL Image: {e}. Data type: {region_data.dtype}, shape: {region_data.shape}")
                    pil_image = Image.new("L", (self.chunk_size, self.chunk_size), color=0) # Fallback to blank
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG") # Default PNG compression
            return buffer.getvalue()
        except Exception as e:
            print(f"Error in get_region_bytes: {str(e)}")
            blank_image = Image.new("L", (self.chunk_size, self.chunk_size), color=0)
            buffer = io.BytesIO()
            blank_image.save(buffer, format="PNG")
            return buffer.getvalue()

    async def get_region_base64(self, dataset_id, channel, scale, x, y):
        """Serve a region as base64 string. Timestamp is ignored."""
        region_bytes = await self.get_region_bytes(dataset_id, channel, scale, x, y)
        return base64.b64encode(region_bytes).decode('utf-8')

    async def test_zarr_access(self, dataset_id=None, channel=None, bypass_cache=False):
        """
        Test function to verify Zarr chunk access is working correctly.
        Attempts to access a known chunk.
        
        Args:
            dataset_id (str, optional): The dataset ID to test. Defaults to a standard test dataset.
            channel (str, optional): The channel to test. Defaults to a standard test channel.
            bypass_cache (bool, optional): If True, bypasses metadata cache for this test. Defaults to False.
            
        Returns:
            dict: A dictionary with status, success flag, and additional info.
        """
        try:
            # Use default values if not provided
            dataset_id = dataset_id or "agent-lens/20250506-scan-time-lapse-2025-05-06_17-56-38"
            channel = channel or "BF_LED_matrix_full"
            
            print(f"Testing Zarr chunk access for dataset: {dataset_id}, channel: {channel}, bypass_cache: {bypass_cache}")
            
            scale = 0 # Typically testing scale0
            print(f"Attempting to prime metadata for dataset: {dataset_id}, channel: {channel}, scale: {scale}")
            # Pass use_cache=!bypass_cache
            metadata_primed = await self.prime_metadata(dataset_id, channel, scale, use_cache=not bypass_cache)
            
            if not metadata_primed: # prime_metadata now returns True/False
                return {
                    "status": "error", 
                    "success": False, 
                    "message": "Failed to prime metadata for test chunk."
                }
            
            return {
                "status": "ok",
                "success": True,
                "message": f"Successfully primed metadata for test chunk (bypass_cache={bypass_cache})."
            }
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Error in test_zarr_access: {str(e)}")
            print(error_traceback)
            
            return {
                "status": "error",
                "success": False,
                "message": f"Error accessing Zarr: {str(e)}",
                "error": str(e),
                "traceback": error_traceback
            }
