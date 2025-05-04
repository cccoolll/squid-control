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
from zarr.storage import LRUStoreCache, FSStore
import fsspec
import time

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
        self.zarr_groups_cache = {}  # Cache for zarr groups
        self.zarr_groups_timestamps = {}  # Timestamps for when groups were cached
        self.zarr_cache_expiry_seconds = 40 * 60  # 40 minutes in seconds

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

    async def get_zarr_group(
        self,
        workspace: str,
        artifact_alias: str,
        timestamp: str,
        channel: str,
        cache_max_size=2**28 # 256 MB LRU cache
    ):
        """
        Access a Zarr group stored within a zip file in an artifact.

        Args:
            workspace (str): The workspace containing the artifact.
            artifact_alias (str): The alias of the artifact (e.g., 'image-map-20250429-treatment-zip').
            timestamp (str): The timestamp folder name.
            channel (str): The channel name (used for the zip filename).
            cache_max_size (int, optional): Max size for LRU cache in bytes. Defaults to 2**28.

        Returns:
            zarr.Group: The root Zarr group object.
        """
        if self._svc is None:
            raise ConnectionError("Artifact Manager service not connected. Call connect_server first.")

        art_id = self._artifact_id(workspace, artifact_alias)
        zip_file_path = f"{timestamp}/{channel}.zip"
        cache_key = f"{art_id}:{timestamp}:{channel}"
        current_time = time.time()
        
        # Check if we have a cached and non-expired Zarr group
        if cache_key in self.zarr_groups_cache:
            cache_time = self.zarr_groups_timestamps.get(cache_key, 0)
            if current_time - cache_time < self.zarr_cache_expiry_seconds:
                print(f"Using cached Zarr group for {cache_key}")
                return self.zarr_groups_cache[cache_key]
            else:
                # Group is expired, remove it from cache
                print(f"Zarr group for {cache_key} has expired (created {(current_time - cache_time)/60:.1f} minutes ago). Refreshing...")
                self.zarr_groups_cache.pop(cache_key, None)
                self.zarr_groups_timestamps.pop(cache_key, None)

        try:
            print(f"Getting download URL for: {art_id}/{zip_file_path}")
            # Get the direct download URL for the zip file
            download_url = await self._svc.get_file(art_id, zip_file_path)
            print(f"Obtained download URL.")

            # Construct the URL for FSStore using fsspec's zip chaining
            store_url = f"zip::{download_url}"

            # Define the synchronous function to open the Zarr store and group
            def _open_zarr_sync(url, cache_size):
                print(f"Opening Zarr store: {url}")
                store = FSStore(url, mode="r")
                if cache_size and cache_size > 0:
                    print(f"Using LRU cache with size: {cache_size} bytes")
                    store = LRUStoreCache(store, max_size=cache_size)
                # It's generally recommended to open the root group
                root_group = zarr.group(store=store)
                print(f"Zarr group opened successfully.")
                return root_group

            # Run the synchronous Zarr operations in a thread pool
            print("Running Zarr open in thread executor...")
            zarr_group = await asyncio.to_thread(_open_zarr_sync, store_url, cache_max_size)
            
            # Cache the Zarr group for future use
            self.zarr_groups_cache[cache_key] = zarr_group
            self.zarr_groups_timestamps[cache_key] = current_time
            print(f"Cached new Zarr group for {cache_key}")
            
            return zarr_group

        except RemoteException as e:
            print(f"Error getting file URL from Artifact Manager: {e}")
            raise FileNotFoundError(f"Could not find or access zip file {zip_file_path} in artifact {art_id}") from e
        except Exception as e:
            print(f"An error occurred while accessing the Zarr group: {e}")
            import traceback
            print(traceback.format_exc())
            raise

# Constants
SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.environ.get("WORKSPACE_TOKEN")
ARTIFACT_ALIAS = "image-map-20250429-treatment-zip"
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
        self.zarr_groups_cache = {}  # Cache for open Zarr groups
        self.zarr_groups_timestamps = {}  # Timestamps for when groups were cached
        self.zarr_cache_expiry_seconds = 40 * 60  # 40 minutes in seconds
        self.is_running = True
        self.session = None
        self.default_timestamp = "2025-04-29_16-38-27"  # Set a default timestamp
        self.scale_key = 'scale0'
    async def connect(self, workspace_token=None, server_url="https://hypha.aicell.io"):
        """Connect to the Artifact Manager service"""
        try:
            token = workspace_token or os.environ.get("WORKSPACE_TOKEN")
            if not token:
                raise ValueError("Workspace token not provided")
            
            self.artifact_manager_server = await connect_to_server({
                "name": "zarr-image-client",
                "server_url": server_url,
                "token": token,
            })
            
            self.artifact_manager = SquidArtifactManager()
            await self.artifact_manager.connect_server(self.artifact_manager_server)
            
            # Initialize aiohttp session for any HTTP requests
            self.session = aiohttp.ClientSession()
            
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
        
        # Close the cached Zarr groups
        self.zarr_groups_cache.clear()
        
        # Close the aiohttp session
        if self.session:
            await self.session.close()
            self.session = None
        
        # Disconnect from the server
        if self.artifact_manager_server:
            await self.artifact_manager_server.disconnect()
            self.artifact_manager_server = None
            self.artifact_manager = None

    async def get_zarr_group(self, dataset_id, timestamp, channel):
        """Get (or reuse from cache) a Zarr group for a specific dataset"""
        cache_key = f"{dataset_id}:{timestamp}:{channel}"
        current_time = time.time()
        
        # Check if the cached group exists and is still valid (less than 40 minutes old)
        if cache_key in self.zarr_groups_cache:
            cache_time = self.zarr_groups_timestamps.get(cache_key, 0)
            if current_time - cache_time < self.zarr_cache_expiry_seconds:
                #print(f"Using cached Zarr group for {cache_key}")
                return self.zarr_groups_cache[cache_key]
            else:
                # Group is expired, remove it from cache
                print(f"Zarr group for {cache_key} has expired (created {(current_time - cache_time)/60:.1f} minutes ago). Refreshing...")
                self.zarr_groups_cache.pop(cache_key, None)
                self.zarr_groups_timestamps.pop(cache_key, None)
        
        try:
            # We no longer need to parse the dataset_id into workspace and artifact_alias
            # Just use the dataset_id directly since it's already the full path
            print(f"Accessing artifact at: {dataset_id}/{timestamp}/{channel}.zip")
            
            # Get the direct download URL for the zip file
            zip_file_path = f"{timestamp}/{channel}.zip"
            download_url = await self.artifact_manager._svc.get_file(dataset_id, zip_file_path)
            
            # Construct the URL for FSStore using fsspec's zip chaining
            store_url = f"zip::{download_url}"
            
            # Define the synchronous function to open the Zarr store and group
            def _open_zarr_sync(url, cache_size):
                print(f"Opening Zarr store: {url}")
                store = FSStore(url, mode="r")
                if cache_size and cache_size > 0:
                    store = LRUStoreCache(store, max_size=cache_size)
                # It's generally recommended to open the root group
                root_group = zarr.group(store=store)
                print(f"Zarr group opened successfully.")
                return root_group
                
            # Run the synchronous Zarr operations in a thread pool
            print("Running Zarr open in thread executor...")
            zarr_group = await asyncio.to_thread(_open_zarr_sync, store_url, 2**28)  # Using default cache size
            
            # Cache the Zarr group for future use
            self.zarr_groups_cache[cache_key] = zarr_group
            self.zarr_groups_timestamps[cache_key] = current_time
            print(f"Cached new Zarr group for {cache_key}")
            
            return zarr_group
        except Exception as e:
            print(f"Error getting Zarr group: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    async def get_region_np_data(self, dataset_id, timestamp, channel, scale, x, y, direct_region=None, width=None, height=None):
        """
        Get a region as numpy array using Zarr for efficient access
        
        Args:
            dataset_id (str): The dataset ID (workspace/artifact_alias)
            timestamp (str): The timestamp folder 
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
            # Use default timestamp if none provided
            timestamp = timestamp or self.default_timestamp
            
            # Determine the output dimensions
            output_width = width if width is not None else self.chunk_size
            output_height = height if height is not None else self.chunk_size
            
            # Get or create the zarr group
            zarr_group = await self.get_zarr_group(dataset_id, timestamp, channel)
            if not zarr_group:
                print(f"No Zarr group found for {dataset_id}/{timestamp}/{channel}")
                return np.zeros((output_height, output_width), dtype=np.uint8)
            # Navigate to the right array in the Zarr hierarchy
            try:
                # Debug: Print available keys in the zarr group
                print(f"Available keys in Zarr group: {list(zarr_group.keys())}")
                # Get the scale array
                scale_array = zarr_group[self.scale_key]
                print(f"Scale array shape: {scale_array.shape}, dtype: {scale_array.dtype}")
                
                # Ensure coordinates are valid
                array_shape = scale_array.shape
                if len(array_shape) < 2:
                    raise ValueError(f"Scale array has unexpected shape: {array_shape}")
                
                # If direct region is provided, use it directly
                if direct_region is not None:
                    y_start, y_end, x_start, x_end = direct_region
                    # Ensure coordinates are within bounds
                    y_start = max(0, min(y_start, array_shape[0] - 1))
                    y_end = max(y_start + 1, min(y_end, array_shape[0]))
                    x_start = max(0, min(x_start, array_shape[1] - 1))
                    x_end = max(x_start + 1, min(x_end, array_shape[1]))
                else:
                    # Calculate the bounds based on chunk coordinates
                    y_start = y * self.chunk_size
                    x_start = x * self.chunk_size
                    
                    # Ensure coordinates are within bounds
                    if y_start >= array_shape[0] or x_start >= array_shape[1]:
                        print(f"Coordinates out of bounds: y={y_start}, x={x_start}, shape={array_shape}")
                        return np.zeros((output_height, output_width), dtype=np.uint8)
                    
                    # Get the specific chunk/region - adjust slicing as needed
                    y_end = min(y_start + self.chunk_size, array_shape[0])
                    x_end = min(x_start + self.chunk_size, array_shape[1])
                
                print(f"Reading region from y={y_start} to y={y_end}, x={x_start} to x={x_end}")
                region_data = scale_array[y_start:y_end, x_start:x_end]
                
                # Debug info about retrieved data
                print(f"Region data shape: {region_data.shape}, dtype: {region_data.dtype}, " 
                      f"min: {region_data.min() if region_data.size > 0 else 'N/A'}, "
                      f"max: {region_data.max() if region_data.size > 0 else 'N/A'}")
                
                # If width and height are specified, resize the output to those dimensions
                if width is not None or height is not None:
                    # Create an empty array with the requested dimensions
                    result = np.zeros((output_height, output_width), dtype=region_data.dtype or np.uint8)
                    
                    # Copy the retrieved data into the result array
                    h, w = region_data.shape
                    result[:min(h, output_height), :min(w, output_width)] = region_data[:min(h, output_height), :min(w, output_width)]
                    
                    print(f"Resized region data from {region_data.shape} to {result.shape}")
                    
                    # Ensure data is in the right format (uint8)
                    if result.dtype != np.uint8:
                        print(f"Converting resized data from {result.dtype} to uint8")
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
                    
                # For direct region requests with no specific width/height, return as is
                elif direct_region is not None:
                    # Ensure data is in the right format (uint8)
                    if region_data.dtype != np.uint8:
                        print(f"Converting direct region data from {region_data.dtype} to uint8")
                        if region_data.dtype == np.float32 or region_data.dtype == np.float64:
                            # Normalize floating point data
                            if region_data.max() > 0:
                                region_data = (region_data / region_data.max() * 255).astype(np.uint8)
                            else:
                                region_data = np.zeros(region_data.shape, dtype=np.uint8)
                        else:
                            # For other integer types, scale appropriately
                            region_data = region_data.astype(np.uint8)
                    
                    return region_data
                
                # Make sure we have a properly shaped array for chunk-based requests
                elif region_data.shape != (self.chunk_size, self.chunk_size):
                    # Resize or pad if necessary
                    print(f"Padding region from {region_data.shape} to {(self.chunk_size, self.chunk_size)}")
                    result = np.zeros((self.chunk_size, self.chunk_size), dtype=region_data.dtype or np.uint8)
                    h, w = region_data.shape
                    result[:min(h, self.chunk_size), :min(w, self.chunk_size)] = region_data[:min(h, self.chunk_size), :min(w, self.chunk_size)]
                    
                    # Ensure data is in the right format (uint8)
                    if result.dtype != np.uint8:
                        print(f"Converting padded data from {result.dtype} to uint8")
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
                    
                # Default case: return the region data as is
                # Ensure data is in the right format (uint8)
                if region_data.dtype != np.uint8:
                    print(f"Converting region data from {region_data.dtype} to uint8")
                    if region_data.dtype == np.float32 or region_data.dtype == np.float64:
                        # Normalize floating point data
                        if region_data.max() > 0:
                            region_data = (region_data / region_data.max() * 255).astype(np.uint8)
                        else:
                            region_data = np.zeros(region_data.shape, dtype=np.uint8)
                    else:
                        # For other integer types, scale appropriately
                        region_data = region_data.astype(np.uint8)
                
                return region_data
                
            except KeyError as e:
                print(f"Error accessing Zarr array path: {e}")
                # If specific key approach failed, try to explore the structure and find data
                try:
                    # Simple approach: try to traverse into any sub-groups until we find an array
                    def find_array(group, depth=0, max_depth=3):
                        if depth >= max_depth:
                            return None
                        
                        for key in group.keys():
                            item = group[key]
                            
                            # Check if it's an array with at least 2 dimensions
                            if hasattr(item, 'shape') and len(item.shape) >= 2:
                                print(f"Found array at path: {key}, shape: {item.shape}")
                                return item
                            
                            # If it's a group, recursively check inside
                            elif hasattr(item, 'keys'):
                                result = find_array(item, depth+1, max_depth)
                                if result is not None:
                                    return result
                        
                        return None
                    
                    array = find_array(zarr_group)
                    if array is not None:
                        # Calculate coordinates based on the found array's dimensions
                        y_start = min(y * self.chunk_size, array.shape[0] - 1)
                        x_start = min(x * self.chunk_size, array.shape[1] - 1)
                        y_end = min(y_start + self.chunk_size, array.shape[0])
                        x_end = min(x_start + self.chunk_size, array.shape[1])
                        
                        region_data = array[y_start:y_end, x_start:x_end]
                        
                        # Pad if necessary
                        if region_data.shape != (self.chunk_size, self.chunk_size):
                            result = np.zeros((self.chunk_size, self.chunk_size), dtype=region_data.dtype or np.uint8)
                            h, w = region_data.shape
                            result[:min(h, self.chunk_size), :min(w, self.chunk_size)] = region_data[:min(h, self.chunk_size), :min(w, self.chunk_size)]
                            return result
                        
                        return region_data
                
                except Exception as nested_e:
                    print(f"Alternative array search failed: {nested_e}")
                
                # If all else fails, return an empty array
                return np.zeros((self.chunk_size, self.chunk_size), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error getting region data: {e}")
            import traceback
            print(traceback.format_exc())
            return np.zeros((self.chunk_size, self.chunk_size), dtype=np.uint8)

    async def get_region_bytes(self, dataset_id, timestamp, channel, scale, x, y):
        """Serve a region as PNG bytes"""
        try:
            # Use default timestamp if none provided
            timestamp = timestamp or self.default_timestamp
            
            # Get region data as numpy array
            region_data = await self.get_region_np_data(dataset_id, timestamp, channel, scale, x, y)
            
            # Convert to PNG bytes
            image = Image.fromarray(region_data)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            print(f"Error in get_region_bytes: {str(e)}")
            blank_image = Image.new("L", (self.chunk_size, self.chunk_size), color=0)
            buffer = io.BytesIO()
            blank_image.save(buffer, format="PNG")
            return buffer.getvalue()

    async def get_region_base64(self, dataset_id, timestamp, channel, scale, x, y):
        """Serve a region as base64 string"""
        # Use default timestamp if none provided
        timestamp = timestamp or self.default_timestamp
        
        region_bytes = await self.get_region_bytes(dataset_id, timestamp, channel, scale, x, y)
        return base64.b64encode(region_bytes).decode('utf-8')

    async def test_zarr_access(self, dataset_id=None, timestamp=None, channel=None):
        """
        Test function to verify Zarr file access is working correctly.
        Attempts to access a known chunk at coordinates (335, 384) in scale0.
        
        Args:
            dataset_id (str, optional): The dataset ID to test. Defaults to agent-lens/image-map-20250429-treatment-zip.
            timestamp (str, optional): The timestamp to use. Defaults to the default timestamp.
            channel (str, optional): The channel to test. Defaults to BF_LED_matrix_full.
            
        Returns:
            dict: A dictionary with status, success flag, and additional info about the chunk.
        """
        try:
            # Use default values if not provided
            dataset_id = dataset_id or "agent-lens/image-map-20250429-treatment-zip"
            timestamp = timestamp or self.default_timestamp
            channel = channel or "BF_LED_matrix_full"
            
            print(f"Testing Zarr access for dataset: {dataset_id}, timestamp: {timestamp}, channel: {channel}")
            
            # Get the zarr group
            zarr_group = await self.get_zarr_group(dataset_id, timestamp, channel)
            if zarr_group is None:
                return {
                    "status": "error", 
                    "success": False, 
                    "message": "Failed to get Zarr group"
                }
                
            # Directly test access to a known chunk at coordinates (335, 384) in scale0
            # We know this chunk exists for sure in our dataset
            test_y, test_x = 335, 384
            test_y_start = test_y * self.chunk_size
            test_x_start = test_x * self.chunk_size
            
            # Try to access the array
            test_array = zarr_group['scale0']
            
            # Get the chunk dimensions and make sure coordinates are in bounds
            array_shape = test_array.shape
            test_y_end = min(test_y_start + self.chunk_size, array_shape[0])
            test_x_end = min(test_x_start + self.chunk_size, array_shape[1])
            
            # Read the chunk directly
            print(f"Reading test chunk at y={test_y_start}:{test_y_end}, x={test_x_start}:{test_x_end}")
            test_chunk = test_array[test_y_start:test_y_end, test_x_start:test_x_end]
            
            # Gather statistics about the chunk for verification
            chunk_stats = {
                "shape": test_chunk.shape,
                "min": float(test_chunk.min()) if test_chunk.size > 0 else None,
                "max": float(test_chunk.max()) if test_chunk.size > 0 else None,
                "mean": float(test_chunk.mean()) if test_chunk.size > 0 else None,
                "non_zero_count": int(np.count_nonzero(test_chunk)),
                "total_size": int(test_chunk.size)
            }
            
            # Consider it successful if we got a non-empty chunk
            success = test_chunk.size > 0 and np.count_nonzero(test_chunk) > 0
            
            return {
                "status": "ok" if success else "error",
                "success": success,
                "message": "Successfully accessed test chunk" if success else "Chunk contained no data",
                "chunk_stats": chunk_stats
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
