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

    def _artifact_alias(self, name):
        """
        Generate an alias for the artifact.

        Args:
            name (str): The artifact name.

        Returns:
            str: The artifact alias.
        """
        return f"agent-lens-{name}"

    def _artifact_id(self, workspace, name):
        """
        Generate the artifact ID.

        Args:
            workspace (str): The workspace.
            name (str): The artifact name.

        Returns:
            str: The artifact ID.
        """
        return f"{workspace}/{self._artifact_alias(name)}"

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
class ZarrTileManager:
    def __init__(self):
        self.artifact_manager = None
        self.artifact_manager_server = None
        self.workspace = "agent-lens"  # Default workspace
        self.tile_size = 256  # Default chunk size for Zarr
        self.channels = [
            "BF_LED_matrix_full",
            "Fluorescence_405_nm_Ex",
            "Fluorescence_488_nm_Ex",
            "Fluorescence_561_nm_Ex",
            "Fluorescence_638_nm_Ex"
        ]
        self.zarr_groups_cache = {}  # Cache for open Zarr groups
        self.is_running = True
        self.session = None
        self.default_timestamp = "2025-04-29_16-38-27"  # Set a default timestamp

    async def connect(self, workspace_token=None, server_url="https://hypha.aicell.io"):
        """Connect to the Artifact Manager service"""
        try:
            token = workspace_token or os.environ.get("WORKSPACE_TOKEN")
            if not token:
                raise ValueError("Workspace token not provided")
            
            self.artifact_manager_server = await connect_to_server({
                "name": "zarr-tile-client",
                "server_url": server_url,
                "token": token,
            })
            
            self.artifact_manager = SquidArtifactManager()
            await self.artifact_manager.connect_server(self.artifact_manager_server)
            
            # Initialize aiohttp session for any HTTP requests
            self.session = aiohttp.ClientSession()
            
            print("ZarrTileManager connected successfully")
            return True
        except Exception as e:
            print(f"Error connecting to artifact manager: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    async def close(self):
        """Close the tile manager and cleanup resources"""
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
        
        if cache_key in self.zarr_groups_cache:
            print(f"Using cached Zarr group for {cache_key}")
            return self.zarr_groups_cache[cache_key]
        
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
            
            return zarr_group
        except Exception as e:
            print(f"Error getting Zarr group: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    async def get_tile_np_data(self, dataset_id, timestamp, channel, scale, x, y):
        """
        Get a tile as numpy array using Zarr for efficient access
        
        Args:
            dataset_id (str): The dataset ID (workspace/artifact_alias)
            timestamp (str): The timestamp folder 
            channel (str): Channel name
            scale (int): Scale level
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            np.ndarray: Tile data as numpy array
        """
        try:
            # Use default timestamp if none provided
            timestamp = timestamp or self.default_timestamp
            
            # Get or create the zarr group
            zarr_group = await self.get_zarr_group(dataset_id, timestamp, channel)
            if not zarr_group:
                return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
            
            # Navigate to the right array in the Zarr hierarchy
            # The exact path will depend on your Zarr structure
            try:
                # Assuming a structure like zarr_group['scale{scale}']
                # You might need to adjust this path based on your actual Zarr structure
                scale_array = zarr_group[f'scale{scale}'] 
                
                # Get the specific chunk/tile - adjust slicing as needed
                tile_data = scale_array[y*self.tile_size:(y+1)*self.tile_size, 
                                       x*self.tile_size:(x+1)*self.tile_size]
                
                # Make sure we have a properly shaped array
                if tile_data.shape != (self.tile_size, self.tile_size):
                    # Resize or pad if necessary
                    result = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                    h, w = tile_data.shape
                    result[:min(h, self.tile_size), :min(w, self.tile_size)] = tile_data[:min(h, self.tile_size), :min(w, self.tile_size)]
                    return result
                
                return tile_data
            except KeyError as e:
                print(f"Error accessing Zarr array path: {e}")
                # Try an alternative path structure if needed
                try:
                    # Alternative path structure if your zarr is organized differently
                    tile_data = zarr_group[y, x]  # Simplified example
                    return tile_data
                except Exception:
                    return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        except Exception as e:
            print(f"Error getting tile data: {e}")
            import traceback
            print(traceback.format_exc())
            return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

    async def get_tile_bytes(self, dataset_id, timestamp, channel, scale, x, y):
        """Serve a tile as PNG bytes"""
        try:
            # Use default timestamp if none provided
            timestamp = timestamp or self.default_timestamp
            
            # Get tile data as numpy array
            tile_data = await self.get_tile_np_data(dataset_id, timestamp, channel, scale, x, y)
            
            # Convert to PNG bytes
            image = Image.fromarray(tile_data)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            print(f"Error in get_tile_bytes: {str(e)}")
            blank_image = Image.new("L", (self.tile_size, self.tile_size), color=0)
            buffer = io.BytesIO()
            blank_image.save(buffer, format="PNG")
            return buffer.getvalue()

    async def get_tile_base64(self, dataset_id, timestamp, channel, scale, x, y):
        """Serve a tile as base64 string"""
        # Use default timestamp if none provided
        timestamp = timestamp or self.default_timestamp
        
        tile_bytes = await self.get_tile_bytes(dataset_id, timestamp, channel, scale, x, y)
        return base64.b64encode(tile_bytes).decode('utf-8')
