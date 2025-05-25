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



# Constants
SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.environ.get("AGENT_LENS_WORKSPACE_TOKEN")
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
        self.empty_regions_ttl = 40 * 60  # 40 minutes in seconds
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

    async def connect(self, workspace_token=None, server_url="https://hypha.aicell.io"):
        """Connect to the Artifact Manager service and initialize http session."""
        try:
            self.server_url = server_url.rstrip('/') # Ensure no trailing slash
            token = workspace_token or os.environ.get("SQUID_WORKSPACE_TOKEN")
            if not token:
                raise ValueError("Workspace token not provided")
            
            self.artifact_manager_server = await connect_to_server({
                "name": "zarr-image-client",
                "server_url": server_url,
                "token": token,
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
        """Add a region key to the empty regions cache with expiration"""
        # Set expiration time
        expiry_time = time.time() + self.empty_regions_ttl
        
        # Add to cache
        self.empty_regions_cache[key] = expiry_time
        
        # Clean up if cache is too large
        if len(self.empty_regions_cache) > self.empty_regions_cache_size:
            # Get the entries sorted by expiry time (oldest first)
            sorted_entries = sorted(
                self.empty_regions_cache.items(),
                key=lambda item: item[1]
            )
            
            # Remove oldest 25% of entries
            entries_to_remove = len(self.empty_regions_cache) // 4
            for i in range(entries_to_remove):
                if i < len(sorted_entries):
                    del self.empty_regions_cache[sorted_entries[i][0]]
            
            print(f"Cleaned up {entries_to_remove} oldest entries from empty regions cache")

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
            expiry_time = self.empty_regions_cache[tile_cache_key]
            if time.time() < expiry_time:
                print(f"Skipping known empty tile: {tile_cache_key}")
                return None
            else:
                del self.empty_regions_cache[tile_cache_key]

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
