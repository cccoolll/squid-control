# step2_get_tiles.py
import os
import asyncio
import aiohttp
import numpy as np
from hypha_rpc import connect_to_server
from dotenv import load_dotenv
from PIL import Image
import io
import zarr
import numcodecs
import blosc
load_dotenv()

SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("AGENT_LENS_WORKSPACE_TOKEN")
ARTIFACT_ALIAS = "microscopy-tiles-complete"

def try_image_open(image_data: bytes) -> np.ndarray:
    """
    Attempt to interpret the bytes as a Zarr array.
    Returns a numpy array if successful, otherwise raises an exception.
    """
    try:
        # Open the bytes as a Zarr array
        with io.BytesIO(image_data) as byte_stream:
            store = zarr.ZipStore(byte_stream, mode='r')  # Use ZipStore with the BytesIO object
            zarr_array = zarr.open(store, mode='r')
            return np.array(zarr_array)
    except Exception as e:
        print(f"Zarr decoding failed: {str(e)}")
        raise

class TileManager:
    def __init__(self):
        self.api = None
        self.artifact_manager = None
        self.tile_size = 2048
        self.channels = [
            "BF_LED_matrix_full",
            "Fluorescence_405_nm_Ex",
            "Fluorescence_488_nm_Ex",
            "Fluorescence_561_nm_Ex",
            "Fluorescence_638_nm_Ex"
        ]
        self.compressor = numcodecs.Blosc(
            cname='zstd',
            clevel=5,
            shuffle=blosc.SHUFFLE,
            blocksize=0
        )
    async def connect(self):
        """Connect to the Artifact Manager service"""
        self.api = await connect_to_server({
            "name": "test-client",
            "server_url": SERVER_URL,
            "token": WORKSPACE_TOKEN
        })
        self.artifact_manager = await self.api.get_service("public/artifact-manager")

    async def list_files(self, channel: str, scale: int):
        """List available files for a specific channel and scale"""
        try:
            dir_path = f"{channel}/scale{scale}"
            files = await self.artifact_manager.list_files(ARTIFACT_ALIAS, dir_path=dir_path, limit=3000)
            return files
        except Exception as e:
            print(f"Error listing files: {str(e)}")
            return []

    async def get_tile(self, channel: str, scale: int, x: int, y: int) -> np.ndarray:
            """
            Get a specific tile from the artifact manager.

            Args:
                channel (str): Channel name (e.g., "BF_LED_matrix_full")
                scale (int): Scale level (0-3)
                x (int): X coordinate of the tile
                y (int): Y coordinate of the tile

            Returns:
                np.ndarray: The tile image as a numpy array
            """
            try:
                # First, list available files to check if the tile exists
                files = await self.list_files(channel, scale)
                file_path = f"{channel}/scale{scale}/{y}.{x}"

                if not any(f['name'] == f"{y}.{x}" for f in files):
                    print(f"Tile not found: {file_path}")
                    return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

                # Get the pre-signed URL for the file
                get_url = await self.artifact_manager.get_file(
                    ARTIFACT_ALIAS,
                    file_path=file_path
                )

                # Download the tile using aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(get_url) as response:
                        if response.status == 200:
                            # Read the compressed binary data
                            compressed_data = await response.read()

                            try:
                                # Decompress the data using blosc
                                decompressed_data = self.compressor.decode(compressed_data)

                                # Convert to numpy array with correct shape and dtype
                                tile_data = np.frombuffer(decompressed_data, dtype=np.uint8)
                                tile_data = tile_data.reshape((self.tile_size, self.tile_size))

                                return tile_data

                            except Exception as e:
                                print(f"Error processing tile data: {str(e)}")
                                return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                        else:
                            print(f"Failed to download tile: {response.status}")
                            return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

            except Exception as e:
                print(f"Error getting tile {file_path}: {str(e)}")
                return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        
    async def get_region(self, channel: str, scale: int, x_start: int, y_start: int,
                         width: int, height: int) -> np.ndarray:
        """
        Get a rectangular region of tiles and stitch them together.

        Args:
            channel (str): Channel name
            scale (int): Scale level (0-10)
            x_start (int): Starting X coordinate (pixels)
            y_start (int): Starting Y coordinate (pixels)
            width (int): Width of the region in pixels
            height (int): Height of the region in pixels

        Returns:
            np.ndarray: Stitched image of the region
        """
        # 1) Calculate tile coordinates needed
        start_tile_x = x_start // self.tile_size
        start_tile_y = y_start // self.tile_size
        end_tile_x = (x_start + width  - 1) // self.tile_size + 1
        end_tile_y = (y_start + height - 1) // self.tile_size + 1

        # 2) Collect all required tile fetch tasks
        tasks = []
        for ty in range(start_tile_y, end_tile_y):
            for tx in range(start_tile_x, end_tile_x):
                tasks.append(self.get_tile(channel, scale, tx, ty))

        # 3) Fetch all tiles concurrently
        tiles = await asyncio.gather(*tasks)

        # 4) Build a combined array
        tile_cols = end_tile_x - start_tile_x
        tile_rows = end_tile_y - start_tile_y
        combined_full_w = tile_cols * self.tile_size
        combined_full_h = tile_rows * self.tile_size
        combined = np.zeros((combined_full_h, combined_full_w), dtype=np.uint8)

        # 5) Stitch the tiles into the combined array
        i = 0
        for row_i in range(tile_rows):
            for col_i in range(tile_cols):
                tile_img = tiles[i]
                i += 1
                if tile_img is None:
                    continue
                # each tile is self.tile_size x self.tile_size
                y_pos = row_i * self.tile_size
                x_pos = col_i * self.tile_size
                combined[y_pos:y_pos + self.tile_size, x_pos:x_pos + self.tile_size] = tile_img

        # 6) Crop to requested region
        x_offset = x_start % self.tile_size
        y_offset = y_start % self.tile_size
        cropped = combined[y_offset:y_offset + height, x_offset:x_offset + width]
        return cropped

async def example_usage():
    manager = TileManager()
    await manager.connect()

    print(f"Listing a few files from BF_LED_matrix_full scale 0:")
    files = await manager.list_files("BF_LED_matrix_full", 0)
    print(f"The number of the tiles: {len(files)}")
    print(files[:10])

    # Example A: get a single tile (channel=BF_LED_matrix_full scale=7 tileX=5,tileY=5)
    tile = await manager.get_tile("BF_LED_matrix_full", 0, 81, 55)
    if tile is not None:
        Image.fromarray(tile).save("example_tile.png")
        print("Saved example tile to example_tile.png")

    # Example B: get a region (channel=BF_LED_matrix_full scale=7)
    #  e.g. top-left corner is (x_start=0, y_start=0), region size=512x512
    region = await manager.get_region("BF_LED_matrix_full", 2, 10000, 10000, 5000, 5000)
    if region is not None:
        Image.fromarray(region).save("example_region.png")
        print("Saved example region to example_region.png")

if __name__ == "__main__":
    asyncio.run(example_usage())

# Created/Modified files during execution:
print(["step2_get_tiles.py", "example_tile.png", "example_region.png"])