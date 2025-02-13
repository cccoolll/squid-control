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

# Constants
SERVER_URL = "https://hypha.aicell.io"
WORKSPACE_TOKEN = os.getenv("AGENT_LENS_WORKSPACE_TOKEN")
ARTIFACT_ALIAS = "microscopy-tiles-complete"
DEFAULT_CHANNEL = "BF_LED_matrix_full"

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
            files = await self.artifact_manager.list_files(ARTIFACT_ALIAS, dir_path=dir_path)
            return files
        except Exception as e:
            print(f"Error listing files: {str(e)}")
            return []

    async def get_tile(self, channel: str, scale: int, x: int, y: int) -> np.ndarray:
        """Get a specific tile from the artifact manager."""
        try:
            files = await self.list_files(channel, scale)
            file_path = f"{channel}/scale{scale}/{y}.{x}"

            if not any(f['name'] == f"{y}.{x}" for f in files):
                print(f"Tile not found: {file_path}")
                return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

            get_url = await self.artifact_manager.get_file(
                ARTIFACT_ALIAS,
                file_path=file_path
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(get_url) as response:
                    if response.status == 200:
                        compressed_data = await response.read()
                        try:
                            decompressed_data = self.compressor.decode(compressed_data)
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

async def start_server(server_url):
    """Start the Hypha server and register the tile streaming service."""
    # Load environment variables
    dotenv.load_dotenv()
    token = os.getenv("AGENT_LENS_WORKSPACE_TOKEN")

    # Connect to the Hypha server
    server = await connect_to_server({
        "server_url": server_url,
        "token": token,
        "workspace": "agent-lens",
    })

    # Initialize tile manager
    tile_manager = TileManager()
    await tile_manager.connect()

    async def get_tile(channel_name: str, z: int, x: int, y: int):
        """Serve a tile as bytes"""
        try:
            print(f"Backend: Fetching tile z={z}, x={x}, y={y}")
            if channel_name is None:
                channel_name = DEFAULT_CHANNEL

            # Get tile data using TileManager
            tile_data = await tile_manager.get_tile(channel_name, z, x, y)

            # Convert to PNG bytes
            image = Image.fromarray(tile_data)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        except Exception as e:
            print(f"Error in get_tile: {str(e)}")
            blank_image = Image.new("L", (tile_manager.tile_size, tile_manager.tile_size), color=0)
            buffer = io.BytesIO()
            blank_image.save(buffer, format="PNG")
            return buffer.getvalue()

    async def get_tile_base64(channel_name: str, z: int, x: int, y: int):
        """Serve a tile as base64 string"""
        tile_bytes = await get_tile(channel_name, z, x, y)
        return base64.b64encode(tile_bytes).decode('utf-8')

    # Register the service with Hypha
    service_info = await server.register_service({
        "name": "Tile Streaming Service (Artifact Manager)",
        "id": "microscope_tile_service_artifact",
        "config": {
            "visibility": "public",
            "require_context": False,
        },
        "get_tile": get_tile,
        "get_tile_base64": get_tile_base64,
    })

    print(f"Service registered successfully!")
    print(f"Service ID: {service_info.id}")
    print(f"Workspace: {server.config.workspace}")
    print(f"Test URL: {server_url}/services/{service_info.id}")

    # Keep the server running
    await server.serve()

if __name__ == "__main__":
    server_url = "https://hypha.aicell.io"
    asyncio.run(start_server(server_url))