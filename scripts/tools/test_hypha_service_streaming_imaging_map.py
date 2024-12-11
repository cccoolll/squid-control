import asyncio  
import os  
import io  
import dotenv  
from hypha_rpc import connect_to_server  
from PIL import Image  
import zarr  
import numpy as np  
  
# Constants from streaming_imaging_map  
ZARR_FOLDER = "/media/reef/harddisk/test_stitching/ome_ngff_output/"  
CHANNEL_NAME = "Fluorescence_561_nm"  
  
def create_blank_tile(tile_size=256):  
    """Create a blank tile (all black) of the given size."""  
    blank_image = Image.new("L", (tile_size, tile_size), color=0)  
    buffer = io.BytesIO()  
    blank_image.save(buffer, format="PNG")  
    buffer.seek(0)  
    return buffer  
  
def get_tile_from_zarr(zarr_path, z=0, x=0, y=0):  
    """Fetch a tile from the Zarr file based on the z, x, y parameters."""  
    try:  
        # Open the Zarr file  
        zarr_group = zarr.open_group(zarr_path, mode="r")  
  
        # Dynamically determine the number of scales  
        zarr_scales = [key for key in zarr_group.keys() if key.startswith("scale")]  
        zarr_scales.sort(key=lambda s: int(s.replace("scale", "")))  
  
        # Map OpenLayers zoom levels to Zarr scales  
        zarr_zoom_mapping = {i: len(zarr_scales) - 1 - i for i in range(len(zarr_scales))}  
  
        # Ensure the requested zoom level is within bounds  
        if z not in zarr_zoom_mapping:  
            return create_blank_tile()  
  
        # Get the corresponding Zarr scale  
        zarr_scale_index = zarr_zoom_mapping[z]  
        scale_key = f"scale{zarr_scale_index}"  
  
        # Ensure the scale exists in the Zarr file  
        if scale_key not in zarr_group:  
            return create_blank_tile()  
  
        # Get the dataset for the requested zoom level  
        dataset = zarr_group[scale_key]  
  
        # Calculate the tile size (assuming 256x256 tiles)  
        tile_size = 256  
  
        # Calculate the pixel range for the requested tile  
        x_start = x * tile_size  
        x_end = (x + 1) * tile_size  
        y_start = y * tile_size  
        y_end = (y + 1) * tile_size  
  
        # Check if the requested tile is within the dataset bounds  
        if x_start >= dataset.shape[1] or y_start >= dataset.shape[0]:  
            return create_blank_tile()  
  
        # Clip the tile to the dataset bounds  
        x_end = min(x_end, dataset.shape[1])  
        y_end = min(y_end, dataset.shape[0])  
  
        # Extract the tile data  
        tile_data = dataset[y_start:y_end, x_start:x_end]  
  
        # Convert the tile data to an image  
        image = Image.fromarray(tile_data)  
        image = image.convert("L")  # Convert to grayscale  
  
        # Save the image to a BytesIO object as PNG  
        buffer = io.BytesIO()  
        image.save(buffer, format="PNG")  
        buffer.seek(0)  
  
        return buffer  
  
    except Exception as e:  
        print(f"Error in get_tile_from_zarr: {str(e)}")  
        return create_blank_tile()  
  
async def start_server(server_url):  
    """Start the Hypha server and register the tile streaming service."""  
    # Load environment variables  
    dotenv.load_dotenv()  
    token = os.getenv("SQUID_WORKSPACE_TOKEN")  
  
    # Connect to the Hypha server  
    server = await connect_to_server({  
        "server_url": server_url,  
        "token": token,  
        "workspace": "squid-control",  
    })  
  
    async def get_tile(z: int, x: int, y: int):  
        """Serve a tile for the fixed channel and z, x, y parameters."""  
        try:  
            print(f"Backend: Fetching tile z={z}, x={x}, y={y}")  
              
            # Path to the Zarr file for the fixed channel  
            zarr_path = os.path.join(ZARR_FOLDER, f"{CHANNEL_NAME}.zarr")  
  
            # Check if the Zarr file exists  
            if not os.path.exists(zarr_path):  
                print(f"Zarr file not found: {zarr_path}")  
                return create_blank_tile()  
  
            # Fetch the tile from the Zarr file  
            tile_buffer = get_tile_from_zarr(zarr_path, z, x, y)

            tile_bytes = tile_buffer.getvalue()  
            print(f"Successfully fetched tile, size={len(tile_bytes)} bytes")  
            return tile_bytes  # Return bytes directly instead of BytesIO  

        except Exception as e:  
            print(f"Error getting tile: {str(e)}")  
            return create_blank_tile().getvalue()  # Return bytes from blank tile 
  
        except Exception as e:  
            print(f"Error getting tile: {str(e)}")  
            return create_blank_tile()  
  
    # Register the service with Hypha  
    service_info = await server.register_service({  
        "name": "Tile Streaming Service",  
        "id": "tile-streaming",  
        "config": {  
            "visibility": "public",  
            "require_context": False,  
        },  
        "get_tile": get_tile,  
    })  
  
    print(f"Service registered successfully!")  
    print(f"Service ID: {service_info.id}")  
    print(f"Workspace: {server.config.workspace}")  
    print(f"Test URL: {server_url}/{server.config.workspace}/services/{service_info.id}/get_tile?z=0&x=0&y=0")  
  
    # Keep the server running  
    await server.serve()  
  
if __name__ == "__main__":  
    server_url = "https://hypha.aicell.io"  
    asyncio.run(start_server(server_url))  