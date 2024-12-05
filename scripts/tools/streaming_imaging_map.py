import os  
import math  
import numpy as np  
import zarr  
from flask import Flask, send_file, request, abort, jsonify  
from io import BytesIO  
from PIL import Image  

app = Flask(__name__)  

# Path to the folder containing Zarr files  
ZARR_FOLDER = "/media/reef/harddisk/test_stitching/ome_ngff_output"  

def get_tile_from_zarr(zarr_path, z, x, y):  
    """  
    Fetch a tile from the Zarr file based on the z, x, y parameters.  

    Parameters:  
        zarr_path (str): Path to the Zarr file.  
        z (int): Zoom level.  
        x (int): Tile x-coordinate.  
        y (int): Tile y-coordinate.  

    Returns:  
        BytesIO: The image tile as a PNG in memory.  
    """  
    # Open the Zarr file  
    zarr_group = zarr.open_group(zarr_path, mode="r")  

    # Check if the requested zoom level exists  
    scale_key = f"scale{z}"  
    if scale_key not in zarr_group:  
        raise ValueError(f"Zoom level {z} not found in Zarr file.")  

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
        raise ValueError("Requested tile is out of bounds.")  

    # Clip the tile to the dataset bounds  
    x_end = min(x_end, dataset.shape[1])  
    y_end = min(y_end, dataset.shape[0])  

    # Extract the tile data  
    tile_data = dataset[y_start:y_end, x_start:x_end]  

    # Convert the tile data to an image  
    image = Image.fromarray(tile_data)  
    image = image.convert("L")  # Convert to grayscale  

    # Save the image to a BytesIO object as PNG  
    buffer = BytesIO()  
    image.save(buffer, format="PNG")  
    buffer.seek(0)  

    return buffer  


@app.route("/")  
def index():  
    """  
    Serve the main webpage with OpenLayers.  
    """  
    return send_file("index.html")  


@app.route("/tile/<channel>/<int:z>/<int:x>/<int:y>.png")  
def get_tile(channel, z, x, y):  
    """  
    Serve a tile for the given channel and z, x, y parameters.  
    """  
    try:  
        # Path to the Zarr file for the requested channel  
        zarr_path = os.path.join(ZARR_FOLDER, f"{channel}.zarr")  

        # Check if the Zarr file exists  
        if not os.path.exists(zarr_path):  
            abort(404, description=f"Channel '{channel}' not found.")  

        # Fetch the tile from the Zarr file  
        tile = get_tile_from_zarr(zarr_path, z, x, y)  

        # Return the tile as a PNG image  
        return send_file(tile, mimetype="image/png")  

    except ValueError as e:  
        abort(400, description=str(e))  
    except Exception as e:  
        abort(500, description=str(e))  


if __name__ == "__main__":  
    app.run(host="0.0.0.0", port=5000, debug=True)  