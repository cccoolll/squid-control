import os  
import zarr
import numpy as np  
from flask import Flask, send_file, abort  
from io import BytesIO  
from PIL import Image  
  
app = Flask(__name__)  
  
# Path to the folder containing Zarr files  
ZARR_FOLDER = "/media/reef/harddisk/stitched_output_whole_view"  
CHANNEL_NAME = "stitched_images"  # Fixed channel name  
def create_blank_tile(tile_size=256):  
    """  
    Create a blank tile (all black) of the given size.  
    """  
    blank_image = Image.new("L", (tile_size, tile_size), color=0)  
    buffer = BytesIO()  
    blank_image.save(buffer, format="PNG")  
    buffer.seek(0)  
    return buffer  
  
def get_tile_from_zarr(zarr_path, z, x, y):  
    """  
    Fetch a tile from the Zarr file based on the z, x, y parameters.  
    """  
    # Open the Zarr file  
    zarr_group = zarr.open_group(f"{zarr_path}/Fluorescence_488_nm_Ex", mode="r")  

    # Dynamically determine the number of scales  
    zarr_scales = [key for key in zarr_group.keys() if key.startswith("scale")]  
    zarr_scales.sort(key=lambda s: int(s.replace("scale", "")))  

    # Map OpenLayers zoom levels to Zarr scales  
    zarr_zoom_mapping = {i: len(zarr_scales) - 1 - i for i in range(len(zarr_scales))}  

    # Ensure the requested zoom level is within bounds  
    if z not in zarr_zoom_mapping:  
        print("Requested zoom level is out of bounds")
        return create_blank_tile() 


    # Get the corresponding Zarr scale  
    zarr_scale_index = zarr_zoom_mapping[z]  
    scale_key = f"scale{zarr_scale_index}"  

    # Ensure the scale exists in the Zarr file  
    if scale_key not in zarr_group:  
        print("Requested scale does not exist in the Zarr file")
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
    print(f'The image is {image.size}')

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
  
@app.route("/tile/<int:z>/<int:x>/<int:y>.png")  
def get_tile(z, x, y):  
    """  
    Serve a tile for the fixed channel and z, x, y parameters.  
    """  
    try:  
        # Path to the Zarr file for the fixed channel  
        zarr_path = os.path.join(ZARR_FOLDER, f"{CHANNEL_NAME}.zarr")  
  
        # Check if the Zarr file exists  
        if not os.path.exists(zarr_path):  
            abort(404, description=f"Channel '{CHANNEL_NAME}' not found.")  
  
        # Fetch the tile from the Zarr file  
        tile = get_tile_from_zarr(zarr_path, z, x, y)  
  
        # Return the tile as a PNG image  
        return send_file(tile, mimetype="image/png")  
  
    except Exception as e:  
        abort(500, description=str(e))  
  
if __name__ == "__main__":  
    app.run(host="0.0.0.0", port=5000, debug=True)  