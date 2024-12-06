import os  
import zarr  
from flask import Flask, send_file, abort  
from io import BytesIO  
from PIL import Image  
  
app = Flask(__name__)  
  
# Path to the folder containing Zarr files  
ZARR_FOLDER = "/media/reef/harddisk/test_stitching/ome_ngff_output/"  
CHANNEL_NAME = "Fluorescence_405_nm"  # Fixed channel name  
  
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
    zarr_group = zarr.open_group(zarr_path, mode="r")  

    # Map OpenLayers zoom levels to Zarr scale levels  
    zarr_scales = [key for key in zarr_group.keys() if key.startswith("scale")]  
    zarr_scales.sort(key=lambda s: int(s.replace("scale", "")))  # Sort by scale level  

    # Map OpenLayers zoom level to Zarr scale  
    # Adjust this mapping as needed to match your data  
    zarr_zoom_mapping = {  
        0: 6,  # OpenLayers zoom 0 -> Zarr scale 6 (most zoomed-out)  
        1: 5,  
        2: 4,  
        3: 3,  
        4: 2,  
        5: 1,  
        6: 0,  # OpenLayers zoom 6 -> Zarr scale 0 (most detailed)  
    }  

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