import os
import cv2
import numpy as np
from glob import glob
import json
from tqdm import tqdm
import zarr
from skimage.transform import downscale_local_mean
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
import asyncio
import websockets
import base64
import io
from PIL import Image
import threading
# Global variable to hold the latest canvas
latest_canvas = None

async def websocket_server(websocket, path):
    """WebSocket server to stream stitched image data."""
    global latest_canvas
    while True:
        if latest_canvas is not None:
            # Convert the canvas to a PNG image
            pil_image = Image.fromarray(latest_canvas)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode the image as base64
            image_data = base64.b64encode(buffer.read()).decode('utf-8')

            # Send the base64-encoded image to the client
            await websocket.send(image_data)

        # Wait for a short interval before sending the next update
        await asyncio.sleep(0.5)

def start_websocket_server():
    """Start the WebSocket server."""
    server = websockets.serve(websocket_server, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(server)
    print("WebSocket server started on ws://0.0.0.0:8765")
    asyncio.get_event_loop().run_forever()
    
def compute_overlap_percent(deltaX, deltaY, image_width, image_height, pixel_size_xy, min_overlap=0):
    """Helper function to calculate percent overlap between images in a grid."""
    shift_x = deltaX / pixel_size_xy
    shift_y = deltaY / pixel_size_xy
    overlap_x = max(0, image_width - shift_x)
    overlap_y = max(0, image_height - shift_y)
    overlap_x = overlap_x * 100.0 / image_width
    overlap_y = overlap_y * 100.0 / image_height
    overlap = max(min_overlap, overlap_x, overlap_y)
    return overlap

def get_pixel_size(parameters, default_pixel_size=1.85, default_tube_lens_mm=50.0, default_objective_tube_lens_mm=180.0, default_magnification=40.0):
    """Calculate pixel size based on imaging parameters."""
    try:
        tube_lens_mm = float(parameters.get('tube_lens_mm', default_tube_lens_mm))
        pixel_size_um = float(parameters.get('sensor_pixel_size_um', default_pixel_size))
        objective_tube_lens_mm = float(parameters['objective'].get('tube_lens_f_mm', default_objective_tube_lens_mm))
        magnification = float(parameters['objective'].get('magnification', default_magnification))
    except KeyError:
        raise ValueError("Missing required parameters for pixel size calculation.")

    pixel_size_xy = pixel_size_um / (magnification / (objective_tube_lens_mm / tube_lens_mm))
    return pixel_size_xy

def rotate_flip_image(image, angle, flip=False):
    """Rotate an image by a specified angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    if flip:
        rotated = cv2.flip(rotated, 1)
    return rotated

def load_imaging_parameters(parameter_file):
    with open(parameter_file, "r") as f:
        parameters = json.load(f)
    return parameters

def get_image_positions(parameters):
    dx = parameters["dx(mm)"] * 1000  # Convert mm to µm
    dy = parameters["dy(mm)"] * 1000  # Convert mm to µm
    Nx = parameters["Nx"]
    Ny = parameters["Ny"]
    pixel_size_xy = get_pixel_size(parameters)
    return dx, dy, Nx, Ny, pixel_size_xy

def parse_image_filenames(image_folder):
    image_files = glob(os.path.join(image_folder, "*.bmp"))
    image_info = []
    channels = set()

    for image_file in image_files:
        filename = os.path.basename(image_file)
        parts = filename.split('_')
        if len(parts) >= 6:
            R, x_idx, y_idx, z_idx, channel_name = parts[0], parts[1], parts[2], parts[3], '_'.join(parts[4:-1])
            extension = parts[-1]
            channels.add(channel_name)
            image_info.append({
                "filepath": image_file,
                "x_idx": int(x_idx),
                "y_idx": int(y_idx),
                "z_idx": int(z_idx),
                "channel_name": channel_name,
                "extension": extension
            })
    return image_info, list(channels)

def initialize_ome_ngff(output_folder, channel_name, canvas_height, canvas_width):
    """Initialize an OME-NGFF dataset for multiscale tiled data."""
    zarr_path = os.path.join(output_folder, f"stitched_{channel_name}.zarr")
    store = parse_url(zarr_path, mode="w").store
    return store

def stream_images_to_ome_ngff(channel_name, image_info, parameters, store, rotation_angle=0):
    """Stream images into the OME-NGFF dataset dynamically."""
    dx, dy, Nx, Ny, pixel_size_xy = get_image_positions(parameters)

    # Filter images for current channel
    channel_images = [info for info in image_info if info["channel_name"] == channel_name]

    if not channel_images:
        return

    # Get image dimensions from first image
    sample_image = cv2.imread(channel_images[0]["filepath"], cv2.IMREAD_GRAYSCALE)
    img_height, img_width = sample_image.shape
    overlap_percent = min(100, compute_overlap_percent(dx, dy, img_width, img_height, pixel_size_xy))

    # Calculate effective step size (accounting for overlap)
    x_step = img_width * (1 - overlap_percent / 100)
    y_step = img_height * (1 - overlap_percent / 100)

    # Initialize full-resolution canvas
    canvas = np.zeros((Ny * img_height, Nx * img_width), dtype=np.uint16)

    # Update the global canvas for the WebSocket server
    global latest_canvas
    latest_canvas = canvas.copy()  # Copy the canvas to avoid threading issues

    print(f"OME-NGFF dataset written for channel {channel_name}")

    # Stream images into the canvas
    for info in tqdm(channel_images, desc=f"Streaming {channel_name}"):
        x_idx = info["x_idx"]
        y_idx = info["y_idx"]

        # Calculate position on canvas
        x_pos = round(x_idx * x_step)
        y_pos = round((Ny - y_idx - 1) * y_step)  # Flip y-axis for canvas

        # Read image and convert to grayscale if it's not already
        image = cv2.imread(info["filepath"], cv2.IMREAD_GRAYSCALE)

        # Rotate the image if a rotation angle is specified
        if rotation_angle != 0:
            image = rotate_flip_image(image, rotation_angle, flip=True)

        # Write the image into the canvas
        canvas[y_pos:y_pos + img_height, x_pos:x_pos + img_width] = image

    # Generate multiscale data
    multiscale = [canvas]
    for scale in [2, 4, 8]:  # Downsample by factors of 2, 4, 8
        downsampled = downscale_local_mean(canvas, (scale, scale)).astype(np.uint16)
        multiscale.append(downsampled)

    # Write multiscale data to OME-NGFF
    write_multiscale(multiscale, store, axes=["y", "x"])
    print(f"OME-NGFF dataset written for channel {channel_name}")

def main():
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()

    # Paths and parameters
    image_folder = "/path/to/your/images"
    parameter_file = os.path.join(image_folder, "acquisition parameters.json")
    output_folder = "/path/to/output"
    os.makedirs(output_folder, exist_ok=True)

    # Load imaging parameters
    parameters = load_imaging_parameters(parameter_file)

    # Parse image filenames and get unique channels
    image_info, channels = parse_image_filenames(image_folder)

    print(f"Found {len(channels)} channels: {channels}")
    print(f"Total images: {len(image_info)}")

    # Process each channel separately
    for channel in channels:
        try:
            dx, dy, Nx, Ny, pixel_size_xy = get_image_positions(parameters)
            img_height, img_width = cv2.imread(image_info[0]["filepath"], cv2.IMREAD_GRAYSCALE).shape
            canvas_width = int(img_width + (Nx - 1) * img_width * (1 - compute_overlap_percent(dx, dy, img_width, img_height, pixel_size_xy) / 100))
            canvas_height = int(img_height + (Ny - 1) * img_height * (1 - compute_overlap_percent(dx, dy, img_width, img_height, pixel_size_xy) / 100))

            # Initialize OME-NGFF dataset
            store = initialize_ome_ngff(output_folder, channel, canvas_height, canvas_width)

            # Stream images into OME-NGFF
            stream_images_to_ome_ngff(channel, image_info, parameters, store, rotation_angle=90)

        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")

if __name__ == "__main__":
    main()