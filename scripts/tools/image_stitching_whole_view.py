import os
import cv2
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm
import json

def load_imaging_parameters(parameter_file):
    """Load imaging parameters from a JSON file."""
    with open(parameter_file, "r") as f:
        parameters = json.load(f)
    return parameters

def rotate_flip_image(image, angle=90, flip=True):
    """Rotate an image by a specified angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    if flip:
      rotated = cv2.flip(rotated, -1) # Flip horizontally and vertically
    return rotated

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

def create_canvas_size(stage_limits, pixel_size_xy):
    """Calculate the canvas size in pixels based on stage limits and pixel size."""
    x_range = (stage_limits["x_positive"] - stage_limits["x_negative"]) * 1000  # Convert mm to µm
    y_range = (stage_limits["y_positive"] - stage_limits["y_negative"]) * 1000  # Convert mm to µm
    canvas_width = int(x_range / pixel_size_xy)
    canvas_height = int(y_range / pixel_size_xy)
    return canvas_width, canvas_height

def parse_image_filenames(image_folder):
    """Parse image filenames to extract FOV information."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".bmp")]
    image_info = []

    for image_file in image_files:
        # Split only for the first 4 parts (region, x, y, z)
        prefix_parts = image_file.split('_', 4)  # Split into max 5 parts
        if len(prefix_parts) >= 5:
            region, x_idx, y_idx, z_idx = prefix_parts[:4]
            # Get the channel name by removing the extension
            channel_name = prefix_parts[4].rsplit('.', 1)[0]

            image_info.append({
                "filepath": os.path.join(image_folder, image_file),
                "region": region,
                "x_idx": int(x_idx),
                "y_idx": int(y_idx),
                "z_idx": int(z_idx),
                "channel_name": channel_name
            })


    return image_info

def create_ome_ngff(output_folder, canvas_width, canvas_height, channels, pyramid_levels=7, chunk_size=(256, 256)):
    """Create an OME-NGFF (zarr) file with a fixed canvas size and pyramid levels."""
    os.makedirs(output_folder, exist_ok=True)
    zarr_path = os.path.join(output_folder, "stitched_images.zarr")
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)

    # Print initial canvas size
    print(f"Initial canvas size: {canvas_width}x{canvas_height}")

    # Create datasets for each channel
    datasets = {}
    for channel in channels:
        print(f"\nCreating dataset for channel: {channel}")
        group = root.create_group(channel)
        datasets[channel] = group

        # Create base resolution (scale0)
        print(f"scale0: {canvas_height}x{canvas_width}")
        datasets[channel].create_dataset(
            "scale0",
            shape=(canvas_height, canvas_width),
            chunks=chunk_size,
            dtype=np.uint8,
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
        )

        # Create pyramid levels
        for level in range(1, pyramid_levels):
            scale_name = f"scale{level}"
            level_shape = (
                max(1, canvas_height // (2 ** level)),
                max(1, canvas_width // (2 ** level))
            )
            level_chunks = (
                min(chunk_size[0], level_shape[0]),
                min(chunk_size[1], level_shape[1])
            )

            print(f"{scale_name}: {level_shape} (chunks: {level_chunks})")

            datasets[channel].create_dataset(
                scale_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=np.uint8,
                compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
            )

    return root, datasets

def update_pyramid(datasets, channel, level, image, x_start, y_start):
    """Update a specific pyramid level with the given image."""
    scale = 2 ** level
    scale_name = f"scale{level}"

    try:
        pyramid_dataset = datasets[channel][scale_name]

        # Calculate the scaled coordinates
        x_scaled = x_start // scale
        y_scaled = y_start // scale

        # Calculate target dimensions
        target_height = max(1, image.shape[0] // scale)
        target_width = max(1, image.shape[1] // scale)

        # Only proceed if the target dimensions are meaningful
        if target_height == 0 or target_width == 0:
            print(f"Skipping level {level} due to zero dimensions")
            return

        # Resize the image
        scaled_image = cv2.resize(
            image, 
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )

        # Calculate the valid region to update
        end_y = min(y_scaled + scaled_image.shape[0], pyramid_dataset.shape[0])
        end_x = min(x_scaled + scaled_image.shape[1], pyramid_dataset.shape[1])

        # Only update if we have a valid region
        if end_y > y_scaled and end_x > x_scaled:
            # Clip the image if necessary
            if end_y - y_scaled != scaled_image.shape[0] or end_x - x_scaled != scaled_image.shape[1]:
                scaled_image = scaled_image[:(end_y - y_scaled), :(end_x - x_scaled)]

            # Update the dataset
            pyramid_dataset[y_scaled:end_y, x_scaled:end_x] = scaled_image
        else:
            print(f"Skipping update for level {level} due to out-of-bounds coordinates")

    except Exception as e:
        print(f"Error updating pyramid level {level}: {e}")
    finally:
        # Clean up
        if 'scaled_image' in locals():
            del scaled_image


def process_images(image_info, coordinates, datasets, pixel_size_xy, stage_limits, pyramid_levels=7, selected_channel=None):
    """
    Process images and place them on the canvas based on physical coordinates.
    """
    if selected_channel is None:
        raise ValueError("selected_channel must be a list of channels to process.")

    x_offset = stage_limits["x_negative"] * 1000  # Convert mm to µm
    y_offset = stage_limits["y_negative"] * 1000  # Convert mm to µm

    # Get canvas dimensions from the dataset
    canvas_width = datasets[selected_channel[0]]["scale0"].shape[1]
    canvas_height = datasets[selected_channel[0]]["scale0"].shape[0]

    for info in tqdm(image_info):
        # Extract FOV information
        filepath = info["filepath"]
        region = info["region"]
        x_idx = info["x_idx"]
        y_idx = info["y_idx"]
        channel = info["channel_name"]

        # Skip images not in the selected channel
        if channel not in selected_channel:
            continue

        # Skip images not in the selected channel
        if channel not in selected_channel:
            continue

        # Get physical coordinates from coordinates.csv
        coord_row = coordinates[(coordinates["region"] == region) & (coordinates["i"] == x_idx) & (coordinates["j"] == info["y_idx"])]
        if coord_row.empty:
            print(f"Warning: No coordinates found for {filepath}")
            continue

        # Extract x, y coordinates
        x_mm, y_mm = coord_row.iloc[0]["x (mm)"], coord_row.iloc[0]["y (mm)"] 
        x_um = (x_mm * 1000) - x_offset
        y_um = (y_mm * 1000) - y_offset

        # Convert physical coordinates to pixel coordinates
        x_start = int(x_um / pixel_size_xy)
        y_start = int(y_um / pixel_size_xy)

        # Read the image
        image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)  # Use ANYDEPTH for 16-bit images
        if image is None:
            print(f"Error: Unable to read {filepath}")
            continue

        # Normalize each image individually
        if image.dtype != np.uint8:
            # Get the actual min and max of the image
            img_min = np.percentile(image, 1)  # 1st percentile to remove outliers
            img_max = np.percentile(image, 99)  # 99th percentile to remove outliers

            # Clip the image to the computed range and normalize to 8-bit
            image = np.clip(image, img_min, img_max)
            image = ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
        image = rotate_flip_image(image)
        # Validate and clip the placement region
        if x_start < 0 or y_start < 0 or x_start >= canvas_width or y_start >= canvas_height:
            print(f"Warning: Image at {filepath} is out of canvas bounds and will be skipped.")
            continue

        # Calculate end positions
        x_end = min(x_start + image.shape[1], canvas_width)
        y_end = min(y_start + image.shape[0], canvas_height)

        # Clip the image if necessary
        if x_end - x_start != image.shape[1] or y_end - y_start != image.shape[0]:
            image = image[:y_end-y_start, :x_end-x_start]

        # Place the image on the base resolution canvas
        datasets[channel]["scale0"][y_start:y_end, x_start:x_end] = image

        # Update pyramid levels
        for level in range(1, pyramid_levels):
            update_pyramid(datasets, channel, level, image, x_start, y_start)

        # Free memory
        del image

def main():
    # Paths and parameters
    data_folder = "/media/reef/harddisk/20241112-hpa_2024-11-12_15-49-12.554140"
    image_folder = os.path.join(data_folder, "0")
    parameter_file = os.path.join(data_folder, "acquisition parameters.json")
    coordinates_file = os.path.join(image_folder, "coordinates-processed.csv")
    output_folder = "/media/reef/harddisk/stitched_output_whole_view"
    os.makedirs(output_folder, exist_ok=True)

    # Load imaging parameters and coordinates
    parameters = load_imaging_parameters(parameter_file)
    coordinates = pd.read_csv(coordinates_file)

    # Get pixel size and canvas size
    pixel_size_xy = get_pixel_size(parameters)
    stage_limits = {
        "x_positive": 120,
        "x_negative": 0,
        "y_positive": 86,
        "y_negative": 0,
        "z_positive": 6
    }
    canvas_width, canvas_height = create_canvas_size(stage_limits, pixel_size_xy)

    # Calculate appropriate number of pyramid levels
    max_dimension = max(canvas_width, canvas_height)
    max_levels = int(np.floor(np.log2(max_dimension)))
    pyramid_levels = min(7, max_levels)  # Cap at 7 or fewer levels

    print(f"Using {pyramid_levels} pyramid levels for {canvas_width}x{canvas_height} canvas")
    # Parse image filenames and get unique channels
    image_info = parse_image_filenames(image_folder)
    channels = list(set(info["channel_name"] for info in image_info))
    print(f"Found {len(image_info)} images with {len(channels)} channels")
    selected_channel = ['BF_LED_matrix_full']
    print(f"Selected channel: {selected_channel}")

    # If dataset eixts, use it
    if os.path.exists(os.path.join(output_folder, "stitched_images.zarr")):
        print("Dataset exists, skipping image processing.")
        datasets = zarr.open(os.path.join(output_folder, "stitched_images.zarr"), mode="a")
    else:
        # Create OME-NGFF file
        root, datasets = create_ome_ngff(output_folder, canvas_width, canvas_height, selected_channel)
        print(f"Dataset created: {datasets}")

    # Process images and stitch them
    process_images(image_info, coordinates, datasets, pixel_size_xy, stage_limits, selected_channel=selected_channel)

if __name__ == "__main__":
    main()