import os
import cv2
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm
import json

def calculate_image_shift(canvas_region, new_image, x_start, y_start, image_shape, overlap_threshold=0.3):
    """
    Calculate shift between overlapping regions using phase correlation.

    Args:
        canvas_region: Region from the canvas where new image will be placed
        new_image: New image to be registered
        x_start, y_start: Initial coordinates based on stage position
        image_shape: Shape of the new image
        overlap_threshold: Minimum ratio of non-zero pixels required in overlap region

    Returns:
        dx, dy: Calculated shift in x and y directions
        is_valid: Boolean indicating if the shift is valid
    """
    # Check if there's enough overlap to calculate shift
    overlap_mask = canvas_region > 0
    overlap_ratio = np.sum(overlap_mask) / (canvas_region.shape[0] * canvas_region.shape[1])

    if overlap_ratio < overlap_threshold:
        return 0, 0, False

    # Apply phase correlation only on overlapping region
    height, width = new_image.shape
    valid_region = canvas_region[overlap_mask]
    new_image_region = new_image[overlap_mask]

    # Calculate phase correlation
    try:
        shift, error, diffphase = cv2.phaseCorrelate(
            np.float32(valid_region), 
            np.float32(new_image_region)
        )
        dx, dy = int(round(shift[0])), int(round(shift[1]))

        # Limit maximum shift to prevent unrealistic results
        max_shift = min(width, height) // 4
        if abs(dx) > max_shift or abs(dy) > max_shift:
            return 0, 0, False

        return dx, dy, True
    except:
        return 0, 0, False
    
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
    print(f"Pixel size: {pixel_size_xy} µm")
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

    # Sort images by region, y_idx, and x_idx (top to bottom, left to right)
    image_info.sort(key=lambda x: (x["region"], x["y_idx"], x["x_idx"]))
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
    
    accumulated_shifts = {}  # Store accumulated shifts for each region
    reference_channel = selected_channel[0]
    current_shifts = {}  # Store shifts for current position (for all channels to use)
    # Process images by position (same x,y,region) together
    position_groups = {}
    
    for info in image_info:
        position_key = (info["region"], info["x_idx"], info["y_idx"])
        if position_key not in position_groups:
            position_groups[position_key] = []
        position_groups[position_key].append(info)
        
    # Sort position keys by region, y, then x
    sorted_positions = sorted(position_groups.keys(), key=lambda pos: (
        pos[0],  # region
        coordinates[(coordinates["region"] == pos[0]) & 
                   (coordinates["i"] == pos[1]) & 
                   (coordinates["j"] == pos[2])]["y (mm)"].iloc[0],
        coordinates[(coordinates["region"] == pos[0]) & 
                   (coordinates["i"] == pos[1]) & 
                   (coordinates["j"] == pos[2])]["x (mm)"].iloc[0]
    ))
        
    for position in tqdm(sorted_positions):
        region, x_idx, y_idx = position

        # Initialize accumulated shift for new region
        if region not in accumulated_shifts:
            accumulated_shifts[region] = {"dx": 0, "dy": 0}

        # Get coordinates for this position
        coord_row = coordinates[(coordinates["region"] == region) & 
                              (coordinates["i"] == x_idx) & 
                              (coordinates["j"] == y_idx)].iloc[0]

        x_mm, y_mm = coord_row["x (mm)"], coord_row["y (mm)"]
        x_um = (x_mm * 1000) - x_offset
        y_um = (y_mm * 1000) - y_offset
        x_start = int(x_um / pixel_size_xy)
        y_start = int(y_um / pixel_size_xy)

        # Process all channels at this position
        for info in position_groups[position]:
            channel = info["channel_name"]
            if channel not in selected_channel:
                continue

            # Read and preprocess image
            image = cv2.imread(info["filepath"], cv2.IMREAD_ANYDEPTH)
            if image is None:
                print(f"Error: Unable to read {info['filepath']}")
                continue

            # Normalize image (your existing normalization code)
            if image.dtype != np.uint8:
                img_min = np.percentile(image, 1)
                img_max = np.percentile(image, 99)
                image = np.clip(image, img_min, img_max)
                image = ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
            image = rotate_flip_image(image)

            # Calculate shift only for reference channel
            if channel == reference_channel:
                region_height, region_width = image.shape
                canvas_region = datasets[channel]["scale0"][
                    y_start:y_start + region_height,
                    x_start:x_start + region_width
                ]

                dx, dy, is_valid = calculate_image_shift(
                    canvas_region, image, x_start, y_start, image.shape
                )

                if is_valid:
                    accumulated_shifts[region]["dx"] += dx
                    accumulated_shifts[region]["dy"] += dy

            # Apply the accumulated shift from the region to all channels
            adjusted_x = x_start + accumulated_shifts[region]["dx"]
            adjusted_y = y_start + accumulated_shifts[region]["dy"]

            # Validate and clip the placement region
            if adjusted_x < 0 or adjusted_y < 0 or adjusted_x >= canvas_width or adjusted_y >= canvas_height:
                print(f"Warning: Image at {info['filepath']} is out of canvas bounds after registration.")
                continue

            # Calculate end positions and clip image if necessary
            x_end = min(adjusted_x + image.shape[1], canvas_width)
            y_end = min(adjusted_y + image.shape[0], canvas_height)

            if x_end - adjusted_x != image.shape[1] or y_end - adjusted_y != image.shape[0]:
                image = image[:y_end-adjusted_y, :x_end-adjusted_x]

            # Place the image and update pyramid
            datasets[channel]["scale0"][adjusted_y:y_end, adjusted_x:x_end] = image

            # Update pyramid levels
            for level in range(1, pyramid_levels):
                update_pyramid(datasets, channel, level, image, adjusted_x, adjusted_y)

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
    print(f"Max dimension: {max_dimension}, max levels: {max_levels}")
    pyramid_levels = 11

    print(f"Using {pyramid_levels} pyramid levels for {canvas_width}x{canvas_height} canvas")
    # Parse image filenames and get unique channels
    image_info = parse_image_filenames(image_folder)
    channels = list(set(info["channel_name"] for info in image_info))
    print(f"Found {len(image_info)} images with {len(channels)} channels")
    selected_channel = ['Fluorescence_488_nm_Ex']
    print(f"Selected channel: {selected_channel}")

    # If dataset eixts, use it
    if os.path.exists(os.path.join(output_folder, "stitched_images.zarr")):
        print("Dataset exists, skipping image processing.")
        datasets = zarr.open(os.path.join(output_folder, "stitched_images.zarr"), mode="a")
    else:
        # Create OME-NGFF file
        root, datasets = create_ome_ngff(output_folder, canvas_width, canvas_height, selected_channel, pyramid_levels=pyramid_levels)
        print(f"Dataset created: {datasets}")

    # Process images and stitch them
    process_images(image_info, coordinates, datasets, pixel_size_xy, stage_limits, selected_channel=selected_channel, pyramid_levels=pyramid_levels)

if __name__ == "__main__":
    main()