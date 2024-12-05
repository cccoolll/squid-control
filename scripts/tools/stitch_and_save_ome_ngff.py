import os
import numpy as np
import zarr
from skimage.transform import downscale_local_mean
from image_stitching import load_imaging_parameters, parse_image_filenames, process_channel

def save_to_ome_ngff(output_folder, channel_name, stitched_image, max_scale=64):
    """
    Save a stitched image to OME-NGFF format with multi-resolution layers.

    Parameters:
        output_folder (str): Path to the output folder.
        channel_name (str): Name of the channel being processed.
        stitched_image (np.ndarray): The stitched image (2D array).
        max_scale (int): Maximum downscaling factor (e.g., 64x).
    """
    # Create a Zarr group for the channel
    zarr_path = os.path.join(output_folder, f"{channel_name}.zarr")
    root = zarr.open_group(zarr_path, mode="w")

    # Create multi-resolution layers
    current_image = stitched_image
    scale = 1
    pyramid = []

    while current_image.shape[0] > 1 and current_image.shape[1] > 1 and scale <= max_scale:
        pyramid.append(current_image)
        # Downscale by a factor of 2 in both dimensions
        current_image = downscale_local_mean(current_image, (2, 2)).astype(stitched_image.dtype)
        scale *= 2

    # Save each resolution level to the Zarr group
    for i, level in enumerate(pyramid):
        root.create_dataset(
            f"scale{i}",
            data=level,
            chunks=(min(1024, level.shape[0]), min(1024, level.shape[1])),
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2),
        )
        print(f"Saved scale{i} with shape {level.shape}")

    # Add OME-NGFF metadata
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "datasets": [{"path": f"scale{i}"} for i in range(len(pyramid))],
            "type": "image",
        }
    ]
    print(f"Saved OME-NGFF file: {zarr_path}")


def main():
    # Paths and parameters
    image_folder = "/media/reef/harddisk/test_stitching/stitching_dataset"
    parameter_file = os.path.join(image_folder, "acquisition parameters.json")
    output_folder = "/media/reef/harddisk/test_stitching/stitched_output"
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
            # Stitch the images for the current channel
            print(f"Processing channel: {channel}")
            stitched_image_path = os.path.join(output_folder, f"stitched_{channel}.tiff")
            process_channel(channel, image_info, parameters, output_folder, rotation_angle=90)

            # Load the stitched image
            stitched_image = tifffile.imread(stitched_image_path)

            # Save the stitched image in OME-NGFF format with multi-resolution layers
            save_to_ome_ngff(output_folder, channel, stitched_image, max_scale=64)

        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")


if __name__ == "__main__":
    main()