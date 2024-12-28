import zarr
import numpy as np
import cv2

def create_preview_image(zarr_path, output_path, max_size=4000):
    """
    Create a preview image from scale6 of an OME-NGFF file, maintaining aspect ratio.
    
    Args:
        zarr_path (str): Path to the zarr directory
        output_path (str): Path where the preview image will be saved
        max_size (int): Maximum size for either dimension
    """
    # Open the zarr dataset
    store = zarr.open(zarr_path, mode='r')
    
    # Get scale6 data from BF_LED_matrix_full channel
    channel = 'BF_LED_matrix_full'
    scale_data = store[channel]['scale10'][:]
    
    # Get original dimensions
    original_height, original_width = scale_data.shape
    print(f"Scale6 image size: {original_width}x{original_height}")
    
    # Calculate scaling factor to maintain aspect ratio
    width_scale = max_size / original_width
    height_scale = max_size / original_height
    scale_factor = min(width_scale, height_scale)
    
    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    print(f"Resizing to: {new_width}x{new_height}")
    
    # Resize the image
    preview_image = cv2.resize(
        scale_data, 
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Save the preview image
    cv2.imwrite(output_path, preview_image)
    print(f"Preview image saved to: {output_path}")
    
    # Return the preview image dimensions
    return preview_image.shape

# Example usage
zarr_path = "/media/reef/harddisk/stitched_output_whole_view/stitched_images.zarr"
output_path = "/media/reef/harddisk/stitched_output_whole_view/preview_scale6.png"

preview_shape = create_preview_image(zarr_path, output_path)
print(f"Preview image created with shape: {preview_shape}")

# Created/Modified files during execution:
print("Created/Modified files:")
print("- preview_scale6.png")