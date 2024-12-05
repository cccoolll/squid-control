import os  
import numpy as np  
import zarr  
from skimage.transform import downscale_local_mean  
import tifffile  
  
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
    stitched_folder = "/media/reef/harddisk/test_stitching/stitched_output"  # Folder containing stitched .tiff files  
    output_folder = "/media/reef/harddisk/test_stitching/ome_ngff_output"  # Folder to save OME-NGFF files  
    os.makedirs(output_folder, exist_ok=True)  
  
    # Find all stitched .tiff files in the folder  
    stitched_files = [f for f in os.listdir(stitched_folder) if f.endswith(".tiff")]  
  
    if not stitched_files:  
        print("No stitched .tiff files found in the folder.")  
        return  
  
    print(f"Found {len(stitched_files)} stitched files: {stitched_files}")  
  
    # Process each stitched file  
    for stitched_file in stitched_files:  
        try:  
            # Load the stitched image  
            channel_name = os.path.splitext(stitched_file)[0].replace("stitched_", "")  
            stitched_image_path = os.path.join(stitched_folder, stitched_file)  
            print(f"Loading stitched image: {stitched_image_path}")  
            stitched_image = tifffile.imread(stitched_image_path)  
  
            # Save the stitched image in OME-NGFF format with multi-resolution layers  
            save_to_ome_ngff(output_folder, channel_name, stitched_image, max_scale=64)  
  
        except Exception as e:  
            print(f"Error processing file {stitched_file}: {str(e)}")  
  
  
if __name__ == "__main__":  
    main()  