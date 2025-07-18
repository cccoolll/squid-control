#!/usr/bin/env python3
"""
OME-Zarr Chunk Visualizer

This script helps you visualize individual chunks from OME-Zarr datasets as PNG images.
It's useful for debugging and understanding the structure of your microscopy data.

Usage:
    python visualize_zarr_chunks.py <zarr_directory> [output_directory]

Example:
    python visualize_zarr_chunks.py /media/zarr_data/test/well_A2_96.zarr/3 ./chunk_images
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Optional, Tuple


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image array to 0-255 range for display.
    
    Args:
        image: Input image array (can be any data type)
        
    Returns:
        Normalized image array as uint8 (0-255 range)
    """
    # Handle different data types
    if image.dtype == np.uint8:
        return image
    elif image.dtype == np.uint16:
        # Scale 16-bit to 8-bit
        return (image / 256).astype(np.uint8)
    elif image.dtype == np.float32 or image.dtype == np.float64:
        # Normalize float data to 0-255
        if image.max() > image.min():
            normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        return normalized
    else:
        # For other types, try to convert to uint8
        return image.astype(np.uint8)


def read_chunk_file(file_path: Path) -> Optional[np.ndarray]:
    """
    Read a chunk file and return it as a numpy array.
    
    Args:
        file_path: Path to the chunk file
        
    Returns:
        Numpy array containing the chunk data, or None if reading fails
    """
    try:
        print(f"  📁 Reading chunk file: {file_path.name}")
        
        # Read the raw bytes from the file
        with open(file_path, 'rb') as f:
            data = f.read()
        
        print(f"  📊 Raw data size: {len(data)} bytes")
        
        # Try to interpret as different data types
        # Start with uint8 (since zarr arrays are configured as uint8)
        try:
            print(f"  🔍 Attempting uint8 interpretation...")
            chunk = np.frombuffer(data, dtype=np.uint8)
            print(f"  📈 uint8 array length: {len(chunk)} elements")
            
            # OME-Zarr chunks are 5D: (T, C, Z, Y, X)
            # For our system: chunks=(1, 1, 1, 256, 256)
            # So total elements should be 1*1*1*256*256 = 65536
            total_elements = len(chunk)
            
            if total_elements == 65536:  # 256*256 = 65536 (uint8)
                print(f"  ✅ Detected 5D OME-Zarr chunk (uint8): 1*1*1*256*256 = 65536 elements")
                # This is a 5D chunk with shape (1, 1, 1, 256, 256)
                # Extract the 2D slice: [0, 0, 0, :, :]
                chunk_5d = chunk.reshape(1, 1, 1, 256, 256)
                print(f"  🔄 Reshaped to 5D: {chunk_5d.shape}")
                result = chunk_5d[0, 0, 0, :, :]  # Extract 2D slice
                print(f"  📋 Extracted 2D slice: {result.shape}")
                return result
            elif total_elements == 32768:  # 256*256/2 = 32768 (uint16)
                print(f"  ⚠️  Detected 5D OME-Zarr chunk (uint16): 1*1*1*256*256 = 32768 uint16 elements")
                # This is a 5D chunk with shape (1, 1, 1, 256, 256) stored as uint16
                chunk_5d = chunk.reshape(1, 1, 1, 256, 256)
                print(f"  🔄 Reshaped to 5D: {chunk_5d.shape}")
                result = chunk_5d[0, 0, 0, :, :]  # Extract 2D slice
                print(f"  📋 Extracted 2D slice: {result.shape}")
                return result
            else:
                print(f"  ❓ Not a standard 5D chunk, trying 2D interpretation...")
                # Try to reshape as 2D image (square root for dimensions)
                size = int(np.sqrt(len(chunk)))
                print(f"  📏 Calculated size: {size}, len(chunk): {len(chunk)}")
                if size * size == len(chunk):
                    print(f"  ✅ Perfect square: {size}x{size} = {len(chunk)}")
                    return chunk.reshape(size, size)
                else:
                    print(f"  ⚠️  Not a perfect square, returning as 1D array")
                    # If not a perfect square, return as 1D array
                    return chunk
                
        except Exception as e:
            print(f"  ❌ uint8 interpretation failed: {e}")
            # Fallback to uint16 (for older data)
            try:
                print(f"  🔍 Attempting uint16 interpretation...")
                # Calculate the number of uint16 values
                num_values = len(data) // 2
                print(f"  📊 uint16 array length: {num_values} elements")
                chunk = np.frombuffer(data, dtype=np.uint16, count=num_values)
                
                # Check if this is a 5D chunk
                if num_values == 65536:  # 256*256 = 65536 (uint16)
                    print(f"  ✅ Detected 5D OME-Zarr chunk (uint16): 1*1*1*256*256 = 65536 elements")
                    # This is a 5D chunk with shape (1, 1, 1, 256, 256)
                    chunk_5d = chunk.reshape(1, 1, 1, 256, 256)
                    print(f"  🔄 Reshaped to 5D: {chunk_5d.shape}")
                    result = chunk_5d[0, 0, 0, :, :]  # Extract 2D slice
                    print(f"  📋 Extracted 2D slice: {result.shape}")
                    return result
                else:
                    print(f"  ❓ Not a standard 5D chunk, trying 2D interpretation...")
                    # Try to reshape as 2D image (square root for dimensions)
                    size = int(np.sqrt(len(chunk)))
                    print(f"  📏 Calculated size: {size}, len(chunk): {len(chunk)}")
                    if size * size == len(chunk):
                        print(f"  ✅ Perfect square: {size}x{size} = {len(chunk)}")
                        return chunk.reshape(size, size)
                    else:
                        print(f"  ⚠️  Not a perfect square, returning as 1D array")
                        # If not a perfect square, return as 1D array
                        return chunk
                    
            except Exception as e:
                print(f"  ❌ uint16 interpretation failed: {e}")
                print(f"  ❌ Could not interpret {file_path} as image data")
                return None
                
    except Exception as e:
        print(f"  ❌ Error reading {file_path}: {e}")
        return None


def save_chunk_as_png(chunk: np.ndarray, output_path: Path, chunk_name: str):
    """
    Save a chunk as a PNG image.
    
    Args:
        chunk: Numpy array containing the chunk data
        output_path: Directory to save the PNG file
        chunk_name: Name of the chunk (used for filename)
    """
    # Normalize the image for display
    normalized_chunk = normalize_image(chunk)
    
    # Create the output filename using the full chunk name to avoid overwrites
    png_filename = f"{chunk_name}.png"
    png_path = output_path / png_filename
    
    # Handle different array shapes
    if len(chunk.shape) == 1:
        # 1D array - try to reshape as square or show as line plot
        size = len(chunk)
        sqrt_size = int(np.sqrt(size))
        
        if sqrt_size * sqrt_size == size:
            # Perfect square - reshape as 2D image
            reshaped_chunk = chunk.reshape(sqrt_size, sqrt_size)
            normalized_reshaped = normalize_image(reshaped_chunk)
            
            # Save as 2D image
            plt.figure(figsize=(8, 6))
            plt.imshow(normalized_reshaped, cmap='gray')
            plt.title(f"Chunk: {chunk_name} (reshaped from 1D)")
            plt.colorbar(label='Pixel Value')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also save raw PNG
            try:
                from PIL import Image
                img = Image.fromarray(normalized_reshaped)
                raw_png_path = output_path / f"{chunk_name}_raw.png"
                img.save(raw_png_path)
                print(f"Saved raw: {raw_png_path}")
            except ImportError:
                print("PIL not available, skipping raw PNG export")
                
        else:
            # Not a perfect square - show as line plot
            plt.figure(figsize=(12, 4))
            plt.plot(chunk)
            plt.title(f"Chunk: {chunk_name} (1D data)")
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save histogram
            hist_path = output_path / f"{chunk_name}_histogram.png"
            plt.figure(figsize=(8, 4))
            plt.hist(chunk, bins=50, alpha=0.7)
            plt.title(f"Chunk: {chunk_name} (value distribution)")
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved histogram: {hist_path}")
            
    elif len(chunk.shape) == 2:
        # 2D array - display as image
        plt.figure(figsize=(8, 6))
        plt.imshow(normalized_chunk, cmap='gray')
        plt.title(f"Chunk: {chunk_name}")
        plt.colorbar(label='Pixel Value')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save the raw data as a simple PNG
        try:
            from PIL import Image
            img = Image.fromarray(normalized_chunk)
            raw_png_path = output_path / f"{chunk_name}_raw.png"
            img.save(raw_png_path)
            print(f"Saved raw: {raw_png_path}")
        except ImportError:
            print("PIL not available, skipping raw PNG export")
            
    else:
        # Higher dimensional array - show info and save first slice
        print(f"  Warning: {len(chunk.shape)}D array, showing first slice")
        first_slice = chunk[0] if len(chunk.shape) > 2 else chunk
        
        if len(first_slice.shape) == 2:
            normalized_slice = normalize_image(first_slice)
            plt.figure(figsize=(8, 6))
            plt.imshow(normalized_slice, cmap='gray')
            plt.title(f"Chunk: {chunk_name} (first slice)")
            plt.colorbar(label='Pixel Value')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Saved: {png_path}")


def visualize_zarr_chunks(zarr_dir: str, output_dir: Optional[str] = None):
    """
    Main function to visualize all chunks in a Zarr directory.
    
    Args:
        zarr_dir: Path to the Zarr directory containing chunks
        output_dir: Directory to save PNG images (optional)
    """
    zarr_path = Path(zarr_dir)
    
    print(f"🔍 Starting OME-Zarr chunk visualization...")
    print(f"📂 Zarr directory: {zarr_dir}")
    
    # Check if the directory exists
    if not zarr_path.exists():
        print(f"❌ Error: Directory {zarr_dir} does not exist")
        return
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_path}")
    else:
        output_path = Path("chunk_images")
        output_path.mkdir(exist_ok=True)
        print(f"📁 Using default output directory: {output_path}")
    
    # Find all chunk files
    chunk_files = []
    for file_path in zarr_path.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            chunk_files.append(file_path)
    
    if not chunk_files:
        print(f"❌ No chunk files found in {zarr_dir}")
        return
    
    print(f"📊 Found {len(chunk_files)} chunk files")
    print(f"📋 Chunk files: {[f.name for f in chunk_files]}")
    
    # Process each chunk file
    for i, chunk_file in enumerate(chunk_files):
        print(f"\n🔄 Processing chunk {i+1}/{len(chunk_files)}: {chunk_file.name}")
        
        # Read the chunk
        chunk_data = read_chunk_file(chunk_file)
        
        if chunk_data is not None:
            print(f"  ✅ Successfully read chunk data")
            print(f"  📐 Shape: {chunk_data.shape}")
            print(f"  🏷️  Data type: {chunk_data.dtype}")
            print(f"  📊 Min value: {chunk_data.min()}")
            print(f"  📊 Max value: {chunk_data.max()}")
            print(f"  📊 Mean value: {chunk_data.mean():.2f}")
            print(f"  📊 Std deviation: {chunk_data.std():.2f}")
            
            # Save as PNG using the full chunk name to avoid overwrites
            print(f"  💾 Saving chunk as PNG...")
            save_chunk_as_png(chunk_data, output_path, chunk_file.name)
        else:
            print(f"  ❌ Failed to read chunk: {chunk_file.name}")
    
    print(f"\n🎉 Visualization complete! Check {output_path} for PNG images.")
    print(f"📊 Summary: Processed {len(chunk_files)} chunks")


def main():
    """Main function to handle command line arguments and run the visualizer."""
    parser = argparse.ArgumentParser(
        description="Visualize OME-Zarr chunks as PNG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_zarr_chunks.py /media/zarr_data/test/well_A2_96.zarr/3
  python visualize_zarr_chunks.py /path/to/zarr/dir ./output_images
        """
    )
    
    parser.add_argument(
        "zarr_dir",
        help="Path to the Zarr directory containing chunks"
    )
    
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Output directory for PNG images (optional, defaults to ./chunk_images)"
    )
    
    args = parser.parse_args()
    
    # Run the visualizer
    visualize_zarr_chunks(args.zarr_dir, args.output_dir)


if __name__ == "__main__":
    main() 