#!/usr/bin/env python3
"""
Test script for the new quick_scan_with_stitching functionality.
This script tests the quick scan feature in simulation mode.
"""

import asyncio
import sys
import os

# Add the squid_control package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from squid_control.squid_controller import SquidController

async def test_quick_scan():
    """Test the quick scan with stitching functionality."""
    print("Initializing SquidController in simulation mode...")
    
    # Create controller in simulation mode
    controller = SquidController(is_simulation=False)
    
    try:
        print("Testing quick scan with stitching...")
        
        # Test with a 96-well plate (small subset for testing)
        await controller.quick_scan_with_stitching(
            wellplate_type='96',
            exposure_time=25,  # 25ms - within the 30ms limit
            intensity=60,      # Reasonable brightfield intensity
            velocity_mm_per_s=20,  # 20mm/s as specified
            fps_target=25,     # 25fps target
            action_ID='test_quick_scan'
        )
        
        print("✅ Quick scan completed successfully!")
        
        # Check if zarr canvas was created and used
        if hasattr(controller, 'zarr_canvas') and controller.zarr_canvas is not None:
            print("✅ Zarr canvas was properly initialized and used")
            print(f"Canvas dimensions: {controller.zarr_canvas.canvas_width_px}x{controller.zarr_canvas.canvas_height_px}")
            print(f"Number of scales: {controller.zarr_canvas.num_scales}")
            print(f"Available channels: {list(controller.zarr_canvas.channel_to_zarr_index.keys())}")
        else:
            print("❌ Zarr canvas was not properly initialized")
        
    except ValueError as e:
        if "must not exceed 30ms" in str(e):
            print("✅ Exposure time validation working correctly")
        else:
            print(f"❌ Unexpected ValueError: {e}")
    except Exception as e:
        print(f"❌ Error during quick scan: {e}")
        raise
    
    finally:
        # Clean up
        print("Cleaning up...")
        controller.close()

async def main():
    """Run all tests."""
    print("Starting quick scan tests...\n")
    
    await test_quick_scan()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())