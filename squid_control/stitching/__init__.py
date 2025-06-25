"""
Image stitching module for squid microscope control.

This module provides live stitching capabilities for creating 
large field-of-view images from multiple microscope acquisitions.
"""

from .zarr_canvas import ZarrCanvas

__all__ = ['ZarrCanvas'] 