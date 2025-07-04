#!/usr/bin/env python3
"""
Demosaic module - Python implementation based on darktable's demosaic.c

This module implements demosaicing algorithms to reconstruct full RGB 
pixels from sensor color filter array (CFA) readings.

For MVP implementation, we focus on:
- Bilinear interpolation for Bayer patterns
- Simple and efficient processing
- Compatible with rawprepare output format
"""

import numpy as np
from typing import Optional
from enum import IntEnum
from scipy.ndimage import convolve

try:
    from ..core.datatypes import ImageBuffer, ROI, DataType, ColorSpace, BayerPattern
except ImportError:
    from core.datatypes import ImageBuffer, ROI, DataType, ColorSpace, BayerPattern


class DemosaicMethod(IntEnum):
    """Demosaicing methods"""
    BILINEAR = 0    # Simple bilinear interpolation
    PPG = 1         # Pattern Pixel Grouping
    VNG4 = 2        # Variable Number of Gradients (future)


class Demosaic:
    """
    Demosaic module for converting RAW CFA data to RGB
    
    MVP version focusing on bilinear interpolation
    """
    
    def __init__(self):
        self.name = "demosaic"
        self.method = DemosaicMethod.BILINEAR
    
    def process(self, 
                input_buffer: ImageBuffer, 
                roi_in: ROI, 
                roi_out: ROI) -> ImageBuffer:
        """
        Process RAW CFA data to RGB
        
        Args:
            input_buffer: Input RAW buffer from rawprepare
            roi_in: Input region of interest
            roi_out: Output region of interest
            
        Returns:
            RGB ImageBuffer
        """
        # Validate input
        if input_buffer.channels != 1:
            raise ValueError("Demosaic input must be single channel RAW data")
        if input_buffer.datatype != DataType.FLOAT32:
            raise ValueError("Demosaic input must be float32")
        if input_buffer.colorspace != ColorSpace.RAW:
            raise ValueError("Demosaic input must be RAW colorspace")
        
        # Extract RAW data
        raw_data = input_buffer.data
        
        # Determine Bayer pattern
        bayer_pattern = input_buffer.bayer_pattern or BayerPattern.RGGB
        
        # Perform demosaicing
        if self.method == DemosaicMethod.BILINEAR:
            rgb_data = self._demosaic_bilinear(raw_data, bayer_pattern)
        elif self.method == DemosaicMethod.PPG:
            rgb_data = self._demosaic_ppg(raw_data, bayer_pattern)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
        
        # Create output buffer
        output_buffer = ImageBuffer(
            data=rgb_data,
            width=roi_out.width,
            height=roi_out.height,
            channels=3,
            datatype=DataType.FLOAT32,
            colorspace=ColorSpace.RGB,
            filters=None,  # No longer RAW
            bayer_pattern=None,
            xtrans=None,
            black_level=0.0,
            white_point=1.0
        )
        
        return output_buffer
    
    def _demosaic_bilinear(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """
        Efficient bilinear demosaicing using convolution
        
        Args:
            raw_data: (H, W) float32 array, range [0, 1]
            bayer_pattern: Bayer pattern type
        
        Returns:
            (H, W, 3) float32 RGB array
        """
        height, width = raw_data.shape
        rgb_data = np.zeros((height, width, 3), dtype=np.float32)
        
        # Define kernels for bilinear interpolation
        kernel_cross = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.float32) / 4.0
        
        kernel_diagonal = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ], dtype=np.float32) / 4.0
        
        # Create channel masks based on Bayer pattern
        r_mask, g_mask, b_mask = self._create_bayer_masks(height, width, bayer_pattern)
        
        # Extract channels
        r_channel = raw_data * r_mask
        g_channel = raw_data * g_mask
        b_channel = raw_data * b_mask
        
        # Interpolate Red channel
        rgb_data[:, :, 0] = r_channel + convolve(r_channel, kernel_cross, mode='reflect') * (1 - r_mask)
        
        # Interpolate Green channel (use cross pattern for missing pixels)
        rgb_data[:, :, 1] = g_channel + convolve(g_channel, kernel_cross, mode='reflect') * (1 - g_mask)
        
        # Interpolate Blue channel
        rgb_data[:, :, 2] = b_channel + convolve(b_channel, kernel_cross, mode='reflect') * (1 - b_mask)
        
        # Handle diagonal positions for R and B channels
        # In Bayer pattern, R and B are diagonally opposite
        if bayer_pattern in [BayerPattern.RGGB, BayerPattern.BGGR]:
            # For positions where both R and B are missing, use diagonal interpolation
            rb_missing = (1 - r_mask) * (1 - b_mask)
            rgb_data[:, :, 0] += convolve(r_channel, kernel_diagonal, mode='reflect') * rb_missing
            rgb_data[:, :, 2] += convolve(b_channel, kernel_diagonal, mode='reflect') * rb_missing
        
        return np.clip(rgb_data, 0.0, 1.0)
    
    def _demosaic_ppg(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """
        Pattern Pixel Grouping demosaicing algorithm (Optimized)
        
        PPG algorithm steps:
        1. Interpolate green channel using adaptive methods
        2. Classify pixels into different pattern groups
        3. Use appropriate interpolation for each group
        4. Interpolate red and blue channels with green guidance
        
        Args:
            raw_data: (H, W) float32 array, range [0, 1]
            bayer_pattern: Bayer pattern type
        
        Returns:
            (H, W, 3) float32 RGB array
        """
        height, width = raw_data.shape
        rgb_data = np.zeros((height, width, 3), dtype=np.float32)
        
        # Create channel masks based on Bayer pattern
        r_mask, g_mask, b_mask = self._create_bayer_masks(height, width, bayer_pattern)
        
        # Extract channels
        r_channel = raw_data * r_mask
        g_channel = raw_data * g_mask
        b_channel = raw_data * b_mask
        
        # Step 1: Interpolate green channel using vectorized adaptive method
        green_interpolated = self._interpolate_green_ppg_vectorized(raw_data, g_mask)
        rgb_data[:, :, 1] = green_interpolated
        
        # Step 2: Classify pixels into pattern groups (simplified)
        pattern_groups = self._classify_pixel_patterns_fast(raw_data)
        
        # Step 3: Interpolate red and blue channels with pattern-aware methods
        red_interpolated = self._interpolate_red_blue_ppg_vectorized(raw_data, r_mask, green_interpolated, pattern_groups)
        blue_interpolated = self._interpolate_red_blue_ppg_vectorized(raw_data, b_mask, green_interpolated, pattern_groups)
        
        rgb_data[:, :, 0] = red_interpolated
        rgb_data[:, :, 2] = blue_interpolated
        
        return np.clip(rgb_data, 0.0, 1.0)
    
    def _interpolate_green_ppg_vectorized(self, raw_data: np.ndarray, g_mask: np.ndarray) -> np.ndarray:
        """
        Vectorized green channel interpolation using adaptive directional interpolation
        """
        height, width = raw_data.shape
        green_result = (raw_data * g_mask).copy()
        
        # Create padded version for easier neighbor access
        padded_raw = np.pad(raw_data, ((2, 2), (2, 2)), mode='reflect')
        
        # Calculate gradients for all pixels at once
        h_grad = (np.abs(padded_raw[2:-2, 1:-3] - padded_raw[2:-2, 3:-1]) + 
                  np.abs(padded_raw[2:-2, 0:-4] - padded_raw[2:-2, 4:]))
        v_grad = (np.abs(padded_raw[1:-3, 2:-2] - padded_raw[3:-1, 2:-2]) + 
                  np.abs(padded_raw[0:-4, 2:-2] - padded_raw[4:, 2:-2]))
        
        # Find missing green pixels
        missing_mask = (g_mask == 0)
        
        # Horizontal interpolation mask
        h_mask = missing_mask & (h_grad < v_grad * 0.7)
        # Vertical interpolation mask  
        v_mask = missing_mask & (v_grad < h_grad * 0.7)
        # Cross interpolation mask
        cross_mask = missing_mask & ~h_mask & ~v_mask
        
        # Vectorized interpolation
        # Horizontal
        green_result[h_mask] = (raw_data[h_mask] * 0 + 
                               (np.roll(raw_data, 1, axis=1) + np.roll(raw_data, -1, axis=1))[h_mask] / 2.0)
        
        # Vertical
        green_result[v_mask] = (raw_data[v_mask] * 0 + 
                               (np.roll(raw_data, 1, axis=0) + np.roll(raw_data, -1, axis=0))[v_mask] / 2.0)
        
        # Cross pattern
        cross_val = ((np.roll(raw_data, 1, axis=1) + np.roll(raw_data, -1, axis=1) + 
                     np.roll(raw_data, 1, axis=0) + np.roll(raw_data, -1, axis=0)) / 4.0)
        green_result[cross_mask] = cross_val[cross_mask]
        
        # Handle edges with simple bilinear interpolation
        self._fill_edge_pixels_fast(green_result, g_mask, raw_data)
        
        return green_result
    
    def _classify_pixel_patterns_fast(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Fast pixel pattern classification using vectorized operations
        """
        height, width = raw_data.shape
        
        # Calculate local variance using convolution
        from scipy.ndimage import uniform_filter
        
        # Local mean (5x5 neighborhood)
        local_mean = uniform_filter(raw_data, size=5, mode='reflect')
        
        # Local variance (approximation)
        local_var = uniform_filter(raw_data**2, size=5, mode='reflect') - local_mean**2
        
        # Gradient magnitude using Sobel-like operators
        grad_x = np.abs(np.roll(raw_data, 1, axis=1) - np.roll(raw_data, -1, axis=1))
        grad_y = np.abs(np.roll(raw_data, 1, axis=0) - np.roll(raw_data, -1, axis=0))
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Classify based on thresholds
        pattern_groups = np.zeros((height, width), dtype=np.int32)
        
        # Smooth areas
        smooth_mask = (local_var < 0.001) & (grad_mag < 0.05)
        pattern_groups[smooth_mask] = 0
        
        # Edge areas
        edge_mask = (grad_mag > 0.1)
        pattern_groups[edge_mask] = 1
        
        # Texture areas (everything else)
        texture_mask = ~smooth_mask & ~edge_mask
        pattern_groups[texture_mask] = 2
        
        return pattern_groups
    
    def _interpolate_red_blue_ppg_vectorized(self, raw_data: np.ndarray, color_mask: np.ndarray, 
                                           green_channel: np.ndarray, pattern_groups: np.ndarray) -> np.ndarray:
        """
        Vectorized red/blue channel interpolation using pattern-aware methods
        """
        height, width = raw_data.shape
        color_result = (raw_data * color_mask).copy()
        
        # Find missing color pixels
        missing_mask = (color_mask == 0)
        
        # For simplicity, use a fast approximation instead of the complex per-pixel logic
        # This maintains the PPG concept while being much faster
        
        # Smooth areas: use color difference method
        smooth_mask = missing_mask & (pattern_groups == 0)
        if np.any(smooth_mask):
            # Calculate color difference from neighbors
            color_diff = self._calculate_color_diff_vectorized(raw_data, color_mask, green_channel)
            color_result[smooth_mask] = (green_channel + color_diff)[smooth_mask]
        
        # Edge areas: use directional interpolation
        edge_mask = missing_mask & (pattern_groups == 1)
        if np.any(edge_mask):
            # Simple directional interpolation
            h_interp = (np.roll(raw_data, 1, axis=1) + np.roll(raw_data, -1, axis=1)) / 2.0
            v_interp = (np.roll(raw_data, 1, axis=0) + np.roll(raw_data, -1, axis=0)) / 2.0
            
            # Calculate gradients to choose direction
            h_grad = np.abs(np.roll(raw_data, 1, axis=1) - np.roll(raw_data, -1, axis=1))
            v_grad = np.abs(np.roll(raw_data, 1, axis=0) - np.roll(raw_data, -1, axis=0))
            
            # Choose direction with smaller gradient
            use_h = h_grad < v_grad
            color_result[edge_mask & use_h] = h_interp[edge_mask & use_h]
            color_result[edge_mask & ~use_h] = v_interp[edge_mask & ~use_h]
        
        # Texture areas: use weighted interpolation
        texture_mask = missing_mask & (pattern_groups == 2)
        if np.any(texture_mask):
            # Simple weighted average of neighbors
            neighbors = ((np.roll(raw_data, 1, axis=1) + np.roll(raw_data, -1, axis=1) +
                         np.roll(raw_data, 1, axis=0) + np.roll(raw_data, -1, axis=0)) / 4.0)
            color_result[texture_mask] = neighbors[texture_mask]
        
        # Handle edges
        self._fill_edge_pixels_fast(color_result, color_mask, raw_data)
        
        return color_result
    
    def _calculate_color_diff_vectorized(self, raw_data: np.ndarray, color_mask: np.ndarray, 
                                       green_channel: np.ndarray) -> np.ndarray:
        """Calculate color difference using vectorized operations"""
        # Create color difference map
        color_diff = np.zeros_like(raw_data)
        
        # Where we have known color values, calculate color - green
        known_mask = (color_mask == 1)
        color_diff[known_mask] = raw_data[known_mask] - green_channel[known_mask]
        
        # Interpolate color difference using simple convolution
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) / 4.0
        from scipy.ndimage import convolve
        
        # Smooth the color difference map
        color_diff_smooth = convolve(color_diff, kernel, mode='reflect')
        
        return color_diff_smooth
    
    def _fill_edge_pixels_fast(self, result: np.ndarray, mask: np.ndarray, raw_data: np.ndarray):
        """Fast edge pixel filling using vectorized operations"""
        height, width = result.shape
        
        # Find unfilled pixels
        unfilled_mask = (mask == 0) & (result == 0)
        
        if np.any(unfilled_mask):
            # Use simple neighbor average for edge pixels
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32) / 8.0
            from scipy.ndimage import convolve
            
            # Convolve with known pixels
            known_data = np.where(mask == 1, raw_data, 0)
            neighbor_avg = convolve(known_data, kernel, mode='reflect')
            
            # Fill unfilled pixels
            result[unfilled_mask] = neighbor_avg[unfilled_mask]
    
    def _create_bayer_masks(self, height: int, width: int, bayer_pattern: BayerPattern) -> tuple:
        """
        Create binary masks for R, G, B channels based on Bayer pattern
        
        Returns:
            (r_mask, g_mask, b_mask) - binary masks indicating pixel positions
        """
        r_mask = np.zeros((height, width), dtype=np.float32)
        g_mask = np.zeros((height, width), dtype=np.float32)
        b_mask = np.zeros((height, width), dtype=np.float32)
        
        if bayer_pattern == BayerPattern.RGGB:
            # R G
            # G B
            r_mask[0::2, 0::2] = 1.0  # Top-left
            g_mask[0::2, 1::2] = 1.0  # Top-right
            g_mask[1::2, 0::2] = 1.0  # Bottom-left
            b_mask[1::2, 1::2] = 1.0  # Bottom-right
            
        elif bayer_pattern == BayerPattern.BGGR:
            # B G
            # G R
            b_mask[0::2, 0::2] = 1.0  # Top-left
            g_mask[0::2, 1::2] = 1.0  # Top-right
            g_mask[1::2, 0::2] = 1.0  # Bottom-left
            r_mask[1::2, 1::2] = 1.0  # Bottom-right
            
        elif bayer_pattern == BayerPattern.GRBG:
            # G R
            # B G
            g_mask[0::2, 0::2] = 1.0  # Top-left
            r_mask[0::2, 1::2] = 1.0  # Top-right
            b_mask[1::2, 0::2] = 1.0  # Bottom-left
            g_mask[1::2, 1::2] = 1.0  # Bottom-right
            
        elif bayer_pattern == BayerPattern.GBRG:
            # G B
            # R G
            g_mask[0::2, 0::2] = 1.0  # Top-left
            b_mask[0::2, 1::2] = 1.0  # Top-right
            r_mask[1::2, 0::2] = 1.0  # Bottom-left
            g_mask[1::2, 1::2] = 1.0  # Bottom-right
            
        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        
        return r_mask, g_mask, b_mask
    
    def modify_roi_out(self, roi_in: ROI) -> ROI:
        """Output ROI is same as input ROI"""
        return roi_in
    
    def modify_roi_in(self, roi_out: ROI) -> ROI:
        """Input ROI is same as output ROI"""
        return roi_out
    
    def set_method(self, method: DemosaicMethod):
        """Set demosaic method"""
        self.method = method
    
    def get_info(self) -> dict:
        """Get module information"""
        return {
            'name': self.name,
            'description': 'Reconstruct full RGB pixels from sensor CFA reading',
            'method': self.method.name,
            'channels_in': 1,
            'channels_out': 3,
            'colorspace_in': 'RAW',
            'colorspace_out': 'RGB'
        }


def test_demosaic():
    """Test the demosaic module"""
    print("=== Demosaic Module Test ===")
    
    # Create test RAW data (simulate Bayer pattern)
    height, width = 128, 128
    raw_data = np.random.uniform(0.3, 0.8, (height, width)).astype(np.float32)
    
    # Create synthetic Bayer pattern
    for y in range(height):
        for x in range(width):
            if y % 2 == 0 and x % 2 == 0:  # R
                raw_data[y, x] *= 1.2
            elif y % 2 == 1 and x % 2 == 1:  # B
                raw_data[y, x] *= 0.8
            # G pixels unchanged
    
    # Create test input buffer
    input_buffer = ImageBuffer(
        data=raw_data,
        width=width,
        height=height,
        channels=1,
        datatype=DataType.FLOAT32,
        colorspace=ColorSpace.RAW,
        bayer_pattern=BayerPattern.RGGB
    )
    
    # Create ROI
    roi = ROI(x=0, y=0, width=width, height=height)
    
    # Test demosaic
    demosaic = Demosaic()
    output_buffer = demosaic.process(input_buffer, roi, roi)
    
    print(f"Input shape: {input_buffer.data.shape}")
    print(f"Output shape: {output_buffer.data.shape}")
    print(f"Output channels: {output_buffer.channels}")
    print(f"Output colorspace: {output_buffer.colorspace}")
    print(f"Output range: [{output_buffer.data.min():.3f}, {output_buffer.data.max():.3f}]")
    
    # Verify output
    assert output_buffer.channels == 3
    assert output_buffer.colorspace == ColorSpace.RGB
    assert output_buffer.data.shape == (height, width, 3)
    assert 0.0 <= output_buffer.data.min() <= output_buffer.data.max() <= 1.0
    
    print("✓ Demosaic test passed!")


if __name__ == "__main__":
    test_demosaic() 