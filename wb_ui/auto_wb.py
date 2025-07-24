# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from iop.whitebalance_dt_port import WhitebalanceDtPort

class AutoWhiteBalance:
    """
    Analyzes an image to automatically determine optimal Temperature and Tint settings.
    """

    @staticmethod
    def gray_world_awb(image_rgb: np.ndarray) -> np.ndarray:
        """
        Calculates white balance coefficients based on the Gray World assumption.

        Args:
            image_rgb (np.ndarray): A linear RGB image in float format [0.0, 1.0].

        Returns:
            np.ndarray: The calculated [R, G, B] coefficients, normalized to the G channel.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Input image must be an RGB image.")
            
        # Calculate the average of each channel
        avg_r = np.mean(image_rgb[:, :, 0])
        avg_g = np.mean(image_rgb[:, :, 1])
        avg_b = np.mean(image_rgb[:, :, 2])
        
        # Coefficients are the inverse of the average, normalized to green
        # Add a small epsilon to avoid division by zero
        coeffs = np.array([
            avg_g / (avg_r + 1e-8),
            1.0,
            avg_g / (avg_b + 1e-8)
        ])
        
        return coeffs

    @staticmethod
    def _objective_function(params: tuple[float, float], target_coeffs: np.ndarray) -> float:
        """
        The function to minimize. It calculates the squared error between the
        coeffs produced by a (temp, tint) pair and the target coeffs.
        """
        temp, gui_tint = params
        try:
            # We need to get the internal_tint for the calculation.
            # We can instantiate the class briefly to use its mapping function.
            # This is slightly inefficient but decouples the logic nicely.
            wb_port_instance = WhitebalanceDtPort(temp_k=temp, gui_tint=gui_tint)
            internal_tint = wb_port_instance.internal_tint

            calculated_coeffs = WhitebalanceDtPort._calculate_coeffs_static(
                temp_k=temp,
                internal_tint=internal_tint
            )
            error = np.sum((calculated_coeffs - target_coeffs) ** 2)
            return error
        except Exception:
            # Return a large error if calculation fails for some reason
            return 1e6

    @staticmethod
    def find_temp_tint_for_coeffs(target_coeffs: np.ndarray) -> tuple[float, float] | None:
        """
        Finds the Temperature and Tint that produce the given RGB coefficients.
        """
        initial_guess = [5000.0, 1.0]
        bounds = [(1900, 25000), (0.135, 2.326)]

        result = minimize(
            AutoWhiteBalance._objective_function,
            x0=initial_guess,
            args=(target_coeffs,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-9, 'gtol': 1e-7, 'maxiter': 200}
        )

        if result.success:
            return result.x[0], result.x[1]  # temp, tint
        else:
            print("Warning: Auto WB optimization did not converge.")
            return None

    @classmethod
    def analyze(cls, image_rgb: np.ndarray) -> tuple[float, float] | None:
        """
        Performs the full auto white balance analysis on an image.
        """
        print("Starting auto white balance analysis...")
        # 1. Get target coeffs from the image
        target_coeffs = cls.gray_world_awb(image_rgb)
        print(f"Gray World target coeffs: {target_coeffs}")
        
        # 2. Find the Temp/Tint that produces these coeffs
        result = cls.find_temp_tint_for_coeffs(target_coeffs)
        
        if result:
            temp, tint = result
            print(f"Found optimal settings: Temp={temp:.1f}K, Tint={tint:.3f}")
            return temp, tint
        else:
            return None 