# -*- coding: utf-8 -*-
"""
A module for color science calculations related to white balance.

This module provides the necessary data and functions to convert between
color temperature/tint and RGB multipliers, based on the methodology
used in darktable.
"""

import numpy as np

# CIE 1931 2-degree standard colorimetric observer data.
# This table maps wavelengths (from 380nm to 780nm in 5nm steps) to the
# CIE XYZ color matching functions.
# Source: darktable/src/external/cie_colorimetric_tables.c
CIE_1931_WAVELENGTHS = np.arange(380, 781, 5)

CIE_1931_XYZ = np.array([
    [0.001368, 0.000039, 0.006450],
    [0.002236, 0.000064, 0.010550],
    [0.004243, 0.000120, 0.020050],
    [0.007650, 0.000217, 0.036210],
    [0.014310, 0.000396, 0.067850],
    [0.023190, 0.000640, 0.110200],
    [0.043510, 0.001210, 0.207400],
    [0.077630, 0.002180, 0.371300],
    [0.134380, 0.004000, 0.645600],
    [0.214770, 0.007300, 1.039050],
    [0.283900, 0.011600, 1.385600],
    [0.328500, 0.016840, 1.622960],
    [0.348280, 0.023000, 1.747060],
    [0.348060, 0.029800, 1.782600],
    [0.336200, 0.038000, 1.772110],
    [0.318700, 0.048000, 1.744100],
    [0.290800, 0.060000, 1.669200],
    [0.251100, 0.073900, 1.528100],
    [0.195360, 0.090980, 1.287640],
    [0.142100, 0.112600, 1.041900],
    [0.095640, 0.139020, 0.812950],
    [0.057950, 0.169300, 0.616200],
    [0.032010, 0.208020, 0.465180],
    [0.014700, 0.258600, 0.353300],
    [0.004900, 0.323000, 0.272000],
    [0.002400, 0.407300, 0.212300],
    [0.009300, 0.503000, 0.158200],
    [0.029100, 0.608200, 0.111700],
    [0.063270, 0.710000, 0.078250],
    [0.109600, 0.793200, 0.057250],
    [0.165500, 0.862000, 0.042160],
    [0.225750, 0.914850, 0.029840],
    [0.290400, 0.954000, 0.020300],
    [0.359700, 0.980300, 0.013400],
    [0.433450, 0.994950, 0.008750],
    [0.512050, 1.000000, 0.005750],
    [0.594500, 0.995000, 0.003900],
    [0.678400, 0.978600, 0.002750],
    [0.762100, 0.952000, 0.002100],
    [0.842500, 0.915400, 0.001800],
    [0.916300, 0.870000, 0.001650],
    [0.978600, 0.816300, 0.001400],
    [1.026300, 0.757000, 0.001100],
    [1.056700, 0.694900, 0.001000],
    [1.062200, 0.631000, 0.000800],
    [1.045600, 0.566800, 0.000600],
    [1.002600, 0.503000, 0.000340],
    [0.938400, 0.441200, 0.000240],
    [0.854450, 0.381000, 0.000190],
    [0.751400, 0.321000, 0.000100],
    [0.642400, 0.265000, 0.000050],
    [0.541900, 0.217000, 0.000030],
    [0.447900, 0.175000, 0.000020],
    [0.360800, 0.138200, 0.000010],
    [0.283500, 0.107000, 0.000000],
    [0.218700, 0.081600, 0.000000],
    [0.164900, 0.061000, 0.000000],
    [0.121200, 0.044580, 0.000000],
    [0.087400, 0.032000, 0.000000],
    [0.063600, 0.023200, 0.000000],
    [0.046770, 0.017000, 0.000000],
    [0.032900, 0.011920, 0.000000],
    [0.022700, 0.008210, 0.000000],
    [0.015840, 0.005723, 0.000000],
    [0.011359, 0.004102, 0.000000],
    [0.008111, 0.002929, 0.000000],
    [0.005790, 0.002091, 0.000000],
    [0.004109, 0.001484, 0.000000],
    [0.002899, 0.001047, 0.000000],
    [0.002049, 0.000740, 0.000000],
    [0.001440, 0.000520, 0.000000],
    [0.001000, 0.000361, 0.000000],
    [0.000690, 0.000249, 0.000000],
    [0.000476, 0.000172, 0.000000],
    [0.000332, 0.000120, 0.000000],
    [0.000235, 0.000085, 0.000000],
    [0.000166, 0.000060, 0.000000],
    [0.000117, 0.000042, 0.000000],
    [0.000083, 0.000030, 0.000000],
    [0.000059, 0.000021, 0.000000],
    [0.000042, 0.000015, 0.000000],
])

# CIE standard daylight components data.
# This table provides components (S0, S1, S2) for calculating the spectral
# power distribution of a daylight illuminant at a given color temperature.
# The wavelengths range from 300nm to 830nm in 5nm steps.
# Source: darktable/src/external/cie_colorimetric_tables.c
CIE_DAYLIGHT_WAVELENGTHS = np.arange(300, 831, 5)

CIE_DAYLIGHT_COMPONENTS = np.array([
    [0.04, 0.02, 0.00],
    [3.02, 2.26, 1.00],
    [6.00, 4.50, 2.00],
    [17.80, 13.45, 3.00],
    [29.60, 22.40, 4.00],
    [42.45, 32.20, 6.25],
    [55.30, 42.00, 8.50],
    [56.30, 41.30, 8.15],
    [57.30, 40.60, 7.80],
    [59.55, 41.10, 7.25],
    [61.80, 41.60, 6.70],
    [61.65, 39.80, 6.00],
    [61.50, 38.00, 5.30],
    [65.15, 40.20, 5.70],
    [68.80, 42.40, 6.10],
    [66.10, 40.45, 4.55],
    [63.40, 38.50, 3.00],
    [64.60, 36.75, 2.10],
    [65.80, 35.00, 1.20],
    [80.30, 39.20, 0.05],
    [94.80, 43.40, -1.10],
    [99.80, 44.85, -0.80],
    [104.80, 46.30, -0.50],
    [105.35, 45.10, -0.60],
    [105.90, 43.90, -0.70],
    [101.35, 40.50, -0.95],
    [96.80, 37.10, -1.20],
    [105.35, 36.90, -1.90],
    [113.90, 36.70, -2.60],
    [119.75, 36.30, -2.75],
    [125.60, 35.90, -2.90],
    [125.55, 34.25, -2.85],
    [125.50, 32.60, -2.80],
    [123.40, 30.25, -2.70],
    [121.30, 27.90, -2.60],
    [121.30, 26.10, -2.60],
    [121.30, 24.30, -2.60],
    [117.40, 22.20, -2.20],
    [113.50, 20.10, -1.80],
    [113.30, 18.15, -1.65],
    [113.10, 16.20, -1.50],
    [111.95, 14.70, -1.40],
    [110.80, 13.20, -1.30],
    [108.65, 10.90, -1.25],
    [106.50, 8.60, -1.20],
    [107.65, 7.35, -1.10],
    [108.80, 6.10, -1.00],
    [107.05, 5.15, -0.75],
    [105.30, 4.20, -0.50],
    [104.85, 3.05, -0.40],
    [104.40, 1.90, -0.30],
    [102.20, 0.95, -0.15],
    [100.00, 0.00, 0.00],
    [98.00, -0.80, 0.10],
    [96.00, -1.60, 0.20],
    [95.55, -2.55, 0.35],
    [95.10, -3.50, 0.50],
    [92.10, -3.50, 1.30],
    [89.10, -3.50, 2.10],
    [89.80, -4.65, 2.65],
    [90.50, -5.80, 3.20],
    [90.40, -6.50, 3.65],
    [90.30, -7.20, 4.10],
    [89.35, -7.90, 4.40],
    [88.40, -8.60, 4.70],
    [86.20, -9.05, 4.90],
    [84.00, -9.50, 5.10],
    [84.55, -10.20, 5.90],
    [85.10, -10.90, 6.70],
    [83.50, -10.80, 7.00],
    [81.90, -10.70, 7.30],
    [82.25, -11.35, 7.95],
    [82.60, -12.00, 8.60],
    [83.75, -13.00, 9.20],
    [84.90, -14.00, 9.80],
    [83.10, -13.80, 10.00],
    [81.30, -13.60, 10.20],
    [76.60, -12.80, 9.25],
    [71.90, -12.00, 8.30],
    [73.10, -12.65, 8.95],
    [74.30, -13.30, 9.60],
    [75.35, -13.10, 9.05],
    [76.40, -12.90, 8.50],
    [69.85, -11.75, 7.75],
    [63.30, -10.60, 7.00],
    [67.50, -11.10, 7.30],
    [71.70, -11.60, 7.60],
    [74.35, -11.90, 7.80],
    [77.00, -12.20, 8.00],
    [71.10, -11.20, 7.35],
    [65.20, -10.20, 6.70],
    [56.45, -9.00, 5.95],
    [47.70, -7.80, 5.20],
    [58.15, -9.50, 6.30],
    [68.60, -11.20, 7.40],
    [66.80, -10.80, 7.10],
    [65.00, -10.40, 6.80],
    [65.50, -10.50, 6.90],
    [66.00, -10.60, 7.00],
    [63.50, -10.15, 6.70],
    [61.00, -9.70, 6.40],
    [57.15, -9.00, 5.95],
    [53.30, -8.30, 5.50],
    [56.10, -8.80, 5.80],
    [58.90, -9.30, 6.10],
    [60.40, -9.55, 6.30],
    [61.90, -9.80, 6.50],
])


def spd_blackbody(wavelengths_nm, temp_k):
    """
    Calculates the spectral power distribution (SPD) of a blackbody radiator
    at a given temperature using Planck's law.

    Args:
        wavelengths_nm (np.ndarray): Wavelengths in nanometers.
        temp_k (float): Temperature in Kelvin.

    Returns:
        np.ndarray: The spectral power distribution for the given wavelengths.
    """
    wavelengths_m = wavelengths_nm * 1e-9
    # Planck's law constants
    c1 = 3.741771e-16  # First radiation constant
    c2 = 1.43877e-2   # Second radiation constant
    # M = c1 * λ^-5 / (exp(c2 / (λ * T)) - 1)
    power = (c1 * np.power(wavelengths_m, -5)) / (np.exp(c2 / (wavelengths_m * temp_k)) - 1)
    return power


def spd_daylight(wavelengths_nm, temp_k):
    """
    Calculates the spectral power distribution (SPD) of a daylight illuminant
    at a given temperature.

    This is based on the method described by Wyszecki, see
    http://www.brucelindbloom.com/index.html?Eqn_DIlluminant.html for reference.

    Args:
        wavelengths_nm (np.ndarray): Wavelengths in nanometers.
        temp_k (float): Temperature in Kelvin.

    Returns:
        np.ndarray: The spectral power distribution for the given wavelengths.
    """
    if temp_k <= 7000.:
        x = -4.6070e9 / (temp_k ** 3) + 2.9678e6 / (temp_k ** 2) + 0.09911e3 / temp_k + 0.244063
    else:
        x = -2.0064e9 / (temp_k ** 3) + 1.9018e6 / (temp_k ** 2) + 0.24748e3 / temp_k + 0.237040

    y = -3.000 * x * x + 2.870 * x - 0.275

    M1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / (0.0241 + 0.2562 * x - 0.7341 * y)
    M2 = (0.0300 - 31.4424 * x + 30.0717 * y) / (0.0241 + 0.2562 * x - 0.7341 * y)

    # Interpolate the S0, S1, S2 components for the given wavelengths.
    S0 = np.interp(wavelengths_nm, CIE_DAYLIGHT_WAVELENGTHS, CIE_DAYLIGHT_COMPONENTS[:, 0])
    S1 = np.interp(wavelengths_nm, CIE_DAYLIGHT_WAVELENGTHS, CIE_DAYLIGHT_COMPONENTS[:, 1])
    S2 = np.interp(wavelengths_nm, CIE_DAYLIGHT_WAVELENGTHS, CIE_DAYLIGHT_COMPONENTS[:, 2])

    # Compute the final SPD value
    return S0 + M1 * S1 + M2 * S2


def spectrum_to_xyz(spd_function, **kwargs):
    """
    Converts a spectral power distribution (SPD) to CIE XYZ values.

    This function takes an SPD function (like spd_blackbody or spd_daylight),
    calculates the SPD over the standard observer's wavelength range, and
    then integrates it against the CIE 1931 color matching functions to
    get the XYZ tristimulus values.

    Args:
        spd_function (callable): A function that returns an SPD for a given
                                 set of wavelengths (e.g., spd_daylight).
        **kwargs: Keyword arguments to be passed to the spd_function (e.g., temp_k).

    Returns:
        np.ndarray: A 3-element array containing the X, Y, Z values.
    """
    # Calculate the spectral power distribution for the standard wavelengths
    spd = spd_function(CIE_1931_WAVELENGTHS, **kwargs)

    # Integrate the SPD against the CIE color matching functions
    # The integration is a sum of (SPD * CMF) over all wavelengths,
    # which is equivalent to a dot product.
    # We use np.trapezoid for a more accurate integration than a simple sum.
    x = np.trapezoid(spd * CIE_1931_XYZ[:, 0], CIE_1931_WAVELENGTHS)
    y = np.trapezoid(spd * CIE_1931_XYZ[:, 1], CIE_1931_WAVELENGTHS)
    z = np.trapezoid(spd * CIE_1931_XYZ[:, 2], CIE_1931_WAVELENGTHS)

    # The result should be normalized so that Y is 1.0
    if y == 0:
        return np.array([0., 0., 0.])

    return np.array([x, y, z]) / y


def temperature_tint_to_xyz(temp_k, tint):
    """
    Converts color temperature and tint to CIE XYZ values.

    This function first calculates the XYZ values for a given temperature,
    and then applies a simplified tint correction.

    Args:
        temp_k (float): The color temperature in Kelvin.
        tint (float): The tint adjustment value. A value > 1.0 shifts
                      towards green, < 1.0 shifts towards magenta.

    Returns:
        np.ndarray: A 3-element array containing the X, Y, Z values.
    """
    # First, get the XYZ values for the given temperature.
    if temp_k < 4000:
        xyz = spectrum_to_xyz(spd_blackbody, temp_k=temp_k)
    else:
        xyz = spectrum_to_xyz(spd_daylight, temp_k=temp_k)

    # Apply a simplified tint correction by adjusting the Y (greenish) component.
    if tint != 1.0:
        xyz[1] *= tint

    return xyz


def temperature_to_xyz(temp_k):
    """
    Converts a color temperature in Kelvin to CIE XYZ values.

    This function is a wrapper around temperature_tint_to_xyz with a neutral tint.

    Args:
        temp_k (float): The color temperature in Kelvin.

    Returns:
        np.ndarray: A 3-element array containing the X, Y, Z values.
    """
    return temperature_tint_to_xyz(temp_k, 1.0)


# XYZ to Camera RGB conversion matrix for the Canon EOS 5D Mark II.
# This matrix is used to convert from the standard CIE XYZ color space to the
# camera's specific native RGB color space. The values are scaled by 1/1,000,000
# as found in darktable's source code.
# Source: darktable/src/common/colormatrices.c
XYZ_TO_CAM_CANON_5D_MARK_II = np.array([
    [967590, 399139, 36026],
    [-52094, 819046, -232071],
    [144455, -143158, 1069305]
]) / 1000000.0


def xyz_to_camera_rgb(xyz, matrix):
    """
    Converts CIE XYZ values to camera-specific RGB values using a given matrix.

    Args:
        xyz (np.ndarray): A 3-element array containing the X, Y, Z values.
        matrix (np.ndarray): A 3x3 conversion matrix from XYZ to camera RGB.
                             This implementation assumes the matrix is actually
                             CAM_to_XYZ, so it uses its inverse.

    Returns:
        np.ndarray: A 3-element array containing the camera-specific RGB values.
    """
    # It is a common convention to store the Camera RGB -> XYZ matrix.
    # To go from XYZ -> Camera RGB, we need its inverse.
    matrix_inv = np.linalg.inv(matrix)
    return np.dot(matrix_inv, xyz)