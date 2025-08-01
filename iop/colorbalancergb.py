
import numpy as np

class Colorbalancergb:
    """
    A Python implementation of darktable's color balance rgb module.

    This class is responsible for applying color grading to an image,
    separating shadows, mid-tones, and highlights.
    """
    # Conversion matrices
    # Transposed for easier use with np.dot(image, matrix)
    LMS_D65_TO_FILMLIGHT_RGB_D65_TRANS = np.array([
        [ 1.08771930, -0.66666667,  0.02061856],
        [-0.0877193 ,  1.66666667, -0.05154639],
        [ 0.        ,  0.        ,  1.03092784]
    ]).T

    FILMLIGHT_RGB_D65_TO_LMS_D65_TRANS = np.array([
        [0.95, 0.38, 0.00],
        [0.05, 0.62, 0.03],
        [0.00, 0.00, 0.97]
    ]).T

    XYZ_D65_TO_LMS_2006_D65_TRANS = np.array([
        [ 0.257085, -0.394427,  0.064856],
        [ 0.859943,  1.175800, -0.076250],
        [-0.031061,  0.106423,  0.559067]
    ]).T

    LMS_2006_D65_TO_XYZ_D65_TRANS = np.array([
        [ 1.80794659,  0.61783960, -0.12546960],
        [-1.29971660,  0.39595453,  0.20478038],
        [ 0.34785879, -0.04104687,  1.74274183]
    ]).T

    # sRGB D65 piece
    SRGB_TO_XYZ_D65_TRANS = np.array([
        [0.4124564, 0.2126729, 0.0193339],
        [0.3575761, 0.7151522, 0.1191920],
        [0.1804375, 0.0721750, 0.9503041]
    ]).T

    XYZ_D65_TO_SRGB_TRANS = np.linalg.inv(SRGB_TO_XYZ_D65_TRANS.T).T


    def __init__(self, **kwargs):
        # Parameters from dt_iop_colorbalancergb_params_t
        # Version 1
        self.shadows_Y = 0.0
        self.shadows_C = 0.0
        self.shadows_H = 0.0
        self.midtones_Y = 0.0
        self.midtones_C = 0.0
        self.midtones_H = 0.0
        self.highlights_Y = 0.0
        self.highlights_C = 0.0
        self.highlights_H = 0.0
        self.global_Y = 0.0
        self.global_C = 0.0
        self.global_H = 0.0
        self.shadows_weight = 1.0
        self.white_fulcrum = 0.0
        self.highlights_weight = 1.0
        self.chroma_shadows = 0.0
        self.chroma_highlights = 0.0
        self.chroma_global = 0.0
        self.chroma_midtones = 0.0
        self.saturation_global = 0.0
        self.saturation_highlights = 0.0
        self.saturation_midtones = 0.0
        self.saturation_shadows = 0.0
        self.hue_angle = 0.0

        # Version 2
        self.brilliance_global = 0.0
        self.brilliance_highlights = 0.0
        self.brilliance_midtones = 0.0
        self.brilliance_shadows = 0.0

        # Version 3
        self.mask_grey_fulcrum = 0.1845

        # Version 4
        self.vibrance = 0.0
        self.grey_fulcrum = 0.1845
        self.contrast = 0.0

        # Version 5
        self.saturation_formula = 1 # DT_COLORBALANCE_SATURATION_DTUCS

        # Update params from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _LMS_to_gradingRGB(self, lms):
        return np.dot(lms, self.LMS_D65_TO_FILMLIGHT_RGB_D65_TRANS)

    def _gradingRGB_to_LMS(self, rgb):
        return np.dot(rgb, self.FILMLIGHT_RGB_D65_TO_LMS_D65_TRANS)

    def _LMS_to_Yrg(self, LMS):
        Y = 0.68990272 * LMS[:, 0] + 0.34832189 * LMS[:, 1]
        
        a = np.sum(LMS, axis=1)
        lms_norm = LMS / a[:, np.newaxis]
        lms_norm[np.isnan(lms_norm)] = 0

        rgb_norm = self._LMS_to_gradingRGB(lms_norm)

        Yrg = np.zeros_like(LMS)
        Yrg[:, 0] = Y
        Yrg[:, 1] = rgb_norm[:, 0]
        Yrg[:, 2] = rgb_norm[:, 1]
        return Yrg

    def _Yrg_to_LMS(self, Yrg):
        Y = Yrg[:, 0]
        r = Yrg[:, 1]
        g = Yrg[:, 2]
        b = 1.0 - r - g
        
        rgb_norm = np.stack([r, g, b], axis=-1)
        lms_norm = self._gradingRGB_to_LMS(rgb_norm)

        denom = (0.68990272 * lms_norm[:, 0] + 0.34832189 * lms_norm[:, 1])
        # Avoid division by zero
        denom[denom == 0] = 1e-9
        
        a = Y / denom
        
        LMS = lms_norm * a[:, np.newaxis]
        return LMS

    def _Yrg_to_Ych(self, Yrg):
        Y = Yrg[:, 0]
        r = Yrg[:, 1] - 0.21902143
        g = Yrg[:, 2] - 0.54371398
        
        c = np.hypot(g, r)
        
        cos_h = np.ones_like(c)
        sin_h = np.zeros_like(c)
        
        non_zero_c = c != 0
        cos_h[non_zero_c] = r[non_zero_c] / c[non_zero_c]
        sin_h[non_zero_c] = g[non_zero_c] / c[non_zero_c]

        return np.stack([Y, c, cos_h, sin_h], axis=-1)

    def _Ych_to_Yrg(self, Ych):
        Y = Ych[:, 0]
        c = Ych[:, 1]
        cos_h = Ych[:, 2]
        sin_h = Ych[:, 3]

        r = c * cos_h + 0.21902143
        g = c * sin_h + 0.54371398
        
        return np.stack([Y, r, g], axis=-1)

    def _gamut_check_Yrg(self, Ych):
        Yrg = self._Ych_to_Yrg(Ych)
        max_c = np.copy(Ych[:, 1])

        D65_r = 0.21902143
        D65_g = 0.54371398
        cos_h = Ych[:, 2]
        sin_h = Ych[:, 3]

        # if(Yrg[1] < 0.f) max_c = fminf(-D65_r / cos_h, max_c);
        mask = Yrg[:, 1] < 0
        max_c[mask] = np.minimum(-D65_r / cos_h[mask], max_c[mask])

        # if(Yrg[2] < 0.f) max_c = fminf(-D65_g / sin_h, max_c);
        mask = Yrg[:, 2] < 0
        max_c[mask] = np.minimum(-D65_g / sin_h[mask], max_c[mask])

        # if(Yrg[1] + Yrg[2] > 1.f) max_c = fminf((1.f - D65_r - D65_g) / (cos_h + sin_h), max_c);
        mask = (Yrg[:, 1] + Yrg[:, 2]) > 1.0
        # handle potential division by zero for cos_h + sin_h
        denom = cos_h[mask] + sin_h[mask]
        denom[denom == 0] = 1e-9
        max_c[mask] = np.minimum((1.0 - D65_r - D65_g) / denom, max_c[mask])
        
        Ych[:, 1] = max_c
        return Ych

    def _opacity_masks(self, x, shadows_w, highlights_w, midtones_weight):
        x_offset = x - self.mask_grey_fulcrum
        # Avoid division by zero
        safe_mask_grey_fulcrum = self.mask_grey_fulcrum if self.mask_grey_fulcrum != 0 else 1e-9
        x_offset_norm = x_offset / safe_mask_grey_fulcrum

        # Sigmoid functions for shadows and highlights
        alpha = 1.0 / (1.0 + np.exp(x_offset_norm * shadows_w))  # shadows
        beta = 1.0 / (1.0 + np.exp(-x_offset_norm * highlights_w)) # highlights
        
        alpha_comp = 1.0 - alpha
        beta_comp = 1.0 - beta
        
        gamma = np.exp(-np.square(x_offset) * midtones_weight / 4.0) * np.square(alpha_comp) * np.square(beta_comp) * 8.0 # midtones
        
        masks = np.stack([alpha, gamma, beta], axis=-1)
        masks_comp = np.stack([alpha_comp, 1.0 - gamma, beta_comp], axis=-1)
        
        return masks, masks_comp


    def process(self, image_data, image_info=None):
        """
        Process the image using the color balance rgb logic.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C) in linear sRGB space.

        Returns:
            np.ndarray: The processed image.
        """
        h, w, c = image_data.shape
        pixels = image_data.reshape(-1, c)
        
        # --- Prepare internal parameters from user settings ---
        # See commit_params in colorbalancergb.c
        _DEG_TO_RAD = lambda x: (x - 30.0) * np.pi / 180.0

        # Global
        Ych_norm = np.array([[1.0, 0.0, 1.0, 0.0]]) # Y, c, cos(h), sin(h)
        RGB_norm = self._LMS_to_gradingRGB(self._Yrg_to_LMS(self._Ych_to_Yrg(Ych_norm)))
        
        Ych_global = np.array([1.0, self.global_C, np.cos(_DEG_TO_RAD(self.global_H)), np.sin(_DEG_TO_RAD(self.global_H))])
        global_params = self._LMS_to_gradingRGB(self._Yrg_to_LMS(self._Ych_to_Yrg(Ych_global[np.newaxis, :])))[0]
        global_params = (global_params - RGB_norm) + RGB_norm * self.global_Y

        # Shadows
        Ych_shadows = np.array([1.0, self.shadows_C, np.cos(_DEG_TO_RAD(self.shadows_H)), np.sin(_DEG_TO_RAD(self.shadows_H))])
        shadows_params = self._LMS_to_gradingRGB(self._Yrg_to_LMS(self._Ych_to_Yrg(Ych_shadows[np.newaxis, :])))[0]
        shadows_params = 1.0 + (shadows_params - RGB_norm) + self.shadows_Y

        # Highlights
        Ych_highlights = np.array([1.0, self.highlights_C, np.cos(_DEG_TO_RAD(self.highlights_H)), np.sin(_DEG_TO_RAD(self.highlights_H))])
        highlights_params = self._LMS_to_gradingRGB(self._Yrg_to_LMS(self._Ych_to_Yrg(Ych_highlights[np.newaxis, :])))[0]
        highlights_params = 1.0 + (highlights_params - RGB_norm) + self.highlights_Y
        
        # Midtones
        Ych_midtones = np.array([1.0, self.midtones_C, np.cos(_DEG_TO_RAD(self.midtones_H)), np.sin(_DEG_TO_RAD(self.midtones_H))])
        midtones_params = self._LMS_to_gradingRGB(self._Yrg_to_LMS(self._Ych_to_Yrg(Ych_midtones[np.newaxis, :])))[0]
        midtones_params = 1.0 / (1.0 + (midtones_params - RGB_norm))
        midtones_Y_param = 1.0 / (1.0 + self.midtones_Y)
        white_fulcrum_param = np.exp2(self.white_fulcrum)
        contrast_param = 1.0 + self.contrast

        # Shadows / Highlights / Midtones weights for mask
        shadows_w = 2.0 + self.shadows_weight * 2.0
        highlights_w = 2.0 + self.highlights_weight * 2.0
        shadows_weight_sq = shadows_w**2
        highlights_weight_sq = highlights_w**2
        midtones_weight_denom = shadows_weight_sq + highlights_weight_sq
        if midtones_weight_denom == 0: midtones_weight_denom = 1e-9 # defensive
        midtones_weight = (shadows_weight_sq * highlights_weight_sq) / midtones_weight_denom


        # --- Start processing ---
        # 1. Color space conversion: RGB -> XYZ -> LMS
        pixels_xyz = np.dot(pixels, self.SRGB_TO_XYZ_D65_TRANS)
        pixels_lms = np.dot(pixels_xyz, self.XYZ_D65_TO_LMS_2006_D65_TRANS)

        # 2. LMS -> Yrg -> Ych
        pixels_Yrg = self._LMS_to_Yrg(pixels_lms)
        pixels_Ych = self._Yrg_to_Ych(pixels_Yrg)
        pixels_Ych[:, 0] = np.maximum(pixels_Ych[:, 0], 0) # Sanitize luminance

        # 3. Opacity masks
        luma_for_mask = pixels_Ych[:, 0]**0.41012058
        opacities, opacities_comp = self._opacity_masks(luma_for_mask, shadows_w, highlights_w, midtones_weight)
        
        # 4. Hue shift
        cos_h, sin_h = pixels_Ych[:, 2].copy(), pixels_Ych[:, 3].copy()
        angle = self.hue_angle * np.pi / 180.0
        cos_rot, sin_rot = np.cos(angle), np.sin(angle)
        pixels_Ych[:, 2] = cos_h * cos_rot - sin_h * sin_rot
        pixels_Ych[:, 3] = cos_h * sin_rot + sin_h * cos_rot
        
        # 5. Linear chroma and vibrance
        chroma_params = np.array([self.chroma_shadows, self.chroma_midtones, self.chroma_highlights])
        chroma_boost = self.chroma_global + np.sum(opacities * chroma_params, axis=1)
        vibrance_eff = self.vibrance * (1.0 - pixels_Ych[:, 1]**np.abs(self.vibrance))
        chroma_factor = np.maximum(1.0 + chroma_boost + vibrance_eff, 0.0)
        pixels_Ych[:, 1] *= chroma_factor
        
        # 6. Gamut check
        pixels_Ych = self._gamut_check_Yrg(pixels_Ych)
        
        # 7. Convert to grading space: Ych -> Yrg -> LMS -> gradingRGB
        pixels_Yrg_graded = self._Ych_to_Yrg(pixels_Ych)
        pixels_lms_graded = self._Yrg_to_LMS(pixels_Yrg_graded)
        pixels_grading_rgb = self._LMS_to_gradingRGB(pixels_lms_graded)
        
        # 8. Apply color balance (offset, power, slope)
        # Global offset
        pixels_grading_rgb += global_params[:3]
        
        # Shadows/Highlights slope
        pixels_grading_rgb *= opacities_comp[:, 2, np.newaxis] * \
                              (opacities_comp[:, 0, np.newaxis] + opacities[:, 0, np.newaxis] * shadows_params[:3]) + \
                              opacities[:, 2, np.newaxis] * highlights_params[:3]

        # Midtones power
        sign = np.sign(pixels_grading_rgb)
        abs_rgb = np.abs(pixels_grading_rgb)
        safe_white_fulcrum = white_fulcrum_param if white_fulcrum_param != 0 else 1e-9
        scaled_rgb = abs_rgb / safe_white_fulcrum
        powered_rgb = np.power(scaled_rgb, midtones_params[:3])
        pixels_grading_rgb = powered_rgb * sign * safe_white_fulcrum
        
        # 9. Convert back for contrast: gradingRGB -> LMS -> Yrg
        pixels_lms_graded = self._gradingRGB_to_LMS(pixels_grading_rgb)
        pixels_Yrg_graded = self._LMS_to_Yrg(pixels_lms_graded)
        
        # 10. Midtones Y power (gamma) and contrast
        safe_white_fulcrum = white_fulcrum_param if white_fulcrum_param != 0 else 1e-9
        Y_mid = np.maximum(pixels_Yrg_graded[:, 0] / safe_white_fulcrum, 0)
        pixels_Yrg_graded[:, 0] = np.power(Y_mid, midtones_Y_param) * safe_white_fulcrum

        safe_grey_fulcrum = self.grey_fulcrum if self.grey_fulcrum != 0 else 1e-9
        Y_contrast = pixels_Yrg_graded[:, 0] / safe_grey_fulcrum
        pixels_Yrg_graded[:, 0] = safe_grey_fulcrum * np.power(Y_contrast, contrast_param)
        
        # 11. Final conversion back to output RGB: Yrg -> LMS -> XYZ -> RGB
        final_lms = self._Yrg_to_LMS(pixels_Yrg_graded)
        final_xyz = np.dot(final_lms, self.LMS_2006_D65_TO_XYZ_D65_TRANS)
        final_rgb = np.dot(final_xyz, self.XYZ_D65_TO_SRGB_TRANS)
        
        return final_rgb.reshape(h, w, c) 