# -*- coding: utf-8 -*-
"""
Manages loading and accessing white balance presets from the wb_presets.json file.
"""

import json
import os
from collections import defaultdict
from tqdm import tqdm

class WBPresetsManager:
    """
    A manager class that loads, parses, and provides access to the white
    balance presets stored in the wb_presets.json file.

    This class is designed to be instantiated once, loading the large JSON file
    into a more efficient dictionary structure for fast lookups.
    """

    def __init__(self):
        """
        Initializes the manager, loading and parsing the preset data.
        """
        self.presets = defaultdict(dict)
        self._load_presets()

    def _load_presets(self):
        """
        Loads the wb_presets.json file and parses it into a structured
        dictionary for efficient access.

        The final structure is:
        self.presets['Canon']['Canon EOS 5D Mark II']['Daylight'] = [r, g, b, g2]
        """
        json_path = os.path.join(os.path.dirname(__file__), 'wb_presets.json')

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading or parsing wb_presets.json: {e}")
            return

        # The JSON has a top-level "wb_presets" key
        all_makers = data.get("wb_presets", [])
        print("Loading and parsing white balance presets...")
        for maker_entry in tqdm(all_makers, desc="Processing camera makers"):
            maker_name = maker_entry.get("maker")
            if not maker_name:
                continue

            for model_entry in maker_entry.get("models", []):
                model_name = model_entry.get("model")
                if not model_name:
                    continue

                if model_name not in self.presets[maker_name]:
                    self.presets[maker_name][model_name] = {}

                for preset_entry in model_entry.get("presets", []):
                    preset_name = preset_entry.get("name")
                    channels = preset_entry.get("channels")
                    if preset_name and channels:
                        self.presets[maker_name][model_name][preset_name] = channels

    def get_coeffs(self, maker, model, preset_name):
        """
        Retrieves the white balance coefficients for a given camera and preset.

        Args:
            maker (str): The manufacturer of the camera (e.g., "Canon").
            model (str): The model of the camera (e.g., "PowerShot G1 X Mark II").
            preset_name (str): The name of the WB preset (e.g., "Daylight").

        Returns:
            list[float] | None: A list containing the 4 channel multipliers
                                [R, G, B, G2], or None if not found.
        """
        return self.presets.get(maker, {}).get(model, {}).get(preset_name) 