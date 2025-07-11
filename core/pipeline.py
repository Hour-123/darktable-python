"""
pipeline - The core processing pipeline for the darktable_python project.
"""

import os
import yaml
import numpy as np
import rawpy
import imageio
import importlib
import time

def load_iop_module(module_name: str):
    """
    Dynamically loads an IOP module class from the iop directory.
    
    It assumes the module file is named 'module_name.py' and the class
    is the CamelCase version of the module_name (e.g., 'tonecurve' -> 'ToneCurve').
    """
    try:
        # Construct the module path (e.g., 'iop.exposure')
        module_path = f"iop.{module_name}"
        
        # Import the module
        imported_module = importlib.import_module(module_path)
        
        # Convert module_name (snake_case) to ClassName (CamelCase)
        class_name = "".join(word.capitalize() for word in module_name.split('_'))
        
        # Get the class from the imported module
        module_class = getattr(imported_module, class_name)
        return module_class
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not load IOP module '{module_name}'.")
        print(f"Please ensure 'iop/{module_name}.py' exists and contains a class named '{class_name}'.")
        raise e

def run_pipeline(config_path: str):
    """
    Runs the entire image processing pipeline based on a YAML config file.
    """
    print("--- Starting Image Processing Pipeline ---")
    start_time = time.time()

    # Get the absolute path of the config file to resolve other paths correctly
    base_dir = os.path.dirname(os.path.abspath(config_path))

    # 1. Load Configuration
    print(f"1. Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    input_path = os.path.join(base_dir, config['input_file'])
    output_path = os.path.join(base_dir, 'output', config['output_file'])
    pipeline_steps = config['pipeline']
    
    # 2. Decode RAW file
    print(f"2. Decoding RAW file: {input_path}")
    try:
        with rawpy.imread(input_path) as raw:
            raw_image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
        image_data = raw_image.astype(np.float32) / 65535.0
        print(f"   - RAW decoding complete. Image dimensions: {image_data.shape}")
    except Exception as e:
        print(f"Error decoding RAW file: {e}")
        return

    # 3. Execute Pipeline Steps
    print("3. Executing processing pipeline...")
    for i, step in enumerate(pipeline_steps):
        module_name = step['module']
        params = step.get('params', {})
        print(f"   - Step {i+1}/{len(pipeline_steps)}: Applying module '{module_name}' with params {params}")
        
        try:
            # Dynamically load the module class
            IopModule = load_iop_module(module_name)
            
            # Initialize the module with its parameters
            iop_instance = IopModule(**params)
            
            # Process the image
            image_data = iop_instance.process(image_data)
        except Exception as e:
            print(f"   - ERROR at step {i+1} ('{module_name}'). Aborting pipeline.")
            print(f"   - Details: {e}")
            return
    
    print("   - Pipeline execution finished.")

    # 4. Finalize and Save Output
    print("4. Finalizing image (Gamma Correction)")
    final_image = np.clip(image_data, 0, 1) ** (1 / 2.2)
    output_image_uint8 = (final_image * 255).astype(np.uint8)

    print(f"5. Saving final image to: {output_path}")
    imageio.imwrite(output_path, output_image_uint8)

    end_time = time.time()
    print(f"--- Pipeline Finished in {end_time - start_time:.2f} seconds ---")


if __name__ == '__main__':
    # Assume the config file is in the parent directory of this script's location
    # core/ -> config is in parent
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(project_root, 'pipeline_config.yaml')
    
    # We need to add the project root to the python path so that imports like 'iop.exposure' work
    import sys
    sys.path.append(project_root)

    run_pipeline(default_config) 