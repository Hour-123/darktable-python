# darktable_python

This project is a Python-based image processing pipeline, inspired by the scene-referred workflow found in modern digital darkroom software like darktable. It uses a modular approach with various Image Operation (IOP) modules that can be chained together using a configuration file.

## Features

The following processing modules are currently implemented:

-   **Bloom**: Creates a soft, glowing effect around highlights.
-   **Blurs**: Applies different types of blurs (Gaussian, Lens, Motion).
-   **Borders**: Adds a photographic border to the image.
-   **Color Balance RGB**: Adjusts the color balance of the image based on lift, gamma, and gain.
-   **Diffuse**: Simulates light diffusion or glows, often used for halation effects.
-   **Dither**: Reduces color banding by adding noise.
-   **Exposure**: Adjusts the overall exposure of the image.
-   **Filmic RGB**: A modern tone mapping module to compress high dynamic range scenes into a display-ready image.
-   **Grain**: Simulates photographic film grain.
-   **Highpass**: A filter that can be used for sharpening.
-   **Invert**: Inverts the image colors.
-   **Lowlight**: Enhances details in the shadow areas.
-   **Sharpen**: Sharpens the image.
-   **Soften**: Applies a softening effect to the image.
-   **Tone Curve**: A classic module for adjusting image tonality using curves.
-   **Vignette**: Adds a vignette effect.
-   **Watermark**: Adds a watermark to the image.

## Directory Structure

```
darktable_python/
├── core/
│   └── pipeline.py         # Main processing pipeline
├── iop/
│   ├── bloom.py
│   ├── blurs.py
│   ├── borders.py
│   ├── ...                 # Individual Image Operation (IOP) modules
│   └── watermark.py
├── assets/
│   └── logo.png            # Assets used by modules (e.g., watermark)
├── output/                 # Default directory for processed images
├── utils/
│   └── watermark_generator.py
├── pipeline_config.yaml    # Configuration file for the pipeline
├── requirements.txt        # Project dependencies
└── README.md
```

## How to Run

1.  **Install Dependencies**

    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure the Pipeline**

    Modify `pipeline_config.yaml` to define the processing steps. You can specify the input file and the sequence of IOP modules to apply, along with their parameters.

    Example `pipeline_config.yaml`:
    ```yaml
    input_file: 'test_images/your_image.arw'
    output_file: 'pipeline_output.jpg'
    pipeline:
      - module: exposure
        enabled: true
        params:
          exposure: 0.5
      - module: filmicrgb
        enabled: true
        params:
          contrast: 1.2
          latitude: 0.1
          balance: 0.0
      - module: sharpen
        enabled: true
        params:
          amount: 1.0
          radius: 1.0
    ```

3.  **Run the Pipeline**

    Execute the main pipeline script:
    ```bash
    python core/pipeline.py
    ```
    The processed image will be saved in the `output/` directory. 