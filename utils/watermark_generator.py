import os
from PIL import Image, ImageDraw, ImageFont

# --- HOW TO USE THIS SCRIPT ---
# This script generates a text-based watermark image and saves it to the 'assets' directory.
# To use it, simply configure the parameters in the `main()` function below and run the script
# from your terminal:
#
# python darktable_python/utils/watermark_generator.py
#
# --- CONFIGURATION ---
# 1. TEXT_TO_RENDER: The text you want in your watermark.
# 2. FONT_PATH: The full path to a .ttf or .otf font file on your system.
#    - On macOS, you can find fonts in /System/Library/Fonts/ or ~/Library/Fonts/.
#    - On Linux, common paths are /usr/share/fonts/ or ~/.fonts/.
# 3. TEXT_COLOR: The color of the text (e.g., 'white', 'black', '#FF0000', 'rgba(255, 255, 255, 128)').
# 4. OUTPUT_FILENAME: The name of the output PNG file. It will be saved in 'darktable_python/assets/'.
# 5. PADDING: The space in pixels around the text.
# -----------------------------

def generate_watermark(text, font_path, color, output_path, padding=20):
    """
    Generates a watermark image from text.

    Args:
        text (str): The text content for the watermark.
        font_path (str): The path to a .ttf or .otf font file.
        color (str): The color of the text (e.g., 'white', 'black', '#FF0000').
        output_path (str): The full path to save the generated PNG image.
        padding (int): The padding around the text.
    """
    try:
        # Try to load the font, starting with a size of 40.
        font = ImageFont.truetype(font_path, 40)
    except IOError:
        print(f"Error: Font file not found at '{font_path}'.")
        print("Please provide a valid path to a .ttf or .otf font file.")
        print("On macOS, you can find fonts in /System/Library/Fonts/ or ~/Library/Fonts/.")
        print("On Linux, common paths are /usr/share/fonts/ or ~/.fonts/.")
        return

    # Create a dummy image to calculate text size
    dummy_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
    
    # Get the bounding box of the text
    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Create the actual image with padding
    image_width = int(text_width + 2 * padding)
    image_height = int(text_height + 2 * padding)
    
    # Create a transparent background image (RGBA with alpha=0)
    image = Image.new('RGBA', (image_width, image_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # Draw the text onto the image
    draw.text((padding, padding), text, font=font, fill=color)

    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, 'PNG')
    print(f"Success! Watermark saved to '{output_path}'")

def main():
    """
    Main function to configure and generate the watermark.
    
    <<< EDIT THE VALUES BELOW TO CUSTOMIZE YOUR WATERMARK >>>
    """
    # --- 1. Set Watermark Text ---
    TEXT_TO_RENDER = "@Hour"

    # --- 2. Set Font Path ---
    # Example for macOS: "/System/Library/Fonts/Supplemental/Arial.ttf"
    # Example for Chinese font on macOS: "/System/Library/Fonts/STHeitiLight.ttc"
    # Example for Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_PATH = "/System/Library/Fonts/MarkerFelt.ttc" # <-- IMPORTANT: CHANGE THIS PATH IF NEEDED

    # --- 3. Set Text Color ---
    TEXT_COLOR = "rgba(255, 255, 255, 150)" # White with ~60% opacity

    # --- 4. Set Output Filename ---
    OUTPUT_FILENAME = "Hour_watermark.png"
    
    # --- 5. Set Padding ---
    PADDING = 15

    # --- Do not edit below this line ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'assets', OUTPUT_FILENAME)

    generate_watermark(
        text=TEXT_TO_RENDER,
        font_path=FONT_PATH,
        color=TEXT_COLOR,
        output_path=output_path,
        padding=PADDING
    )

if __name__ == '__main__':
    main() 