# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import rawpy
import imageio.v2 as imageio
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from iop.whitebalance_dt_port import WhitebalanceDtPort
from wb_ui.auto_wb import AutoWhiteBalance

class WhiteBalanceUI:
    def __init__(self, root):
        self.root = root
        self.root.title("White Balance Real-time Preview")
        self.root.geometry("1400x900")

        self.raw_image = None
        self.processed_image = None
        self.tk_image = None
        self.image_path = None
        self._updating_controls = False # Flag to prevent recursive updates

        # --- Layout ---
        top_frame = ttk.Frame(root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        image_frame = ttk.Frame(root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Top controls ---
        self.load_button = ttk.Button(control_frame, text="Load DNG Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.auto_button = ttk.Button(control_frame, text="Auto", command=self.run_auto_wb, state=tk.DISABLED)
        self.auto_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.save_button = ttk.Button(control_frame, text="Save JPEG", command=self.save_image, state=tk.DISABLED)
        self.save_button.grid(row=0, column=2, padx=15, pady=5)
        
        separator = ttk.Separator(control_frame, orient=tk.VERTICAL)
        separator.grid(row=0, column=3, sticky='ns', padx=10)

        # --- Mode Switcher ---
        self.wb_mode = tk.StringVar(value="temp_tint")
        
        mode_label = ttk.Label(control_frame, text="Mode:")
        mode_label.grid(row=0, column=4, padx=(10, 0))

        temp_mode_radio = ttk.Radiobutton(control_frame, text="Temperature/Tint", variable=self.wb_mode, value="temp_tint", command=self.switch_mode)
        temp_mode_radio.grid(row=0, column=5, padx=5)

        rgb_mode_radio = ttk.Radiobutton(control_frame, text="RGB Coefficients", variable=self.wb_mode, value="rgb", command=self.switch_mode)
        rgb_mode_radio.grid(row=0, column=6, padx=5)

        # --- Control Sliders Pane ---
        sliders_frame = ttk.Frame(top_frame, padding="10 0 0 0")
        sliders_frame.pack(side=tk.TOP, fill=tk.X, expand=True)

        # -- Temp/Tint controls --
        self.temp_tint_frame = ttk.Frame(sliders_frame)
        self.temp_tint_frame.pack(fill=tk.X, expand=True)
        
        self.temp_label = ttk.Label(self.temp_tint_frame, text="Temperature (K):")
        self.temp_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.temp_scale = ttk.Scale(self.temp_tint_frame, from_=1900, to=25000, orient=tk.HORIZONTAL, length=300, command=self.update_image)
        self.temp_scale.set(5000)
        self.temp_scale.pack(side=tk.LEFT, padx=5, pady=5)
        self.temp_value_label = ttk.Label(self.temp_tint_frame, text="5000K", width=6)
        self.temp_value_label.pack(side=tk.LEFT)

        self.tint_label = ttk.Label(self.temp_tint_frame, text="Tint:")
        self.tint_label.pack(side=tk.LEFT, padx=15, pady=5)
        self.tint_scale = ttk.Scale(self.temp_tint_frame, from_=0.135, to=2.326, orient=tk.HORIZONTAL, length=300, command=self.update_image)
        self.tint_scale.set(1.0)
        self.tint_scale.pack(side=tk.LEFT, padx=5, pady=5)
        self.tint_value_label = ttk.Label(self.temp_tint_frame, text="1.000", width=6)
        self.tint_value_label.pack(side=tk.LEFT)

        # -- RGB Coeffs controls --
        self.rgb_frame = ttk.Frame(sliders_frame)
        # self.rgb_frame.pack(fill=tk.X, expand=True) # Packed later by switch_mode
        
        self.r_label = ttk.Label(self.rgb_frame, text="R:")
        self.r_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.r_scale = ttk.Scale(self.rgb_frame, from_=0, to=8.0, orient=tk.HORIZONTAL, length=200, command=self.update_image)
        self.r_scale.set(1.0)
        self.r_scale.pack(side=tk.LEFT, padx=5, pady=5)
        self.r_value_label = ttk.Label(self.rgb_frame, text="1.000", width=6)
        self.r_value_label.pack(side=tk.LEFT)

        self.g_label = ttk.Label(self.rgb_frame, text="G:")
        self.g_label.pack(side=tk.LEFT, padx=15, pady=5)
        self.g_scale = ttk.Scale(self.rgb_frame, from_=0, to=8.0, orient=tk.HORIZONTAL, length=200, command=self.update_image)
        self.g_scale.set(1.0)
        self.g_scale.pack(side=tk.LEFT, padx=5, pady=5)
        self.g_value_label = ttk.Label(self.rgb_frame, text="1.000", width=6)
        self.g_value_label.pack(side=tk.LEFT)

        self.b_label = ttk.Label(self.rgb_frame, text="B:")
        self.b_label.pack(side=tk.LEFT, padx=15, pady=5)
        self.b_scale = ttk.Scale(self.rgb_frame, from_=0, to=8.0, orient=tk.HORIZONTAL, length=200, command=self.update_image)
        self.b_scale.set(1.0)
        self.b_scale.pack(side=tk.LEFT, padx=5, pady=5)
        self.b_value_label = ttk.Label(self.rgb_frame, text="1.000", width=6)
        self.b_value_label.pack(side=tk.LEFT)

        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.switch_mode() # Initial setup

    def switch_mode(self):
        self._updating_controls = True # Prevent updates while switching
        mode = self.wb_mode.get()
        if mode == "temp_tint":
            self.rgb_frame.pack_forget()
            self.temp_tint_frame.pack(fill=tk.X, expand=True)
            self._sync_temp_tint_from_rgb()
        else: # "rgb"
            self.temp_tint_frame.pack_forget()
            self.rgb_frame.pack(fill=tk.X, expand=True)
            self._sync_rgb_from_temp_tint()
        self._updating_controls = False
        self.update_image()

    def _sync_rgb_from_temp_tint(self):
        temp = self.temp_scale.get()
        tint = self.tint_scale.get()
        wb_module = WhitebalanceDtPort(temp_k=temp, gui_tint=tint)
        coeffs = wb_module.coeffs

        self.r_scale.set(coeffs[0])
        self.g_scale.set(coeffs[1])
        self.b_scale.set(coeffs[2])

        self.r_value_label.config(text=f"{coeffs[0]:.3f}")
        self.g_value_label.config(text=f"{coeffs[1]:.3f}")
        self.b_value_label.config(text=f"{coeffs[2]:.3f}")
    
    def _sync_temp_tint_from_rgb(self):
        r, g, b = self.r_scale.get(), self.g_scale.get(), self.b_scale.get()
        if g < 1e-6: g = 1e-6 # Avoid division by zero
        
        # Normalize coeffs to green channel for the solver
        target_coeffs = np.array([r / g, 1.0, b / g])
        
        result = AutoWhiteBalance.find_temp_tint_for_coeffs(target_coeffs)
        if result:
            temp, tint = result
            self.temp_scale.set(temp)
            self.tint_scale.set(tint)
            self.temp_value_label.config(text=f"{int(temp)}K")
            self.tint_value_label.config(text=f"{tint:.3f}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            initialdir="../database",
            title="Select a DNG file",
            filetypes=(("DNG files", "*.dng"), ("All files", "*.*"))
        )
        if not file_path:
            return

        self.image_path = file_path
        try:
            with rawpy.imread(file_path) as raw:
                # Demosaic without using rawpy's auto WB. Gamma and brightness correction for better viewing.
                self.raw_image = raw.postprocess(use_camera_wb=False, no_auto_bright=True, output_bps=16, gamma=(1, 1))
            self.update_image()
            self.save_button.config(state=tk.NORMAL)
            self.auto_button.config(state=tk.NORMAL)
        except Exception as e:
            print(f"Error loading image: {e}")
            # Fallback for any error, including potential thumbnail issues
            try:
                with rawpy.imread(file_path) as raw:
                    self.raw_image = raw.postprocess(use_camera_wb=False, no_auto_bright=True, output_bps=16, gamma=(1,1))
                self.update_image()
                self.save_button.config(state=tk.NORMAL)
                self.auto_button.config(state=tk.NORMAL)
            except Exception as e2:
                print(f"Fallback processing also failed: {e2}")

    def run_auto_wb(self):
        if self.raw_image is None:
            return

        # Normalize the raw image to float [0,1] for analysis
        img_float = self.raw_image.astype(np.float32) / 65535.0
        
        result = AutoWhiteBalance.analyze(img_float)
        
        if result:
            temp, tint = result
            self.wb_mode.set("temp_tint") # Switch to temp/tint mode to display results
            self._updating_controls = True
            self.rgb_frame.pack_forget()
            self.temp_tint_frame.pack(fill=tk.X, expand=True)
            self.temp_scale.set(temp)
            self.tint_scale.set(tint)
            self._updating_controls = False
            self.update_image() # This will refresh labels and preview

    def update_image(self, _=None):
        if self.raw_image is None or self._updating_controls:
            return

        mode = self.wb_mode.get()
        coeffs = np.array([1.0, 1.0, 1.0])

        if mode == "temp_tint":
            temp = self.temp_scale.get()
            tint = self.tint_scale.get()
            self.temp_value_label.config(text=f"{int(temp)}K")
            self.tint_value_label.config(text=f"{tint:.3f}")
            wb_module = WhitebalanceDtPort(temp_k=temp, gui_tint=tint)
            coeffs = wb_module.coeffs
        else: # "rgb"
            r = self.r_scale.get()
            g = self.g_scale.get()
            b = self.b_scale.get()
            self.r_value_label.config(text=f"{r:.3f}")
            self.g_value_label.config(text=f"{g:.3f}")
            self.b_value_label.config(text=f"{b:.3f}")
            coeffs = np.array([r, g, b])

        # Normalize the raw image to float [0,1] for processing
        img_float = self.raw_image.astype(np.float32) / 65535.0
        
        # Apply white balance
        self.processed_image = np.clip(img_float * coeffs.reshape(1, 1, 3), 0.0, 1.0)

        # Convert to 8-bit for display
        img_8bit = (self.processed_image * 255).astype(np.uint8)
        
        # Display the image
        pil_img = Image.fromarray(img_8bit)
        
        # Resize for display
        max_size = (1200, 800)
        pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.tk_image)

    def save_image(self):
        if self.processed_image is None or self.image_path is None:
            return
        
        base_name = os.path.basename(self.image_path)
        name, _ = os.path.splitext(base_name)
        
        mode = self.wb_mode.get()
        if mode == "temp_tint":
            temp = int(self.temp_scale.get())
            tint = self.tint_scale.get()
            output_filename = f"{name}_wb_{temp}K_{tint:.3f}.jpg"
        else:
            r, g, b = self.r_scale.get(), self.g_scale.get(), self.b_scale.get()
            output_filename = f"{name}_wb_R{r:.2f}_G{g:.2f}_B{b:.2f}.jpg"

        output_path = os.path.join(os.path.dirname(__file__), 'output', output_filename)
        
        try:
            img_to_save = (self.processed_image * 255).astype(np.uint8)
            imageio.imwrite(output_path, img_to_save)
            print(f"Image saved to {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")


if __name__ == "__main__":
    app_root = tk.Tk()
    ui = WhiteBalanceUI(app_root)
    app_root.mainloop() 