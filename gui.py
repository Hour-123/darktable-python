import customtkinter as ctk
import yaml
import os
import subprocess
from PIL import Image
from tkinter import filedialog

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Darktable Python UI")
        self.geometry("1200x800")

        # Top level layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Paned window for resizable panes
        self.paned_window = ctk.CTkFrame(self, fg_color="transparent")
        self.paned_window.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10,0))
        self.paned_window.grid_columnconfigure(2, weight=1) # Right frame will expand to fill space
        self.paned_window.grid_rowconfigure(0, weight=1)

        # Left Frame for controls
        self.controls_frame = ctk.CTkScrollableFrame(self.paned_window, label_text="IOP Modules", width=350)
        self.controls_frame.grid(row=0, column=0, sticky="nsew")

        # --- Bind scroll events for macOS Trackpad ---
        self.controls_frame.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        
        # Separator
        self.separator = ctk.CTkFrame(self.paned_window, width=7, cursor="sb_h_double_arrow", fg_color=("gray80", "gray20"))
        self.separator.grid(row=0, column=1, sticky="ns")
        self.separator.bind("<ButtonPress-1>", self.on_separator_press)
        self.separator.bind("<B1-Motion>", self.on_separator_drag)

        # Right Frame for image display
        self.image_frame = ctk.CTkFrame(self.paned_window)
        self.image_frame.grid(row=0, column=2, sticky="nsew")
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="Output image will be displayed here.")
        self.image_label.pack(padx=10, pady=10, expand=True, fill="both")

        # Bottom Frame for button and logs
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # --- Controls inside bottom frame ---
        self.bottom_controls_frame = ctk.CTkFrame(self.bottom_frame)
        self.bottom_controls_frame.pack(fill="x", expand=True, padx=0, pady=0)
        self.bottom_controls_frame.grid_columnconfigure(2, weight=1)

        self.run_button = ctk.CTkButton(self.bottom_controls_frame, text="Run Pipeline", command=self.run_pipeline)
        self.run_button.grid(row=0, column=0, padx=10, pady=10)

        ctk.CTkLabel(self.bottom_controls_frame, text="Output Filename:").grid(row=0, column=1, padx=(10,0), pady=10, sticky="e")
        
        self.output_file_entry = ctk.CTkEntry(self.bottom_controls_frame)
        self.output_file_entry.grid(row=0, column=2, padx=(0,10), pady=10, sticky="ew")

        self.log_textbox = ctk.CTkTextbox(self.bottom_frame, height=100)
        self.log_textbox.pack(fill="x", expand=True, padx=10, pady=(0,10))

        # Load config and create controls
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(script_dir, 'pipeline_config.yaml')
        self.load_config()
        self.create_input_file_controls() # Create these before other controls
        self.create_controls()

    def create_input_file_controls(self):
        """Creates the widgets for input file selection."""
        input_frame = ctk.CTkFrame(self.controls_frame)
        input_frame.pack(fill="x", expand=True, padx=5, pady=(0, 10))
        input_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(input_frame, text="Input File").grid(row=0, column=0, columnspan=2, padx=5, pady=(5,0), sticky="w")
        
        self.input_file_entry = ctk.CTkEntry(input_frame, state="disabled")
        self.input_file_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        browse_button = ctk.CTkButton(input_frame, text="Browse...", width=80, command=self.select_input_file)
        browse_button.grid(row=1, column=1, padx=5, pady=5)
    
    def select_input_file(self):
        """Opens a file dialog to select a new input image."""
        project_root = os.path.dirname(os.path.abspath(self.config_path))
        filepath = filedialog.askopenfilename(
            title="Select an Image File",
            initialdir=project_root,
            filetypes=(("Image Files", "*.arw *.cr2 *.nef *.dng *.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*"))
        )
        if filepath:
            # Convert to relative path to keep config portable
            relative_path = os.path.relpath(filepath, project_root)
            self.config['input_file'] = relative_path
            
            # Update the UI
            self.input_file_entry.configure(state="normal")
            self.input_file_entry.delete(0, "end")
            self.input_file_entry.insert(0, relative_path)
            self.input_file_entry.configure(state="disabled")

    def on_separator_press(self, event):
        self._drag_start_x = event.x

    def on_separator_drag(self, event):
        delta_x = event.x - self._drag_start_x
        new_width = self.controls_frame.winfo_width() + delta_x
        if new_width > 200: # Set a minimum width for the controls pane
            self.controls_frame.configure(width=new_width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling, essential for trackpad support on macOS."""
        # This is a workaround to get to the underlying canvas.
        self.controls_frame._parent_canvas.yview_scroll(int(-1*(event.delta)), "units")

    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # After loading, if the UI is ready, update the input file entry
        if hasattr(self, 'input_file_entry'):
            self.input_file_entry.configure(state="normal")
            self.input_file_entry.delete(0, "end")
            self.input_file_entry.insert(0, self.config.get('input_file', ''))
            self.input_file_entry.configure(state="disabled")
        
        # Also update the output file entry
        if hasattr(self, 'output_file_entry'):
            self.output_file_entry.delete(0, "end")
            self.output_file_entry.insert(0, self.config.get('output_file', 'output.jpg'))

    def create_controls(self):
        # Store references to the widgets
        self.control_widgets = {}

        for i, step in enumerate(self.config.get('pipeline', [])):
            module_name = step.get('module')
            if not module_name:
                continue

            module_frame = ctk.CTkFrame(self.controls_frame, border_width=1)
            module_frame.pack(fill="x", expand=True, padx=5, pady=5)

            # --- Header: Checkbox and Title ---
            header_frame = ctk.CTkFrame(module_frame)
            header_frame.pack(fill="x", expand=True)

            enabled_var = ctk.BooleanVar(value=step.get('enabled', True))
            
            checkbox = ctk.CTkCheckBox(header_frame, text=f"{i+1}. {module_name}", variable=enabled_var)
            checkbox.pack(side="left", padx=5, pady=5)
            
            self.control_widgets[module_name] = {'enabled': enabled_var, 'params': {}}

            # --- Parameters ---
            params = step.get('params', {})
            for param_name, param_value in params.items():
                param_frame = ctk.CTkFrame(module_frame)
                param_frame.pack(fill="x", expand=True, padx=15, pady=2)
                
                label = ctk.CTkLabel(param_frame, text=param_name, width=20)
                label.pack(side="left", padx=5)

                if isinstance(param_value, (int, float)):
                    # --- Hybrid Slider/Entry Widget ---
                    hybrid_frame = ctk.CTkFrame(param_frame)
                    hybrid_frame.pack(side="left", fill="x", expand=True, padx=5)
                    hybrid_frame.grid_columnconfigure(0, weight=1) # Slider
                    hybrid_frame.grid_columnconfigure(1, weight=0) # Entry
                    hybrid_frame.grid_columnconfigure(2, weight=0) # Label

                    # Assuming a range for sliders, this needs to be improved later
                    # A better way would be to have min/max in yaml config per-parameter
                    min_val, max_val = (0, 2)
                    if 'exposure' in param_name:
                        min_val, max_val = (-2, 2)
                    if 'contrast' in param_name:
                        min_val, max_val = (0, 2)
                    if 'radius' in param_name or 'amount' in param_name:
                        min_val, max_val = (0, 5)

                    var = ctk.DoubleVar(value=param_value)
                    
                    # Entry for direct input
                    entry = ctk.CTkEntry(hybrid_frame, width=100, textvariable=var)
                    entry.grid(row=0, column=1, padx=(5,0))
                    
                    # Slider
                    slider = ctk.CTkSlider(hybrid_frame, from_=min_val, to=max_val, variable=var)
                    slider.grid(row=0, column=0, sticky="ew")

                    self.control_widgets[module_name]['params'][param_name] = var
                else: # string, boolean, etc.
                    entry_var = ctk.StringVar(value=str(param_value))
                    entry = ctk.CTkEntry(param_frame, textvariable=entry_var)
                    entry.pack(side="left", fill="x", expand=True, padx=5)
                    self.control_widgets[module_name]['params'][param_name] = entry_var

    def save_config(self):
        """Saves the current UI control states back to the self.config object."""
        for i, step in enumerate(self.config['pipeline']):
            module_name = step['module']
            if module_name in self.control_widgets:
                # Update enabled status
                step['enabled'] = self.control_widgets[module_name]['enabled'].get()
                # Update params
                for param_name, var in self.control_widgets[module_name]['params'].items():
                    step['params'][param_name] = var.get()
        
        # Update output filename from the entry widget
        self.config['output_file'] = self.output_file_entry.get()

        # Write the updated config back to the YAML file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def display_image(self):
        """Loads and displays the output image."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'output', self.config['output_file'])
        if os.path.exists(output_path):
            try:
                # Let the image_frame calculate its size
                self.image_frame.update_idletasks()

                # Open the image
                image = Image.open(output_path)
                original_width, original_height = image.size

                # Get the frame size
                frame_width = self.image_frame.winfo_width()
                frame_height = self.image_frame.winfo_height()

                if frame_width == 1 or frame_height == 1:
                    print("Warning: Frame size is not yet determined.")
                    return

                # Calculate the new size to fit the frame while maintaining aspect ratio
                aspect_ratio = original_width / original_height
                new_width = frame_width
                new_height = int(new_width / aspect_ratio)

                if new_height > frame_height:
                    new_height = frame_height
                    new_width = int(new_height * aspect_ratio)
                
                # Resize the image
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create a CTkImage
                ctk_image = ctk.CTkImage(light_image=resized_image, size=(new_width, new_height))

                # Adjust the label size to the image size
                self.image_label.configure(image=ctk_image, text="")
                self._image_ref = ctk_image # Keep a reference

            except Exception as e:
                self.log_textbox.insert("end", f"Error displaying image: {e}\n")
        else:
            self.image_label.configure(text=f"Output file not found:\n{output_path}")

    def run_pipeline(self):
        # 1. Save the current UI state to the YAML file
        self.save_config()
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.insert("end", f"Configuration saved to {self.config_path}\n")
        self.log_textbox.insert("end", "Running pipeline...\n")
        self.update_idletasks() # Force UI update

        # 2. Run the pipeline script as a subprocess
        try:
            # We need to add the project root to the python path for imports to work
            project_root = os.path.dirname(os.path.abspath(__file__))
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root

            process = subprocess.Popen(
                ['python', 'core/pipeline.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                env=env,
                cwd=project_root
            )

            # 3. Stream output to the log textbox
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    self.log_textbox.insert("end", line)
                    self.log_textbox.see("end")
                    self.update_idletasks()

                process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                self.log_textbox.insert("end", "\n--- Pipeline finished successfully! ---\n")
                # 4. Display the output image
                self.display_image()
            else:
                self.log_textbox.insert("end", f"\n--- Pipeline failed with exit code {return_code} ---\n")

        except Exception as e:
            self.log_textbox.insert("end", f"\n--- An error occurred while running the pipeline ---\n{e}\n")

if __name__ == "__main__":
    app = App()
    app.mainloop() 