
import numpy as np
from scipy.ndimage import convolve
from skimage.feature import structure_tensor
import imageio.v2 as imageio
import os

class Diffuse():
    """
    Simulates anisotropic diffusion to create effects like bloom, blur, or sharpening.
    This implementation uses a diffusion tensor guided by the image's structure
    tensor to achieve true anisotropic behavior (e.g., diffusing along edges).
    """

    def __init__(self, iterations: int = 1, speed: float = 0.5,
                 anisotropy: float = 0.9, edge_sensitivity: float = 0.1):
        """
        Initializes the Diffuse operation.

        Args:
            iterations (int): Number of diffusion iterations. More iterations lead
                              to a stronger, more diffused effect.
            speed (float): The speed of diffusion at each iteration. Range [0, 1].
            anisotropy (float): Controls the direction of diffusion. 
                                > 0: Diffuses along edges (isophote).
                                < 0: Diffuses across edges (gradient).
                                = 0: Diffuses uniformly (isotropic).
                                A value near 1.0 is great for edge-preserving blur (bloom).
                                A value near -1.0 can be used for sharpening.
                                Range [-1, 1].
            edge_sensitivity (float): Threshold to detect edges. Lower values
                                      are more sensitive to faint edges.
        """
        self.iterations = iterations
        self.speed = np.clip(speed, 0, 1)
        self.anisotropy = np.clip(anisotropy, -1, 1)
        self.edge_sensitivity = edge_sensitivity

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the anisotropic diffusion process using a diffusion tensor.
        """
        diffused_image = np.copy(image)
        
        for channel in range(image.shape[2]):
            img_c = diffused_image[..., channel]

            # 1. Eigendecomposition of the structure tensor
            Axx, Axy, Ayy = structure_tensor(img_c, sigma=1, mode='nearest')
            
            # Manually construct the structure tensor S for each pixel
            S = np.zeros((image.shape[0], image.shape[1], 2, 2))
            S[..., 0, 0] = Axx
            S[..., 0, 1] = Axy
            S[..., 1, 0] = Axy
            S[..., 1, 1] = Ayy

            evals, vecs = np.linalg.eig(S)

            # Sort eigenvalues and eigenvectors in descending order for consistency
            order = np.argsort(-evals, axis=-1)
            l1 = np.take_along_axis(evals, order, axis=-1)[..., 0]
            v1 = np.take_along_axis(vecs, order[..., None, :], axis=-1)[..., :, 0] # Gradient dir，指向穿过边缘
            v2 = np.take_along_axis(vecs, order[..., None, :], axis=-1)[..., :, 1] # Isophote dir，指向沿着边缘

            # 2. Compute edge-preserving and anisotropic diffusion coefficients
            c = np.exp(-(l1 / (self.edge_sensitivity + 1e-8))**2)
            
            if self.anisotropy >= 0:
                d_para = c # Diffusion parallel to edge
                d_perp = c * (1 - self.anisotropy) # Diffusion perpendicular to edge
            else: # self.anisotropy < 0
                d_perp = c
                d_para = c * (1 + self.anisotropy)

            # 3. Construct the diffusion tensor D for each pixel
            v1_out_v1 = v1[..., :, None] * v1[..., None, :]
            v2_out_v2 = v2[..., :, None] * v2[..., None, :]
            D = d_perp[..., None, None] * v1_out_v1 + d_para[..., None, None] * v2_out_v2
            
            # 4. Iteratively apply the diffusion
            img_c_iter = img_c.copy()
            for _ in range(self.iterations):
                grad_y, grad_x = np.gradient(img_c_iter)
                grad_field = np.stack((grad_x, grad_y), axis=-1)
                
                # 5. Calculate flux using the diffusion tensor
                flux = np.einsum('...ij,...j->...i', D, grad_field)
                
                flux_x = flux[..., 0]
                flux_y = flux[..., 1]
                
                _, div_x = np.gradient(flux_x)
                div_y, _ = np.gradient(flux_y)
                divergence = div_x + div_y
                
                img_c_iter += self.speed * divergence

            diffused_image[..., channel] = img_c_iter

        return np.clip(diffused_image, 0, 1)

if __name__ == '__main__':
    print("--- Running Diffuse Module Local Test ---")

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    input_path = os.path.join(project_root, 'output', 'pipeline_with_soften_output.jpg')
    output_path = os.path.join(project_root, 'output', 'diffuse_test_output.jpg')

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please run the main pipeline first to generate an output file.")
    else:
        # Load the image
        print(f"Loading image from: {input_path}")
        image = imageio.imread(input_path)
        image_float = image.astype(np.float32) / 255.0

        # Instantiate the diffuser with parameters for a visible bloom effect
        print("Initializing Diffuse module...")
        diffuser = Diffuse(iterations=20, speed=0.5, anisotropy=0.9, edge_sensitivity=0.02)
        
        # Process the image
        print("Processing image with anisotropic diffusion...")
        processed_image = diffuser.process(image_float)

        # Save the output
        output_image_uint8 = (np.clip(processed_image, 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(output_path, output_image_uint8)
        print(f"--- Test complete. Image saved to: {output_path} ---") 