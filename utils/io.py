import os
import imageio
import numpy as np

def load_input(input_path):
    """Load input image or volume"""
    if input_path.lower().endswith(('.tiff', '.tif', '.nii', '.nii.gz', '.npy')):
        return imageio.volread(input_path), True  # Volume
    else:
        return imageio.imread(input_path), False  # Image

def save_output(output_data, output_path, is_volume=False):
    """Save the output image or volume"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if is_volume:
        imageio.volwrite(output_path, output_data.numpy())
    else:
        output_np = output_data.squeeze().permute(1, 2, 0).numpy()
        if output_np.max() <= 1.0:
            output_np = (output_np * 255).astype(np.uint8)
        imageio.imwrite(output_path, output_np)