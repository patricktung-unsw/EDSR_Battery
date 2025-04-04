import argparse
import os
import torch
import numpy as np
import imageio
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from local modules
from models.model_factory import get_model
from utils.metrics import psnr
from datasets.patch_sampler import PatchSampler

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with super-resolution models for 2D and 3D data')
    # Add your existing arguments
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image/volume or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to save output images/volumes')
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size for large images/volumes')
    parser.add_argument('--overlap', type=int, default=16, help='Overlap between patches')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--use_patches', action='store_true', help='Process large images/volumes in patches')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')

    # Add volume-specific arguments
    parser.add_argument('--input_type', type=str, default='auto', choices=['auto', 'image', 'volume'],
                        help='Type of input data (auto, image, volume)')
    parser.add_argument('--visualize_slice', type=int, default=None,
                        help='Slice index to visualize (default is middle slice)')
    return parser.parse_args()

def process_image(model, image, device='cuda', use_patches=False, patch_size=64, overlap=16):
    """Process a single image with the model"""
    # Convert image to tensor and normalize
    if isinstance(image, np.ndarray):
        # Handle different input types
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Add batch and channel dimensions if needed
        if len(image.shape) == 2:  # Single channel
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            image = np.expand_dims(image, axis=0)  # Add channel dimension
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Already has channel dimension
            image = np.transpose(image, (2, 0, 1))  # CHW format
            image = np.expand_dims(image, axis=0)  # Add batch dimension
        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
            image = np.transpose(image, (2, 0, 1))  # CHW format
            image = np.expand_dims(image, axis=0)  # Add batch dimension

        image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image

    # Move to device
    image_tensor = image_tensor.to(device)

    # Use patches for large images to avoid CUDA out of memory
    if use_patches and (image_tensor.shape[2] > patch_size or image_tensor.shape[3] > patch_size):
        return process_large_image(model, image_tensor, patch_size, overlap, device)
    else:
        # Process the entire image at once
        with torch.no_grad():
            output = model(image_tensor).cpu()
        return output

def process_large_image(model, image, patch_size=64, overlap=16, device='cuda'):
    """Process a large image by dividing it into patches"""
    model.eval()
    with torch.no_grad():
        # Get dimensions of input image
        _, _, h, w = image.shape

        # Calculate output dimensions (after upscaling)
        scale = model.scale
        output_h = h * scale
        output_w = w * scale
        output = torch.zeros((1, 3, output_h, output_w), device='cpu')

        # Create a weight map for blending overlapping regions
        weights = torch.zeros((1, 3, output_h, output_w), device='cpu')

        # Calculate stride (with overlap)
        stride = patch_size - overlap

        # Process each patch
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # Extract patch
                patch = image[:, :, i:i+patch_size, j:j+patch_size].to(device)

                # Process patch
                patch_output = model(patch).cpu()

                # Calculate output position
                out_i = i * scale
                out_j = j * scale
                out_h = patch_size * scale
                out_w = patch_size * scale

                # Create a weight map for blending (higher weight in center, lower at edges)
                weight = torch.ones_like(patch_output)
                if overlap > 0:
                    # Apply linear ramp for blending at edges
                    for c in range(3):
                        for ii in range(overlap * scale):
                            for jj in range(out_w):
                                weight[0, c, ii, jj] *= ii / (overlap * scale)
                                weight[0, c, out_h - 1 - ii, jj] *= ii / (overlap * scale)
                        for ii in range(out_h):
                            for jj in range(overlap * scale):
                                weight[0, c, ii, jj] *= jj / (overlap * scale)
                                weight[0, c, ii, out_w - 1 - jj] *= jj / (overlap * scale)

                # Add to output with blending
                output[:, :, out_i:out_i+out_h, out_j:out_j+out_w] += patch_output * weight
                weights[:, :, out_i:out_i+out_h, out_j:out_j+out_w] += weight

        # Average overlapping regions
        output = output / (weights + 1e-8)  # Add small epsilon to avoid division by zero

    return output

def process_volume(model, volume, device='cuda', use_patches=False, patch_size=64, overlap=16):
    """Process a 3D volume with the model"""
    if isinstance(volume, np.ndarray):
        # Handle different input types
        if volume.dtype == np.uint8:
            volume = volume.astype(np.float32) / 255.0

        # Get dimensions of volume
        if len(volume.shape) == 3:  # (depth, height, width)
            depth, height, width = volume.shape
            # Process slice by slice
            output = np.zeros((depth * model.scale, height * model.scale, width * model.scale, 3), dtype=np.float32)

            for z in tqdm(range(depth), desc="Processing volume slices"):
                # Extract slice, add channel and batch dimensions
                slice_data = volume[z]
                slice_tensor = torch.from_numpy(slice_data).float().unsqueeze(0).unsqueeze(0)
                # Repeat to create 3 channels (assuming grayscale input)
                slice_tensor = slice_tensor.repeat(1, 3, 1, 1)

                # Process slice with the model
                slice_output = process_image(
                    model,
                    slice_tensor,
                    device=device,
                    use_patches=use_patches,
                    patch_size=patch_size,
                    overlap=overlap
                )

                # Convert output back to numpy and store in the output volume
                slice_result = slice_output.squeeze().transpose(1, 2, 0).numpy()
                output[z * model.scale:(z+1) * model.scale] = slice_result

            return torch.from_numpy(output.transpose(3, 0, 1, 2))  # Convert to CDHW format
        else:
            # Handle as regular 2D image
            return process_image(model, volume, device, use_patches, patch_size, overlap)
    else:
        # Assume it's already a tensor
        return process_image(model, volume, device, use_patches, patch_size, overlap)

def visualize_result(input_img, output_img, target_img=None):
    """Visualize the input and output images"""
    if target_img is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(input_img.squeeze().transpose(1, 2, 0))
        axes[0].set_title('Input (Low Resolution)')
        axes[0].axis('off')

        axes[1].imshow(output_img.squeeze().transpose(1, 2, 0))
        axes[1].set_title('Output (Super Resolution)')
        axes[1].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(input_img.squeeze().transpose(1, 2, 0))
        axes[0].set_title('Input (Low Resolution)')
        axes[0].axis('off')

        axes[1].imshow(output_img.squeeze().transpose(1, 2, 0))
        axes[1].set_title('Output (Super Resolution)')
        axes[1].axis('off')

        axes[2].imshow(target_img.squeeze().transpose(1, 2, 0))
        axes[2].set_title('Target (High Resolution)')
        axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_volume_result(input_vol, output_vol, slice_idx=None, target_vol=None):
    """Visualize slices from 3D volumes"""
    # If no slice index provided, show the middle slice
    if slice_idx is None:
        slice_idx = input_vol.shape[1] // 2  # Assuming CDHW format

    # Extract slices
    input_slice = input_vol[:, slice_idx].unsqueeze(1)
    output_slice = output_vol[:, slice_idx * output_vol.shape[1] // input_vol.shape[1]].unsqueeze(1)

    # Use the existing 2D visualization function for slices
    if target_vol is None:
        visualize_result(input_slice, output_slice)
    else:
        target_slice = target_vol[:, slice_idx * target_vol.shape[1] // input_vol.shape[1]].unsqueeze(1)
        visualize_result(input_slice, output_slice, target_slice)

def save_output(output_img, output_path):
    """Save the output image"""
    # Convert to numpy array and scale to 0-255
    output_np = output_img.squeeze().transpose(1, 2, 0).numpy()

    # Handle single channel images
    if output_np.shape[2] == 1:
        output_np = output_np.squeeze()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # For volumetric data, handle differently
    if len(output_np.shape) > 3:
        # Save as TIFF stack
        imageio.volwrite(output_path, output_np)
    else:
        # Save as image
        if output_np.max() <= 1.0:
            output_np = (output_np * 255).astype(np.uint8)
        imageio.imwrite(output_path, output_np)

def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Add file type detection
    parser.add_argument('--input_type', type=str, default='auto', choices=['auto', 'image', 'volume'],
                        help='Type of input data (auto, image, volume)')

    # Load configuration
    config = {
        'scale': args.scale,
        'num_channels': 3,
        'filters': 256,
        'n_resblock': 32,
        'res_scaling': 0.1
    }

    if args.config:
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))

    # Create model and load checkpoint
    model = get_model('EDSR_2D', config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Process input (file or directory)
    if os.path.isfile(args.input):
        # Process a single file
        input_path = args.input
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output, filename)

        # Auto-detect or use specified input type
        is_volume = False
        if args.input_type == 'auto':
            # Check file extension for common volume formats
            if input_path.lower().endswith(('.tiff', '.tif', '.nii', '.nii.gz', '.npy')):
                # Try to load as a volume
                try:
                    input_data = imageio.volread(input_path)
                    is_volume = len(input_data.shape) == 3
                except:
                    is_volume = False
        else:
            is_volume = args.input_type == 'volume'

        if is_volume:
            # Read input volume
            input_vol = imageio.volread(input_path)
            print(f"Processing volume with shape: {input_vol.shape}")

            # Process volume
            output_vol = process_volume(
                model,
                input_vol,
                device=device,
                use_patches=args.use_patches,
                patch_size=args.patch_size,
                overlap=args.overlap
            )

            # Save output
            save_output(output_vol, output_path)

            # Visualize if requested
            if args.visualize:
                input_tensor = torch.from_numpy(input_vol.astype(np.float32) / 255.0)
                # Add channel dimension if needed
                if len(input_tensor.shape) == 3:  # DHW format
                    input_tensor = input_tensor.unsqueeze(0)  # CDHW format

                visualize_volume_result(input_tensor, output_vol)

        else:
            # Process a directory
            input_dir = args.input
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)

            # Get all image files in the directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(ext)])

            # Process each image
            for input_path in tqdm(image_files, desc="Processing images"):
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, filename)

                # Read input image
                input_img = imageio.imread(input_path)

                # Process image
                output_img = process_image(
                    model,
                    input_img,
                    device=device,
                    use_patches=args.use_patches,
                    patch_size=args.patch_size,
                    overlap=args.overlap
                )

                # Save output
                save_output(output_img, output_path)

                # Visualize if requested (only the first image)
                if args.visualize and image_files.index(input_path) == 0:
                    input_tensor = torch.from_numpy(input_img).float() / 255.0
                    # Reshape if needed
                    if len(input_tensor.shape) == 2:  # Single channel
                        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                    elif len(input_tensor.shape) == 3 and input_tensor.shape[2] in [1, 3]:  # Has channel dim
                        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

                    visualize_result(input_tensor, output_img)

    print(f"Inference completed. Results saved to {args.output}")

if __name__ == '__main__':
    main()