import numpy as np
import torch
from tqdm import tqdm

def process_image(model, image, device='cuda', use_patches=False, patch_size=64, overlap=16):
    """Process a single image with the model"""
    # Convert image to tensor and normalize
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if len(image.shape) == 2:  # Single channel
            image = np.expand_dims(image, axis=(0, 1))
        elif len(image.shape) == 3 and image.shape[2] in [1, 3]:  # RGB or grayscale
            image = np.transpose(image, (2, 0, 1))[np.newaxis, :]
        image_tensor = torch.from_numpy(image).float().to(device)
    else:
        image_tensor = image.to(device)

    # Use patches for large images
    if use_patches and (image_tensor.shape[2] > patch_size or image_tensor.shape[3] > patch_size):
        return process_large_image(model, image_tensor, patch_size, overlap, device)
    else:
        with torch.no_grad():
            return model(image_tensor).cpu()

def process_large_image(model, image, patch_size=64, overlap=16, device='cuda'):
    """Process a large image by dividing it into patches"""
    model.eval()
    with torch.no_grad():
        _, _, h, w = image.shape
        scale = model.scale
        output_h, output_w = h * scale, w * scale
        output = torch.zeros((1, 3, output_h, output_w), device='cpu')
        weights = torch.zeros((1, 3, output_h, output_w), device='cpu')
        stride = patch_size - overlap

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = image[:, :, i:i+patch_size, j:j+patch_size].to(device)
                patch_output = model(patch).cpu()
                out_i, out_j = i * scale, j * scale
                out_h, out_w = patch_size * scale, patch_size * scale

                weight = torch.ones_like(patch_output)
                if overlap > 0:
                    for c in range(3):
                        for ii in range(overlap * scale):
                            for jj in range(out_w):
                                weight[0, c, ii, jj] *= ii / (overlap * scale)
                                weight[0, c, out_h - 1 - ii, jj] *= ii / (overlap * scale)
                        for ii in range(out_h):
                            for jj in range(overlap * scale):
                                weight[0, c, ii, jj] *= jj / (overlap * scale)
                                weight[0, c, ii, out_w - 1 - jj] *= jj / (overlap * scale)

                output[:, :, out_i:out_i+out_h, out_j:out_j+out_w] += patch_output * weight
                weights[:, :, out_i:out_i+out_h, out_j:out_j+out_w] += weight

        return output / (weights + 1e-8)

def process_volume(model, volume, device='cuda', use_patches=False, patch_size=64, overlap=16):
    """Process a 3D volume with the model"""
    if isinstance(volume, np.ndarray):
        if volume.dtype == np.uint8:
            volume = volume.astype(np.float32) / 255.0
        depth, height, width = volume.shape
        output = np.zeros((depth, height * model.scale, width * model.scale, 3), dtype=np.float32)
        for z in tqdm(range(depth), desc="Processing volume slices"):
            slice_data = volume[z]
            slice_tensor = torch.from_numpy(slice_data).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            slice_output = process_image(
                model, slice_tensor, device=device, use_patches=use_patches,
                patch_size=patch_size, overlap=overlap
            )
            output[z] = slice_output.squeeze().permute(1, 2, 0).numpy()
        return torch.from_numpy(output.transpose(3, 0, 1, 2))  # CDHW format