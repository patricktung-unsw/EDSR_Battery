import matplotlib.pyplot as plt

def visualize_result(input_img, output_img, target_img=None):
    """Visualize the input and output images"""
    fig, axes = plt.subplots(1, 3 if target_img is not None else 2, figsize=(18, 6))
    axes[0].imshow(input_img.squeeze().transpose(1, 2, 0))
    axes[0].set_title('Input (Low Resolution)')
    axes[0].axis('off')

    axes[1].imshow(output_img.squeeze().transpose(1, 2, 0))
    axes[1].set_title('Output (Super Resolution)')
    axes[1].axis('off')

    if target_img is not None:
        axes[2].imshow(target_img.squeeze().transpose(1, 2, 0))
        axes[2].set_title('Target (High Resolution)')
        axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_volume_result(input_vol, output_vol, slice_idx=None, target_vol=None):
    """Visualize slices from 3D volumes"""
    if slice_idx is None:
        slice_idx = input_vol.shape[1] // 2  # Middle slice
    input_slice = input_vol[:, slice_idx].unsqueeze(1)
    output_slice = output_vol[:, slice_idx * output_vol.shape[1] // input_vol.shape[1]].unsqueeze(1)
    if target_vol is not None:
        target_slice = target_vol[:, slice_idx * target_vol.shape[1] // input_vol.shape[1]].unsqueeze(1)
        visualize_result(input_slice, output_slice, target_slice)
    else:
        visualize_result(input_slice, output_slice)