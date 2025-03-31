import torch
import math

def psnr(pred, target):
    """Calculate PSNR between predicted and target images"""
    mse = torch.mean((pred - target) ** 2)  # Mean Squared Error
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    max_pixel_value = 1.0  # Assuming images are normalized to [0, 1]
    psnr_value = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr_value

def ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """Calculate SSIM between predicted and target images"""
    # Ensure the images are 4D tensors (batch_size, channels, height, width)
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)

    # Gaussian kernel
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    # Create a Gaussian filter
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    # Apply Gaussian filter
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)

    mu1 = torch.nn.functional.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()