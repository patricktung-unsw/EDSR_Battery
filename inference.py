import argparse
import os
import torch
import yaml
from utils.image_processing import process_image, process_volume
from utils.visualization import visualize_result, visualize_volume_result
from utils.io import save_output, load_input
from models.model_factory import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with super-resolution models')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to save output images')
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size for large images')
    parser.add_argument('--overlap', type=int, default=16, help='Overlap between patches')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--use_patches', action='store_true', help='Process large images in patches')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--model', type=str, default='EDSR', help='Model to use (e.g., EDSR, RCAN)')
    return parser.parse_args()

def update_config_from_args(config, args):
    """
    Update the configuration dictionary with command-line arguments.
    Only updates keys that exist in the config and have non-None values in args.
    """
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value

def main():
    args = parse_args()

    # Load configuration
    config = {'scale': args.scale}
    if args.config:
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    print("Configuration loaded:", config)

    # Update configuration with command-line arguments
    update_config_from_args(config, args)
    print("Final configuration:", config)

    # Create save directory if it doesn't exist
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    print(f"Save directory: {config['save_dir']}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model and load checkpoint
    model = get_model(args.model, config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Process input (file or directory)
    input_data, is_volume = load_input(args.input)
    if is_volume:
        output_data = process_volume(
            model, input_data, device=device, use_patches=args.use_patches,
            patch_size=args.patch_size, overlap=args.overlap
        )
        save_output(output_data, args.output, is_volume=True)
        if args.visualize:
            visualize_volume_result(input_data, output_data)
    else:
        output_data = process_image(
            model, input_data, device=device, use_patches=args.use_patches,
            patch_size=args.patch_size, overlap=args.overlap
        )
        save_output(output_data, args.output, is_volume=False)
        if args.visualize:
            visualize_result(input_data, output_data)

    print(f"Inference completed. Results saved to {args.output}")

if __name__ == '__main__':
    main()