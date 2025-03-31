# %%
import argparse
import torch
import os
import random
import numpy as np

import imageio
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url, extract_archive
from torchvision import transforms
from PIL import Image

import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
class DIV2K(Dataset):
    def __init__(self, root='./data', split='train', scale=4,  src_size=48, file_extension='png', volume_depth=128):
        """
        Args:"
        ""
        "        root (string): Root directory of the dataset.
        "        split (string): 'train' or 'valid'."
        "        scale (int): Scale factor for the images."
        "        src_size (int): Size of the source images.""
        "        "        file_extension (string): File extension of the images.""
        "        volume_depth (int): Number of slices in the 3D low -res volume.""
        "       """
        assert split in ['train', 'valid']
        assert scale in [2, 3, 4]

        self.split = split
        self.scale = scale
        self.src_size = src_size
        self.file_extension = file_extension

        # # Download urls
        # self.urls = {'train': {'tgt': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        #                        'src': f'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X{scale}.zip'},
        #             'valid': {'tgt': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        #                       'src': f'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X{scale}.zip'}}

        self.files = self.extract_dataset(root, split, scale)
        self.volume_depth = volume_depth
        self.current_volume_idx = None


    # Extract data and define the file paths
    def extract_dataset(self, root, split, scale):
        files = {}
        # set the root directory for the dataset
        # root = r"C:\Users\Patrick\Downloads\data"
        # root = './data'
        # iterate over the source and target samples
        for sample in ['src', 'tgt']:
            if sample == 'tgt':
                image_dir = os.path.join(root, f'{split}_HR')
            else:
                image_dir = os.path.join(root, f'{split}_LR_bicubic', 'X'+str(scale))

            # url = self.urls[split][sample]
            # filename = url.split('/')[-1]

            # # Check if the zip files already exists
            # if not os.path.exists(os.path.join(root, filename)):
            #     dataset_zip = download_url(url, root=root)

            # # define the directory where the images are extracted
            # if sample == 'tgt':
            #     image_dir = os.path.join(root, filename.split('.')[0])
            # else:
            #     image_dir = os.path.join(root, f'DIV2K_{self.split}_LR_bicubic', 'X'+str(scale))
            # # check if the zip file is already extracted
            # if not os.path.exists(image_dir):
            #     extract_archive(os.path.join(root, filename))

            # Get the file paths
            # files[sample] = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            files[sample] = sorted(glob.glob(os.path.join(image_dir, f'*.{self.file_extension}')))

        # check if the number of source and target images are the same
        assert len(files['src']) == len(files['tgt']), '{} != {}'.format(len(files['src']), len(files['tgt']))
        return files


    # Crop & resize the image to the apporiate scales
    def resize(self, src, tgt, scale, src_size, center=False):
        tgt_size = src_size * scale

        if center:
            src_left = src.size[0]//2 - src_size
            src_top = src.size[1]//2 - src_size
        else:
            src_left = random.choice(range(src.size[0] - src_size))
            src_top = random.choice(range(src.size[1] - src_size))

        tgt_left = src_left * scale
        tgt_top = src_top * scale

        src = TVF.crop(src, src_top, src_left, src_size, src_size)
        tgt = TVF.crop(tgt, tgt_top, tgt_left, tgt_size, tgt_size)

        return src, tgt


    # Data augmentation - horizonal flip, vertical flip, rotate
    def apply_transforms(self, src, tgt):
        # Convert NumPy arrays to PIL Images
        if isinstance(src, np.ndarray):
            src = Image.fromarray(src)
        if isinstance(tgt, np.ndarray):
            tgt = Image.fromarray(tgt)

        ## Radomly flip the images in the horizontal axis
        if random.random() > 0.5:
            src = TVF.hflip(src)
            tgt = TVF.hflip(tgt)

        ## Randomly rotate the images
        rotate_angle = random.choice([0, 90, 180, 270])
        src = TVF.rotate(src, angle=rotate_angle)
        tgt = TVF.rotate(tgt, angle=rotate_angle)

        return src, tgt


    # Returns the number of samples in our dataset (Required for torch DataLoader)
    def __len__(self):
        # return len(self.files['src']) * self.volume_depth
        return len(self.files['src']) * 3


    # Get a sample from out dataset (Required for torch DataLoader)
    def __getitem__(self, idx):
        # Calculate the slice index from the 3D volume
        volume_idx = idx // (self.volume_depth)
        slice_idx = idx % (self.volume_depth)
        # Load the 3D volume if it's a new volume
        if not hasattr(self, 'current_volume_src') or self.current_volume_idx != volume_idx:
            # Set the current volume index
            self.current_volume_idx = volume_idx
            # Load the source and target images for the current volume
            self.current_volume_src = imageio.volread(self.files['src'][volume_idx])
            self.current_volume_tgt = imageio.volread(self.files['tgt'][volume_idx])
            # Set the min and max values for the volumes for normalization
            self.current_volume_src_min = self.current_volume_src.min()
            self.current_volume_src_max = self.current_volume_src.max()
            self.current_volume_tgt_min = self.current_volume_tgt.min()
            self.current_volume_tgt_max = self.current_volume_tgt.max()
        # Extract the slice from the 3D volume
        src = self.current_volume_src[slice_idx, :, :]
        tgt = self.current_volume_tgt[slice_idx * self.scale, :, :]

        if self.split == 'train':
            # src, tgt = self.resize(src, tgt, self.scale, self.src_size)
            src, tgt = self.apply_transforms(src, tgt)
        # elif self.src_size is not None:
        #     src, tgt = self.resize(src, tgt, self.scale, self.src_size, center=True)

        # Convert images to numpy arrays
        src_np = np.array(src)
        tgt_np = np.array(tgt)

        # Convert to float32 before normalization
        src_np = src_np.astype(np.float32)
        tgt_np = tgt_np.astype(np.float32)

        # Min-Max Normalization to [0, 1]
        if self.current_volume_src_max > self.current_volume_src_min:  # Avoid division by zero
            src_np = (src_np - self.current_volume_src_min) / (self.current_volume_src_max - self.current_volume_src_min)
        if self.current_volume_tgt_max > self.current_volume_tgt_min:  # Avoid division by zero
            tgt_np = (tgt_np - self.current_volume_tgt_min) / (self.current_volume_tgt_max - self.current_volume_tgt_min)

        # Repeat the single channel to create 3 channels (RGB)
        src_np = np.repeat(src_np[:, :, np.newaxis], 3, axis=2)
        tgt_np = np.repeat(tgt_np[:, :, np.newaxis], 3, axis=2)

        # Ensure the shape is (height, width, 3)
        if src_np.shape[2] != 3:
            raise ValueError(f"Invalid shape for source image: {src_np.shape}")
        if tgt_np.shape[2] != 3:
            raise ValueError(f"Invalid shape for target image: {tgt_np.shape}")

        # Change the order of the channels to CxHxW to match PyTorch's convention
        src_np = np.transpose(src_np, (2, 0, 1))
        tgt_np = np.transpose(tgt_np, (2, 0, 1))

        # return TVF.to_tensor(src), TVF.to_tensor(tgt)
        return torch.from_numpy(src_np), torch.from_numpy(tgt_np)

# # %%
# def get_data_loaders():
#     #! CHANGE THIS HARD CODED VALUE
#     SCALE = 2

#     val_data = DIV2K(split='valid', scale=SCALE, file_extension='tiff', volume_depth=330)
#     val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

#     train_data = DIV2K(split='train', scale=SCALE, file_extension='tiff', volume_depth=330)
#     train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)

#     return train_loader, val_loader