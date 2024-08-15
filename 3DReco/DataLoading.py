import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate, shift
import random
import matplotlib.pyplot as plt

class SimulationDataset(Dataset):
    def __init__(self, root_dir, transform=True):
        self.root_dir = root_dir
        self.folders = [f for f in os.listdir(root_dir) if f.startswith('Simu')]
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        folder_path = os.path.join(self.root_dir, folder)

        # Find the correct filenames
        input_file = [f for f in os.listdir(folder_path) if f.startswith('BPUImage_wA_')][0]
        target_file = [f for f in os.listdir(folder_path) if f.startswith('EmissionImage_')][0]

        # Read the binary files
        input_image = np.fromfile(os.path.join(folder_path, input_file), dtype=np.float32).reshape(200, 200, 200)
        target_image = np.fromfile(os.path.join(folder_path, target_file), dtype=np.float32).reshape(100, 200, 200)
        
        # Pad the target image to make it 200x200x200, keeping the center
        target_image = np.pad(target_image, ((50, 50), (0, 0), (0, 0)), 'constant')

        if self.transform:
            input_image, target_image = self.apply_augmentations(input_image, target_image)

        # Convert to PyTorch tensors and unsqueeze to add channel dimension
        input_tensor = torch.from_numpy(input_image).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_image).float().unsqueeze(0)

        return input_tensor, target_tensor

    def apply_augmentations(self, input_image, target_image):
        # Apply random augmentations: flips, rotations, translations, and intensity scaling

        # Horizontal flip
        if random.random() < 0.5:
            input_image = np.flip(input_image, axis=1)
            target_image = np.flip(target_image, axis=1)

        # Vertical flip
        if random.random() < 0.5:
            input_image = np.flip(input_image, axis=0)
            target_image = np.flip(target_image, axis=0)

        # 3D rotation
        angles = np.random.uniform(0, 180, 3)
        for axis, angle in zip([(1, 2), (0, 2), (0, 1)], angles):
            input_image = rotate(input_image, angle, axes=axis, reshape=False)
            target_image = rotate(target_image, angle, axes=axis, reshape=False)

        # 3D translation
        shifts = np.random.uniform(-50, 50, 3)
        input_image = shift(input_image, shifts, mode='nearest')
        target_image = shift(target_image, shifts, mode='nearest')

        # Intensity scaling
        scale_factor = np.random.uniform(0.1, 4)
        input_image *= scale_factor
        target_image *= scale_factor

        return input_image, target_image

class NormalizedSimulationDataset(SimulationDataset):
    def __getitem__(self, idx):
        input_tensor, target_tensor = super().__getitem__(idx)
        
        # Normalize the images to [0, 1]
        input_tensor = self.normalize(input_tensor)
        target_tensor = self.normalize(target_tensor)

        return input_tensor, target_tensor

    def normalize(self, image):
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return torch.zeros_like(image)

class SimulationDatasetAttenuation(Dataset):
    def __init__(self, root_dir, transform=True):
        self.root_dir = root_dir
        self.folders = [f for f in os.listdir(root_dir) if f.startswith('Simu')]
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        folder_path = os.path.join(self.root_dir, folder)

        # Find the correct filenames
        input_file = [f for f in os.listdir(folder_path) if f.startswith('BPUImage_wA_')][0]
        target_file = [f for f in os.listdir(folder_path) if f.startswith('EmissionImage_')][0]
        material_file = "Materials_image.bin"  # Assuming this is the fixed name for material images

        # Read the binary files
        input_image = np.fromfile(os.path.join(folder_path, input_file), dtype=np.float32).reshape(200, 200, 200)
        target_image = np.fromfile(os.path.join(folder_path, target_file), dtype=np.float32).reshape(100, 200, 200)
        material_image = np.fromfile(os.path.join(folder_path, material_file), dtype=np.int32).reshape(100, 200, 200)

        # Pad the target and material images to make them 200x200x200
        target_image = np.pad(target_image, ((50, 50), (0, 0), (0, 0)), 'constant')
        material_image = np.pad(material_image, ((50, 50), (0, 0), (0, 0)), 'constant')

        if self.transform:
            input_image, target_image, material_image = self.apply_augmentations(input_image, target_image, material_image)

        # Convert to PyTorch tensors and unsqueeze to add channel dimension
        input_tensor = torch.from_numpy(input_image).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_image).float().unsqueeze(0)
        material_tensor = torch.from_numpy(material_image).long().unsqueeze(0)

        return input_tensor, target_tensor, material_tensor

    def apply_augmentations(self, input_image, target_image, material_image):
        # Apply random augmentations: flips, translations, and intensity scaling

        # Horizontal flip
        if random.random() < 0.5:
            input_image = np.flip(input_image, axis=1)
            target_image = np.flip(target_image, axis=1)
            material_image = np.flip(material_image, axis=1)

        # Vertical flip
        if random.random() < 0.5:
            input_image = np.flip(input_image, axis=0)
            target_image = np.flip(target_image, axis=0)
            material_image = np.flip(material_image, axis=0)

        # Depth flip
        if random.random() < 0.5:
            input_image = np.flip(input_image, axis=2)
            target_image = np.flip(target_image, axis=2)
            material_image = np.flip(material_image, axis=2)

        # 3D translation
        shifts = np.random.uniform(-50, 50, 3)
        input_image = shift(input_image, shifts, mode='nearest')
        target_image = shift(target_image, shifts, mode='nearest')
        material_image = shift(material_image, shifts, mode='nearest')

        # Clip material values to a specific range
        material_image = np.clip(material_image, 0, 3)

        return input_image, target_image, material_image

# DataLoader creation function
def get_dataloader(root_dir, batch_size=4, num_workers=4, shuffle=True):
    dataset = SimulationDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Usage example
if __name__ == "__main__":
    root_dir = "/home/youness/data/Optuna_Project/GAN_3gamma/Preprocess/FullPipeline/DataSet"
    dataloader = get_dataloader(root_dir, batch_size=1)

    for batch_input, batch_target in dataloader:
        print(f"Batch input shape: {batch_input.shape}")
        print(f"Batch target shape: {batch_target.shape}")
        print('Sum of values:', batch_input.sum(), batch_target.sum())

        # Show the center slice of the batch_input and batch_target on the same figure
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(batch_input[0, 0, :, :, 100], cmap='gray')
        ax[0].set_title("Input Image")
        ax[1].imshow(batch_target[0, 0, :, :, 100], cmap='gray')
        ax[1].set_title("Target Image")
        plt.show()

        break

