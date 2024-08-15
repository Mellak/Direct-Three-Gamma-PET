import numpy as np
import pandas as pd
import sys
import os

# Get simulation number and file index from command-line arguments
simu_number = sys.argv[1]
file_idx = sys.argv[2]

# Define the file path for the simulation data
file_path = f"/homes/ymellak/Direct3G_f/O_Simu/Out{simu_number}/Sim_{file_idx}.hits.npy"
file = np.load(file_path, allow_pickle=True)

# Filter data to include only rows with PDGEncoding == 22
OR_file = file[file['PDGEncoding'] == 22]

# Get unique event IDs where photonID == 0
eventID_0 = np.unique(OR_file[OR_file['photonID'] == 0]['eventID'])

# Filter OR_file to only include rows with eventIDs in eventID_0
OR_file = OR_file[np.isin(OR_file['eventID'], eventID_0)]
print(f"OR_file shape after intersection: {OR_file.shape}")

# Extract source position coordinates (X, Y, Z) from the filtered data
photonIDs_0 = pd.DataFrame(OR_file[OR_file['photonID'] == 0])
photonIDs_0 = photonIDs_0.groupby('eventID').head(1)
photonIDs_0 = photonIDs_0[['sourcePosX', 'sourcePosY', 'sourcePosZ']].to_numpy()
print(f"photonIDs_0 shape: {photonIDs_0.shape}")

# Define the image size, voxel size, and image center
image_size = [200, 200, 100]
voxel_size = 3
image_center = np.array(image_size) // 2

# Initialize an empty image array
image = np.zeros(image_size)

# Function to convert mm coordinates to voxel indices
def mm_to_voxel(mm_coordinate, voxel_size=3, image_center=None):
    """Converts millimeter coordinates to voxel indices."""
    if image_center is None:
        raise ValueError("Image center must be provided")
    return np.round(mm_coordinate / voxel_size + image_center).astype(int)

# Print debugging information
print(f'Source shape: {photonIDs_0.shape}')
print(f"X-coordinate range: {np.min(photonIDs_0[:, 0])} to {np.max(photonIDs_0[:, 0])}")
print(f"Y-coordinate range: {np.min(photonIDs_0[:, 1])} to {np.max(photonIDs_0[:, 1])}")
print(f"Z-coordinate range: {np.min(photonIDs_0[:, 2])} to {np.max(photonIDs_0[:, 2])}")
print(f'Image center: {image_center}')
print('--------------')
print(f'First voxel min-max: {np.min(photonIDs_0[:, 0]) // voxel_size + 100} to {np.max(photonIDs_0[:, 0]) // voxel_size + 100}')
print(f'Second voxel min-max: {np.min(photonIDs_0[:, 1]) // voxel_size + 100} to {np.max(photonIDs_0[:, 1]) // voxel_size + 100}')
print(f'Third voxel min-max: {np.min(photonIDs_0[:, 2]) // voxel_size + 50} to {np.max(photonIDs_0[:, 2]) // voxel_size + 50}')

# Populate the image array, incrementing the voxel value for each source coordinate
for _src_coords in photonIDs_0:
    src_coords = mm_to_voxel(_src_coords, voxel_size, image_center)
    if (0 <= src_coords[0] < image_size[0] and 
        0 <= src_coords[1] < image_size[1] and 
        0 <= src_coords[2] < image_size[2]):
        image[src_coords[0], src_coords[1], src_coords[2]] += 1

# Convert the image to float32 and transpose to match the desired orientation
image = image.astype(np.float32)
image = np.transpose(image, (2, 1, 0))
print(f"Final image shape: {image.shape}")

# Save the image array as a binary file
output_path = f"/homes/ymellak/Direct3G_f/EmissionImages/Simu{simu_number}/EmissionImage_{file_idx}.bin"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
image.tofile(output_path)

# Remove the original simulation data file
# os.remove(file_path)
