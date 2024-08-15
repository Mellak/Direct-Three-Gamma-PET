import numpy as np
import os
import sys
import glob

def parse_image_size(image_size_str):
    return tuple(map(int, image_size_str.split(',')))

number = sys.argv[1]
directory = sys.argv[2]
file_prefix = sys.argv[3]
image_size = sys.argv[4]

image_shape = parse_image_size(image_size)
combined_image = np.zeros(image_shape, dtype=np.float32)

file_pattern = os.path.join(directory, f"{file_prefix}*.bin")
files_to_process = glob.glob(file_pattern)

for file_path in files_to_process:
    print('processing', os.path.basename(file_path))
    
    image = np.memmap(file_path, dtype=np.float32, mode='r', shape=image_shape)
    np.add(combined_image, image, out=combined_image)

# Batch delete files
for file_path in files_to_process:
    os.remove(file_path)

output_path = f'/homes/ymellak/Direct3G_f/3DImages/Simu{number}/{file_prefix}{number}.bin'
combined_image.tofile(output_path)
