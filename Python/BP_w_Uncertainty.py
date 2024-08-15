import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import multiprocessing as mp
import os

def mm_to_voxel(mm_coordinate, voxel_size=3, image_center=None):
    """Converts mm coordinate to voxel index."""
    if image_center is None:
        raise ValueError("Image center must be provided")
    return np.round(mm_coordinate // voxel_size + image_center).astype(int)

def line_points_3d(start, end, num=100):
    """Generate points along a 3D line from start to end in voxel coordinates."""
    return np.linspace(start, end, num)

def gaussian_density(distance, sigma):
    """Calculate Gaussian density."""
    return np.exp(-0.5 * (distance ** 2) / (sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def generate_non_symmetric_line_sparse(C_mm, C_prime_mm, x0_mm, factor1, factor2, image_size, voxel_size=3):
    image_center = np.array(image_size) // 2
    # Convert mm coordinates to voxel indices
    C = mm_to_voxel(C_mm, voxel_size, image_center)
    C_prime = mm_to_voxel(C_prime_mm, voxel_size, image_center)
    x0 = mm_to_voxel(x0_mm, voxel_size, image_center)

    # Calculate points based on factors
    point1 = x0 + factor1 * (C - x0)
    point2 = x0 + factor2 * (C_prime - x0)
    
    # Sigma based on distances to point1 and point2, adjusted for voxel size
    sigma1 = np.linalg.norm((point1 - x0) * voxel_size)
    sigma2 = np.linalg.norm((point2 - x0) * voxel_size)
    
    # Generate line points in voxel coordinates
    line = line_points_3d(C, C_prime, num=100)
    sparse_values = []

    for point in line:
        distance = np.linalg.norm((point - x0) * voxel_size)  # Adjust distance for voxel size
        sigma = sigma1 if np.dot(point - x0, C_prime - x0) <= 0 else sigma2
        density = gaussian_density(distance, sigma)
        
        
        ix, iy, iz = np.round(point).astype(int)
        if 0 <= ix < image_size[0] and 0 <= iy < image_size[1] and 0 <= iz < image_size[2]:
            sparse_values.append(((ix, iy, iz), density))
    
    # Normalize the density values on sum of all densities:
    sum_density = sum([density for _, density in sparse_values])
    sparse_values = [(voxel, density/sum_density) for voxel, density in sparse_values]
    
    return sparse_values




def generate_symmetric_line_sparse_final(C_mm, C_prime_mm, x0_mm, sigma_minus, sigma_plus, image_size, voxel_size=3):
    image_center = np.array(image_size) // 2
    
    # Convert mm coordinates to voxel indices
    C = mm_to_voxel(C_mm, voxel_size, image_center)
    C_prime = mm_to_voxel(C_prime_mm, voxel_size, image_center)
    x0 = mm_to_voxel(x0_mm, voxel_size, image_center)
    
    # Calculate the number of points based on the number of voxels the line spans
    distance_mm = np.linalg.norm(C_prime_mm - C_mm) // voxel_size
    num_points = int(np.round(distance_mm)) + 1
    line = line_points_3d(C, C_prime, num=num_points)
    
    # Dictionary to accumulate densities
    voxel_densities = {}
    for point in line:
        distance = np.linalg.norm((point - x0) * voxel_size)  # Adjust distance for voxel size
        sigma = sigma_plus if np.dot(point - x0, C_prime - x0) >= 0 else sigma_minus
        density = gaussian_density(distance, sigma)
        
        ix, iy, iz = np.round(point).astype(int)
        voxel_key = (ix, iy, iz)
        if 0 <= ix < image_size[0] and 0 <= iy < image_size[1] and 0 <= iz < image_size[2]:
            if voxel_key not in voxel_densities:
                voxel_densities[voxel_key] = []
            voxel_densities[voxel_key].append(density)
    
    # Calculate average density for each voxel
    average_sparse_values = []
    for voxel, densities in voxel_densities.items():
        average_density = sum(densities) / len(densities)
        average_sparse_values.append((voxel, average_density))
    
    # Normalize the density values based on the sum of all densities
    sum_density = sum(density for _, density in average_sparse_values)
    normalized_sparse_values = [(voxel, density / sum_density) for voxel, density in average_sparse_values]
    
    return normalized_sparse_values

def fill_f_image(f_image, sparse_values):
    for voxel, density in sparse_values:
        ix, iy, iz = voxel
        f_image[ix, iy, iz] = f_image[ix, iy, iz] + max(0, density)
        
        
def generate_symmetric_line_sparse_final_parallel_factorized(args):
    C_mm, C_prime_mm, x0_mm, sigma_minus, sigma_plus, image_size, voxel_size = args
    image_center = np.array(image_size) // 2
    
    # Convert mm coordinates to voxel indices
    C = mm_to_voxel(C_mm, voxel_size, image_center)
    C_prime = mm_to_voxel(C_prime_mm, voxel_size, image_center)
    x0 = mm_to_voxel(x0_mm, voxel_size, image_center)
    
    # Generate line points in voxel coordinates
    distance_mm = np.linalg.norm(C_prime_mm - C_mm) // voxel_size
    num_points = int(np.round(distance_mm)) + 1
    line = line_points_3d(C, C_prime, num=num_points)
    
    # Compute distances and densities
    distances = np.linalg.norm((line - x0) * voxel_size, axis=1)
    directionality = np.dot(line - x0, C_prime - x0)
    sigmas = np.where(directionality >= 0, sigma_plus, sigma_minus)
    densities = gaussian_density(distances, sigmas)
    
    # Calculate voxel indices
    voxel_indices = np.round(line).astype(int)
    valid_mask = (voxel_indices[:, 0] < image_size[0]) & (voxel_indices[:, 1] < image_size[1]) & (voxel_indices[:, 2] < image_size[2])
    
    # Filter valid voxels and densities
    valid_voxel_indices = voxel_indices[valid_mask]
    valid_densities = densities[valid_mask]
    
    # Average densities per voxel using a coordinate-based approach
    unique_voxels, indices = np.unique(valid_voxel_indices, axis=0, return_inverse=True)
    summed_densities = np.bincount(indices, weights=valid_densities)
    counts = np.bincount(indices)
    average_densities = summed_densities / counts
    
    # Normalize the density values
    sum_density = np.sum(average_densities)
    normalized_densities = average_densities / sum_density
    
    # Prepare final sparse values list
    normalized_sparse_values = list(zip(map(tuple, unique_voxels), normalized_densities))
    
    return normalized_sparse_values
    
    
    
def generate_symmetric_line_sparse_final_parallel_factorized2(args):
    C_mm, C_prime_mm, x0_mm, sigma_minus, sigma_plus, P1_mm, image_size, voxel_size = args
    image_center = np.array(image_size) // 2
    
    # Convert mm coordinates to voxel indices
    C = mm_to_voxel(C_mm, voxel_size, image_center)
    C_prime = mm_to_voxel(C_prime_mm, voxel_size, image_center)
    x0 = mm_to_voxel(x0_mm, voxel_size, image_center)
    P1 = mm_to_voxel(P1_mm, voxel_size, image_center)
    
    # Generate line points in voxel coordinates
    distance_mm = np.linalg.norm(C_prime_mm - C_mm) // voxel_size
    num_points = int(np.round(distance_mm)) + 1
    line_b2b = line_points_3d(C, C_prime, num=num_points)
    # Calculate voxel indices
    voxel_indices = np.round(line_b2b).astype(int)
    # get unique voxel indices
    voxel_indices = np.unique(voxel_indices, axis=0)
    valid_mask = (voxel_indices[:, 0] < image_size[0]) & (voxel_indices[:, 1] < image_size[1]) & (voxel_indices[:, 2] < image_size[2])
    valid_voxel_indices = voxel_indices[valid_mask]
    # Calculate the sum of attenuation values along the LOR
    attenuation_values = attenuation_511[valid_voxel_indices[:, 0], valid_voxel_indices[:, 1], valid_voxel_indices[:, 2]] 
    attenuation_sum = np.sum(attenuation_values)
    # each voxel represent 3mm: 0.096 cm^-1
    attenuation_sum = attenuation_sum * 3/10
    #print('attenuation_sum:', attenuation_sum)
    attenuation_factor_511 = np.exp(attenuation_sum)

    # ---Third gamma photon----
    # Generate line points in voxel coordinates
    distance_mm_3gamma = np.linalg.norm(P1_mm - x0_mm) // voxel_size
    num_points_3gamma = int(np.round(distance_mm_3gamma)) + 1
    line_third_gamma = line_points_3d(x0, P1, num=num_points_3gamma)

    # Calculate voxel indices
    voxel_indices_3gamma = np.round(line_third_gamma).astype(int)
    # get unique voxel indices
    voxel_indices_3gamma = np.unique(voxel_indices_3gamma, axis=0)
    
    
    valid_mask_3gamma = (voxel_indices_3gamma[:, 0] < image_size[0]) & (voxel_indices_3gamma[:, 1] < image_size[1]) & (voxel_indices_3gamma[:, 2] < image_size[2])
    valid_voxel_indices_3gamma = voxel_indices_3gamma[valid_mask_3gamma]
    #valid_voxel_indices_3gamma = np.clip(valid_voxel_indices_3gamma, [0, 0, 0], [200, 200, 100])

    # Calculate the sum of attenuation values along the LOR
    attenuation_values_3gamma = attenuation_1157[valid_voxel_indices_3gamma[:, 0], valid_voxel_indices_3gamma[:, 1], valid_voxel_indices_3gamma[:, 2]]
    attenuation_sum_3gamma = np.sum(attenuation_values_3gamma)
    # each voxel represent 3mm: 0.096 cm^-1
    attenuation_sum_3gamma = attenuation_sum_3gamma * 3/10
    #print('attenuation_sum:', attenuation_sum_3gamma)
    attenuation_factor_1157 = np.exp(attenuation_sum_3gamma)


    # Compute distances and densities
    distances = np.linalg.norm((line_b2b - x0) * voxel_size, axis=1)
    directionality = np.dot(line_b2b - x0, C_prime - x0)
    sigmas = np.where(directionality >= 0, sigma_plus, sigma_minus)
    densities = gaussian_density(distances, sigmas)
    
    
    # Calculate voxel indices
    voxel_indices = np.round(line_b2b).astype(int)
    valid_mask = (voxel_indices[:, 0] < image_size[0]) & (voxel_indices[:, 1] < image_size[1]) & (voxel_indices[:, 2] < image_size[2])
    
    # Filter valid voxels and densities
    valid_voxel_indices = voxel_indices[valid_mask]
    valid_densities = densities[valid_mask]
    
    # Average densities per voxel using a coordinate-based approach
    unique_voxels, indices = np.unique(valid_voxel_indices, axis=0, return_inverse=True)
    summed_densities = np.bincount(indices, weights=valid_densities)
    counts = np.bincount(indices)
    average_densities = summed_densities / counts
    
    # Normalize the density values
    sum_density = np.sum(average_densities)

    normalized_densities =  average_densities * 3 # / sum_density

    normalized_densities = normalized_densities * attenuation_factor_511 * attenuation_factor_1157
    
    # Prepare final sparse values list
    normalized_sparse_values = list(zip(map(tuple, unique_voxels), normalized_densities))
    
    return normalized_sparse_values
    


        
def main(simu_number, file_idx):


    ppoints_array = np.load('/homes/ymellak/Direct3G_f/PSource_w_U/Simu'+str(simu_number)+'/PSource_w_U'+str(file_idx)+'.npy', allow_pickle=True)
    

    f_image = np.zeros(image_size)

    process_args = []
    for C_mm, C_mm_p, p_emission_point, point_s_minus, point_s_plus, point_e_minus, point_e_plus, prompt_gamma_1 in ppoints_array:
        sigma_e_minus = np.linalg.norm((point_e_minus - p_emission_point) * voxel_size)
        sigma_e_plus = np.linalg.norm((point_e_plus - p_emission_point) * voxel_size)
        sigma_s_minus = np.linalg.norm((point_s_minus - p_emission_point) * voxel_size)
        sigma_s_plus = np.linalg.norm((point_s_plus - p_emission_point) * voxel_size)
        sigma_minus = np.sqrt(sigma_s_minus**2 + sigma_e_minus**2)
        sigma_plus = np.sqrt(sigma_s_plus**2 + sigma_e_plus**2)
        
        
        args = (C_mm, C_mm_p, p_emission_point, sigma_minus, sigma_plus, prompt_gamma_1, image_size, voxel_size)
        process_args.append(args)
    
    start_time = time.time()
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(generate_symmetric_line_sparse_final_parallel_factorized2, process_args)
    
    
    end_time = time.time()
    print('Time taken: ', end_time - start_time)

    start_time = time.time()
    for sparse_values in results:
        fill_f_image(f_image, sparse_values)
    end_time = time.time()
    print('Time taken to fill image: ', end_time - start_time)
    f_image = f_image.astype(np.float32)
    # flip dimensions and make them 2, 1, 0
    #f_image = np.flip(f_image, axis=0)
    #f_image = np.flip(f_image, axis=1)
    # rotate 90 degrees
    # transpose it:
    f_image = np.transpose(f_image, (2, 1, 0))
    print(f_image.shape)
    # flip z axis
    #f_image = np.flip(f_image, axis=2)
    f_image.tofile('/homes/ymellak/Direct3G_f/BPImages/Simu'+str(simu_number)+'/BPUImage_wA_' + str(file_idx) +'.bin')
    

print(len(sys.argv))
print(sys.argv)

image_size = [200, 200, 200]
voxel_size = 3     

simu_number = sys.argv[1]
file_idx    = sys.argv[2]

attenuation_511 = np.fromfile("/homes/ymellak/Direct3G_f/Gate/Phantoms/Phantom"+str(simu_number)+"/Attenuation_511.bin", dtype=np.float32).reshape([200, 200, 200])
attenuation_1157 = np.fromfile("/homes/ymellak/Direct3G_f/Gate/Phantoms/Phantom"+str(simu_number)+"/Attenuation_1157.bin", dtype=np.float32).reshape([200, 200, 200])

#print(attenuation_1157.shape, attenuation_1157.shape)

main(simu_number, file_idx)


