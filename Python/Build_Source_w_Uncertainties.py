import os
import random
import time
import numpy as np
import math
from torch.distributions import Normal, Independent
from itertools import combinations, permutations
import functools
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor






def calculate_plane_normal(pt1, pt2, pt3):
    vec1 = pt2 - pt1  # Vector from pt1 to pt2
    vec2 = pt3 - pt1  # Vector from pt1 to pt3
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(normal)  # Normalize the vector
    return normal

def rotate_around_axis(vector, axis, theta, origin):
    """
    Rotate a vector around an axis by theta radians originating from 'origin'.
    """
    # Move vector to origin
    vector_relative = vector - origin

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula components
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rodrigues' Rotation Formula
    rotated_vector = (vector_relative * cos_theta +
                      np.cross(axis, vector_relative) * sin_theta +
                      axis * np.dot(axis, vector_relative) * (1 - cos_theta))

    # Move vector back to its original position relative to the origin
    rotated_vector += origin

    return rotated_vector


    
def find_intersection_with_line(pt1, pt2, origin, direction):
    # Convert points to numpy arrays for vector operations
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    origin = np.array(origin)
    direction = np.array(direction)
    
    # Calculate the direction vector for the line defined by pt1 and pt2
    line_dir = pt2 - pt1
    
    # Create the matrix A where the columns are the direction vectors of the lines negated appropriately
    A = np.column_stack((line_dir, -direction))
    
    # Calculate the right hand side of the equation (origin - pt1)
    B = origin - pt1
    
    # We need to solve A * [t, s]^T = B
    # Use the least squares solution as A is generally not square
    try:
        ts, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        t, s = ts
        # Calculate the intersection point using t
        intersection = pt1 + t * line_dir
        return intersection
    except np.linalg.LinAlgError:
        # This might happen if the matrix A is singular, i.e., lines are parallel or coincident
        return None


def line_cone_intersection(line_point1, line_point2, prompt_gamma_1, prompt_gamma_2):
    me = 9.10938356/1e31 #Kg
    c = 299792458 #m/s
    c2 = c*c #m2/s2

    E_0 =  1.157 * 1e6 * 1.63 * 9.8 / 1e20 #Mev -- > Kg.m/s2 #1.157
    E_1 = prompt_gamma_1[0] * 1e6 * 1.63 * 9.8 / 1e20 #Mev -- > Kg.m/s2

    
    # Debora equation:
    #cone_angle = np.arccos(1 - me*c2 *(E_1/(E_0 - E_1)/E_0))
    cone_angle = np.arccos(1 + me*c2 * (1/E_0 - 1/(E_0-E_1)))

    D = line_point1 - line_point2
    D = D / np.linalg.norm(D)

    C = prompt_gamma_1[1:]
    O = line_point1
    CO = O-C
    V = prompt_gamma_1[1:] - prompt_gamma_2[1:]
    V = V / np.linalg.norm(V)

    myb = 2*((np.dot(D,V)*np.dot(CO,V)-np.dot(D,CO)*np.cos(cone_angle)*np.cos(cone_angle)))
    mya = np.dot(D,V) * np.dot(D,V) - np.cos(cone_angle)*np.cos(cone_angle)
    myc = np.dot(CO,V) * np.dot(CO,V) - np.dot(CO,CO) * np.cos(cone_angle)*np.cos(cone_angle)
    delta = myb*myb - 4 *mya*myc

    if(delta>0):
        t1 = (-myb-np.sqrt(delta))/(2*mya)
        my_pt1 = O + t1*D
        t2 = (-myb+np.sqrt(delta))/(2*mya)
        my_pt2 = O + t2*D
        if np.dot(my_pt1 - line_point1,my_pt1 - line_point2) > 0:
            my_pt1 = np.nan
        # check if my_pt2 is between line_point1 and line_point2:
        if np.dot(my_pt2 - line_point1,my_pt2 - line_point2) > 0:
            my_pt2 = np.nan
        # return a list of two points
        return [my_pt1, my_pt2]
        
    elif(delta==0):
        t = (-myb)/(2*mya)
        my_pt = O + t*D
        return [my_pt]

    else:
        return [np.nan]

# define a function to get angle given E0 and E1:
def get_compton_angle(E0, E1):
    me = 9.10938356/1e31 #Kg
    c = 299792458 #m/s
    c2 = c*c #m2/s2

    E_0 =  E0 * 1e6 * 1.63 * 9.8 / 1e20 #Mev -- > Kg.m/s2 #1.157
    E_1 = E1 * 1e6 * 1.63 * 9.8 / 1e20 #Mev -- > Kg.m/s2

    # Debora equation:
    cone_angle = np.arccos(1 + me*c2 * (1/E_0 - 1/(E_0-E_1)))
    return cone_angle

def line_cone_intersection2(line_point1, line_point2, prompt_gamma_1, prompt_gamma_2, cone_angle):
    
    #cone_angle = get_compton_angle(1.157, prompt_gamma_1[0])

    D = line_point1 - line_point2
    D = D / np.linalg.norm(D)

    C = prompt_gamma_1 #[1:]
    O = line_point1
    CO = O-C
    V = prompt_gamma_1 - prompt_gamma_2
    V = V / np.linalg.norm(V)

    myb = 2*((np.dot(D,V)*np.dot(CO,V)-np.dot(D,CO)*np.cos(cone_angle)*np.cos(cone_angle)))
    mya = np.dot(D,V) * np.dot(D,V) - np.cos(cone_angle)*np.cos(cone_angle)
    myc = np.dot(CO,V) * np.dot(CO,V) - np.dot(CO,CO) * np.cos(cone_angle)*np.cos(cone_angle)
    delta = myb*myb - 4 *mya*myc

    if(delta>0):
        t1 = (-myb-np.sqrt(delta))/(2*mya)
        my_pt1 = O + t1*D
        t2 = (-myb+np.sqrt(delta))/(2*mya)
        my_pt2 = O + t2*D
        if np.dot(my_pt1 - line_point1,my_pt1 - line_point2) > 0:
            my_pt1 = np.nan
        # check if my_pt2 is between line_point1 and line_point2:
        if np.dot(my_pt2 - line_point1,my_pt2 - line_point2) > 0:
            my_pt2 = np.nan
        # return a list of two points
        return [my_pt1, my_pt2]
        
    elif(delta==0):
        t = (-myb)/(2*mya)
        my_pt = O + t*D
        return [my_pt]

    else:
        return [np.nan]
    
def add_uncertitude(energy, FWHM_percentage=5):
    sc_energy = 1.157
    return energy + np.random.normal(0, (FWHM_percentage/ 100 / 2.355) * sc_energy)

def add_spatial_uncertitude(point, uncertitude_mm=3):
    return point + np.random.uniform(-uncertitude_mm/2, uncertitude_mm/2, 3)


def calculate_theta_c_with_error_propagation(r2, r1, r0, uncertitude_in_mm=3):
    # Convert inputs into numpy arrays
    r2 = np.array(r2)
    r1 = np.array(r1)
    r0 = np.array(r0)
    
    # Define vectors u and v
    u = r2 - r1
    v = r1 - r0
    
    # Calculate norms of u and v
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # Calculate the dot product of u and v
    dot_uv = np.dot(u, v)
    
    # Calculate the cosine of theta_C
    cos_theta_c = dot_uv / (norm_u * norm_v)
    
    # Calculate sigma for uniform distribution of errors
    sigma = uncertitude_in_mm / np.sqrt(12)  # Standard deviation for uniform distribution
    sigma_u = np.sqrt(2) * sigma  # Error in vector u = r2 - r1, combining errors from r2 and r1
    sigma_v = sigma              # Error in vector v = r1 - r0 (since r0 is fixed and known)
    
    # Calculate derivatives of cos(theta_C)
    partial_cos_theta_u = -dot_uv / (norm_u**3 * norm_v)
    partial_cos_theta_v = -dot_uv / (norm_u * norm_v**3)
    
    # Calculate Δcos(θ_C) using error propagation
    delta_cos_theta_c = np.sqrt(
        (partial_cos_theta_u * (sigma_u / norm_u))**2 +
        (partial_cos_theta_v * (sigma_v / norm_v))**2
    )
    
    if cos_theta_c != 1:
        sin_theta_c = np.sqrt(1 - cos_theta_c**2)
        delta_theta_c = np.abs(-1 / sin_theta_c * delta_cos_theta_c)
    else:
        delta_theta_c = 0


    theta_c = np.arccos(cos_theta_c)
    
    
    return theta_c, delta_theta_c #theta_c_plus, theta_c_minus


def propagate_energy_uncertitude(E0, E1, FWHM_percentage=5):
    
    theta_C = get_compton_angle(E0, E1)
    # Calculate cos(theta_C)
    cos_theta_C = np.cos(theta_C)
    
    # Convert FWHM to standard deviation (sigma)
    sigma_Ee = ((FWHM_percentage / 2.355) * E1) / 100  # Converting FWHM to sigma based on Ee
    
    # Calculate the derivative of cos(theta_C) with respect to Ee
    derivative = -0.5125 / ((E0 - E1)**2)
    
    # Calculate uncertainty in cos(theta_C)
    delta_cos_theta_C = abs(derivative) * sigma_Ee
    sin_theta_C = np.sin(theta_C)
    # Calculate the uncertainty in theta_C using error propagation
    if sin_theta_C != 0:
        Delta_theta_C = delta_cos_theta_C / sin_theta_C
    else:
        Delta_theta_C = 0  # When cos_theta_C is exactly 1 or -1, the derivative goes to infinity (tangent is vertical)

    return theta_C, Delta_theta_C

def compute_uncertain_intersections(line_point1, line_point2, prompt_gamma_1, prompt_gamma_2, point_c, theta_c, delta_theta_c):
    
    source_array_local = np.empty((0, 3, 3), dtype=float)
    
    normal = calculate_plane_normal(prompt_gamma_1, line_point1, line_point2)
    
    vector_c = point_c - prompt_gamma_1
    delta_theta_plus  = delta_theta_c #/2
    delta_theta_minus = -delta_theta_c #/2

    # Rotate vectors
    vector_plus  = rotate_around_axis(vector_c, normal, delta_theta_plus, prompt_gamma_1)
    vector_minus = rotate_around_axis(vector_c, normal, delta_theta_minus, prompt_gamma_1)

    

    # Compute intersections
    intersection_plus  = find_intersection_with_line(line_point1, line_point2, prompt_gamma_1, vector_plus)
    intersection_minus = find_intersection_with_line(line_point1, line_point2, prompt_gamma_1, vector_minus)
    
    
    # order the points given direction of line_point1 to line_point2:
    if np.linalg.norm(intersection_plus - line_point1) > np.linalg.norm(intersection_plus - line_point2):
        intersection_plus, intersection_minus = intersection_minus, intersection_plus

    source_array_local = np.append(source_array_local, [[intersection_plus, point_c, intersection_minus]], axis=0)
    
    
    return source_array_local


# Define a function that takes line_point1, line_point2, prompt_gamma_1, prompt_gamma_2:
def Predict_Points_w_uncertitudes(line_point1, line_point2, prompt_gamma_1, prompt_gamma_2):
    theta_e, delta_theta_e = propagate_energy_uncertitude(1.157, prompt_gamma_1[0], FWHM_percentage=5)

    # get the intersection points with estimated angle:
    # ppoints = line_cone_intersection2(line_point1, line_point2, prompt_gamma_1[1:], prompt_gamma_2[1:], theta_e)
    # get the intersection points with Klein-Nishina
    ppoints = line_cone_intersection(line_point1, line_point2, prompt_gamma_1, prompt_gamma_2)
    ppoints = [point for point in ppoints if not np.isnan(point).any()]
    output_list = []
    if len(ppoints) != 0:
        for p_point in ppoints:
            theta_s, delta_theta_s = calculate_theta_c_with_error_propagation(prompt_gamma_1[1:], prompt_gamma_2[1:], p_point, uncertitude_in_mm=3)
            selected_element_s = compute_uncertain_intersections(line_point1, line_point2, prompt_gamma_1[1:], prompt_gamma_2[1:], p_point, theta_s, delta_theta_s)    
            selected_element_e = compute_uncertain_intersections(line_point1, line_point2, prompt_gamma_1[1:], prompt_gamma_2[1:], p_point, theta_e, delta_theta_e)
            
            #print('selected_element_s:', selected_element_s.shape)
            #print('selected_element_e:', selected_element_e.shape)
            
            point_s_minus = selected_element_s[0,0]
            point_s_plus  = selected_element_s[0,2]

            point_e_minus = selected_element_e[0,0]
            point_e_plus  = selected_element_e[0,2]

            minus_point = line_point1
            plus_point  = line_point2

            # save the points in a list organized as minus_point, plus_point, point_p, point_s_minus, point_s_plus, point_e_minus, point_e_plus
            
            final_list = [minus_point, plus_point, p_point, point_s_minus, point_s_plus, point_e_minus, point_e_plus, prompt_gamma_1[1:]]
            # make it a numpy array
            final_list = np.array(final_list)
            #print(final_list.shape)
            #return final_list
            output_list.append(final_list)
    return output_list


def process_events_sequential(detector_data):
    outputs = []
    for event_interactions in detector_data:
        #event_interactions = detector_data[detector_data[:, 0] == event_idx]
        #source = source_data[source_data[:, 0] == event_idx][0, 1:]
        line_point1, line_point2 = event_interactions[0, 1:], event_interactions[1, 1:]
        prompt_gamma_1 = event_interactions[2, :]
        prompt_gamma_2 = event_interactions[3, :]
        
        # Apply spatial uncertitude
        prompt_gamma_1[1:] = add_spatial_uncertitude(prompt_gamma_1[1:], uncertitude_mm=3)
        prompt_gamma_2[1:] = add_spatial_uncertitude(prompt_gamma_2[1:], uncertitude_mm=3)
        # Apply energy uncertitude
        prompt_gamma_1[0] = add_uncertitude(prompt_gamma_1[0], FWHM_percentage=5)

        output = Predict_Points_w_uncertitudes(line_point1, line_point2, prompt_gamma_1, prompt_gamma_2)
        if len(output) > 0:
            outputs.append(np.array(output))
    return outputs


import multiprocessing

def process_single_event(args):
    event_interactions = args
    outputs = []
    #event_interactions = detector_data[detector_data[:, 0] == event_idx]
    #source = source_data[source_data[:, 0] == event_idx][0, 1:]
    line_point1, line_point2 = event_interactions[0, 1:], event_interactions[1, 1:]
    prompt_gamma_1 = event_interactions[2, :]
    prompt_gamma_2 = event_interactions[3, :]
    
    prompt_gamma_1[1:] = add_spatial_uncertitude(prompt_gamma_1[1:], uncertitude_mm=0.5)
    prompt_gamma_2[1:] = add_spatial_uncertitude(prompt_gamma_2[1:], uncertitude_mm=0.5)
    prompt_gamma_1[0] = add_uncertitude(prompt_gamma_1[0], FWHM_percentage=5)

    output = Predict_Points_w_uncertitudes(line_point1, line_point2, prompt_gamma_1, prompt_gamma_2)
    if len(output) > 0:
        outputs.append(np.array(output))
    return outputs

def process_events_parallel(detector_data):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #event_indices = np.unique(detector_data[:, 0])
    args = [(data) for data in detector_data]
    result = pool.map(process_single_event, args)
    pool.close()
    pool.join()
    # Flatten the list of outputs
    outputs = [item for sublist in result for item in sublist]
    return outputs
    


simu_number = sys.argv[1]
file_idx    = sys.argv[2]


detector_data = np.load("/homes/ymellak/Direct3G_f/Detectors/Simu"+str(simu_number)+"/Detector_"+str(file_idx)+".npy")
print(detector_data.shape)

outputs_parallel = process_events_parallel(detector_data)


outputs_parallel = np.concatenate(outputs_parallel, axis=0)

np.save('/homes/ymellak/Direct3G_f/PSource_w_U/Simu'+str(simu_number)+'/PSource_w_U'+str(file_idx)+'.npy', outputs_parallel)


