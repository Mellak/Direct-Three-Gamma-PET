import numpy as np
import pandas as pd
import sys
import os

# Ensure correct number of command-line arguments
if len(sys.argv) != 3:
    print("Usage: python Extract_detector_data.py <simu_number> <file_idx>")
    sys.exit(1)

# Retrieve simulation number and file index from command-line arguments
simu_number = sys.argv[1]
file_idx = sys.argv[2]

# Define the file path based on the simulation number and file index
file_path = f"/homes/ymellak/Direct3G_f/O_Simu/Out{simu_number}/Sim_{file_idx}.hits.npy"
print(f"Loading file from: {file_path}")

# Load the simulation data from the specified file
file = np.load(file_path, allow_pickle=True)

# Filter the data to include only rows where PDGEncoding is 22
OR_file = file[file['PDGEncoding'] == 22]

# Filter events to include those that have photonID 0, 1, and 2
eventID_0 = np.unique(OR_file[OR_file['photonID'] == 0]['eventID'])
eventID_1 = np.unique(OR_file[OR_file['photonID'] == 1]['eventID'])
eventID_2 = np.unique(OR_file[OR_file['photonID'] == 2]['eventID'])
print(f"eventID_0 shape: {eventID_0.shape}, eventID_1 shape: {eventID_1.shape}, eventID_2 shape: {eventID_2.shape}")

# Find the intersection of eventIDs for photonID 0, 1, and 2
intersection = np.intersect1d(np.intersect1d(eventID_0, eventID_1), eventID_2)
OR_file = OR_file[np.isin(OR_file['eventID'], intersection)]
print(f"OR_file shape after intersection: {OR_file.shape}")

# Function to filter events by energy deposition (edep) based on photonID
def filter_by_edep(OR_file, photonID, threshold):
    photon_data = OR_file[OR_file['photonID'] == photonID]
    df = pd.DataFrame(photon_data).groupby('eventID')['edep'].sum()
    return np.unique(df[df > threshold].index)

# Filter events where the sum of edep is greater than specified thresholds
eventID_1 = filter_by_edep(OR_file, 1, 0.509)
eventID_2 = filter_by_edep(OR_file, 2, 0.509)
eventID_3 = filter_by_edep(OR_file, 0, 1.14)

# Find the intersection of eventIDs based on the filtered edep values
intersection = np.intersect1d(np.intersect1d(eventID_1, eventID_2), eventID_3)
OR_file = OR_file[np.isin(OR_file['eventID'], intersection)]
print(f"Selected eventIDs shape: {intersection.shape}, OR_file shape: {OR_file.shape}")

# Extract and modify photon data for photonID 1 and 2
def extract_photon_data(OR_file, photonID, edep_value):
    photon_data = OR_file[OR_file['photonID'] == photonID]
    photon_df = pd.DataFrame(photon_data).groupby('eventID').head(1)
    photon_array = photon_df[['edep', 'posX', 'posY', 'posZ']].to_numpy()
    photon_array[:, 0] = edep_value  # Set edep to the specified value
    return np.expand_dims(photon_array, axis=1)

photonIDs_1 = extract_photon_data(OR_file, 1, 0.511)
photonIDs_2 = extract_photon_data(OR_file, 2, 0.511)

# Process photonID 0 data: select the first 5 lines for each eventID, pad with zeros if necessary
photonIDs_0 = OR_file[OR_file['photonID'] == 0]
three_gamma_points = []
for idx, eventID in enumerate(np.unique(photonIDs_0['eventID'])):
    event = photonIDs_0[photonIDs_0['eventID'] == eventID][['edep', 'posX', 'posY', 'posZ']][:5]
    event_array = np.array(event.tolist())
    if event_array.shape[0] < 5:
        padding = np.zeros((5 - event_array.shape[0], event_array.shape[1]))
        event_array = np.vstack([event_array, padding])
    three_gamma_points.append(event_array)
    if idx % 1000 == 0:
        print(f'Processed {idx} events')

# Convert the list of three gamma points to a numpy array
three_gamma_points = np.array(three_gamma_points)
print(f"Three gamma points shape: {three_gamma_points.shape}, photonIDs_1 shape: {photonIDs_1.shape}, photonIDs_2 shape: {photonIDs_2.shape}")

# Combine photonID 1, 2, and 0 data into a single detector array
detector_array = np.concatenate((photonIDs_1, photonIDs_2, three_gamma_points), axis=1)

# Save the detector array to a new file
save_path = f"/homes/ymellak/Direct3G_f/Detectors/Simu{simu_number}/Detector_{file_idx}.npy"
np.save(save_path, detector_array)
print(f"Saved detector array to: {save_path}")

# Remove the original simulation data file
# os.remove(file_path)
# print(f"Removed original file: {file_path}")

