# SLURM Job Scripts for Direct-3G-PET Pipeline - From Simulations to Histo-image

This repository contains a set of SLURM job scripts used to run the Direct3G_f pipeline on a high-performance computing cluster. These scripts process three-gamma imaging data, from simulation to image reconstruction.

## Scripts Overview

1. **Gate_Launcher.sh**: Launches GATE simulations.
2. **BuildDetectors.sh**: Extracts three-gamma events (.hits files), including back-to-back and third gamma events.
3. **BuildEmissionSite.sh**: Extracts emission sites from .hits files and builds an image of their distribution.
4. **MergeImages.sh**: Merges subimages coming from the same simulations.
5. **Estimate_w_U.sh**: Estimates the emission points using a physics-based method (intersection between LOR and Compton cone).
6. **BPImages.sh**: Backprojects the estimated points with errors on the LORs onto the image space.
7. **LoopLauncher.sh**: Manages the overall execution of simulations.
8. **PipeLineLauncherLoop.sh**: Orchestrates the execution of the entire pipeline for a single simulation.

## Pipeline Workflow

1. Gate simulations are launched
2. Three-gamma events are extracted from simulation output
3. Emission sites are extracted and saved on a 3D image.
4. Subimages from the same simulation are merged
5. Emission points are estimated using physics-based methods
6. Estimated points are backprojected to create final images with attenuation correction.
7. Final subimages are merged

## Usage

To run a complete simulation pipeline:

1. Ensure all scripts are in the correct directories as referenced in the scripts.
2. Make sure the Singularity container and Python scripts are in their specified locations.
3. Execute the LoopLauncher.sh script to start the pipeline.

## Requirements

- SLURM workload manager
- Singularity
- Python environment with necessary dependencies (provided via Singularity container)
- GATE (Geant4 Application for Tomographic Emission) for simulations

## Note

Ensure you have the necessary permissions and resource allocations on your HPC system before running these scripts. Adjust partition names, file paths, and resource requests as needed for your specific cluster configuration.