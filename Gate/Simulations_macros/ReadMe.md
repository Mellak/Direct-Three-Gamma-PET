# GATE Simulation Macros

This directory contains macros and scripts for running Monte Carlo simulations using GATE, focused on modeling medical imaging systems, specifically the mMR scanner and associated phantoms.

## Files Overview

- **`main_mMR.mac`**: Main script controlling the simulation setup, including geometry, materials, physics, and sources.
- **`mMR_scanner.mac`**: Defines the mMR scanner's geometry and materials.
- **`visu.mac`**: Handles visualization settings (optional).
- **`GateMaterials_Xemis.db`**: Material database file for the simulation.
- **`my_physics.mac`**: Configures the physical processes in the simulation.
- **`outputs_batch_01.mac`**: Manages simulation output settings.
- **`Test_Simulations.sh`**: Shell script for running test simulations on an HPC system.

## Phantoms

Phantoms used in simulations are stored in `../Gamma3DataExtraction/XCatSimulations/Phantoms/Phantom{simu_number}/`.

## Important Note

The macros use **absolute paths** (e.g., `/gate/geometry/setMaterialDatabase /homes/ymellak/Direct3G_f/Gate/Simulations_macros/GateMaterials_Xemis.db`). Ensure to update these paths if the files are moved or the directory structure changes.

## Running Simulations

1. **Set Up**: Ensure all paths in the macro files are correct.
2. **Modify Script**: Adjust `Test_Simulations.sh` as needed (e.g., `simu_number`).
3. **Run**: Submit the script with `sbatch Test_Simulations.sh`. Outputs will be saved in `O_Simu/Out{simu_number}/`, with logs in `output_simu/`.
