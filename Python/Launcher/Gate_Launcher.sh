#!/bin/bash
#SBATCH --job-name=simu
#SBATCH --output=output_simu/simulation_array_%a.out
#SBATCH --error=output_simu/simulation_array_%a.err
#SBATCH --array=1-1000%1000
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU

# Access the simu_number variable
simu_number=$simu_number
number=$SLURM_ARRAY_TASK_ID

# Define the directory path
directory="/homes/ymellak/Direct3G_f/O_Simu/Out${simu_number}/"


# Check if the directory exists, if not create it
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi


srun singularity run /homes/ymellak/test_dir/gate_latest.sif "-a [number,$number][simu_number,$simu_number] /homes/ymellak/Direct3G_f/Gate/Simulations_macros/main_mMR.mac"

