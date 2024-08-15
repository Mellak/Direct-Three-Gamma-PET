#!/bin/bash
#SBATCH --job-name=Sources
#SBATCH --output=output_sbatch/process_%a.out
#SBATCH --error=output_sbatch/process_%a.err
#SBATCH --array=1-1000%1000
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU

# Define the parameter based on the SLURM_ARRAY_TASK_ID
number=$SLURM_ARRAY_TASK_ID
simu_number=$simu_number

# Check if the directory exists, if not create it
directory="/homes/ymellak/Direct3G_f/EmissionImages/Simu${simu_number}/"
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi


srun singularity exec /homes/ymellak/python/SIF/pytorch.sif python '/homes/ymellak/Direct3G_f/Python/BuildEmissionSites.py' $simu_number $number



