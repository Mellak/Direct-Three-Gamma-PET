#!/bin/bash
#SBATCH --job-name=Source_w_U
#SBATCH --output=Source_w_U/process_%a.out
#SBATCH --error=Source_w_U/process_%a.err
#SBATCH --array=1-1000%1000
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU

# Define the parameter based on the SLURM_ARRAY_TASK_ID
simu_number=$simu_number
number=$SLURM_ARRAY_TASK_ID
# Check if the directory exists, if not create it
directory="/homes/ymellak/Direct3G_f/PSource_w_U/Simu${simu_number}/"
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi


srun singularity exec /homes/ymellak/python/SIF/pytorch.sif python '/homes/ymellak/Direct3G_f/Python/Build_Source_w_Uncertainties.py' $simu_number $number



