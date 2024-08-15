#!/bin/bash
#SBATCH --job-name=BP
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU  #MPI-CPU,5810,2080CPU,1080CPU
#SBATCH --output=output_sbatch/process_%a.out
#SBATCH --error=output_sbatch/process_%a.err
#SBATCH --array=1-1000%1000
#SBATCH --cpus-per-task=6

# Define the parameter based on the SLURM_ARRAY_TASK_ID
simu_number=$simu_number
number=$SLURM_ARRAY_TASK_ID

# Check if the directory exists, if not create it
directory="/homes/ymellak/Direct3G_f/BPImages/Simu${simu_number}/"
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi


srun singularity exec /homes/ymellak/python/SIF/pytorch.sif python '/homes/ymellak/Direct3G_f/Python/BP_w_Uncertainty.py' $simu_number $number



