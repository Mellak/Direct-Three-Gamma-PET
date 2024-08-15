#!/bin/bash
#SBATCH --job-name=BDetectors
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU  #MPI-CPU,5810,2080CPU,1080CPU
#SBATCH --output=output_detectors/process_%a.out
#SBATCH --error=output_detectors/process_%a.err
#SBATCH --array=1-1000%1000

simu_number=$simu_number
# Define the parameter based on the SLURM_ARRAY_TASK_ID
number=$SLURM_ARRAY_TASK_ID

# Check if the directory exists, if not create it
directory="/homes/ymellak/Direct3G_f/Detectors/Simu${simu_number}/"

if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi


srun singularity exec /homes/ymellak/python/SIF/pytorch.sif python '/homes/ymellak/Direct3G_f/Python/Extract_detector_data.py' $simu_number $number
