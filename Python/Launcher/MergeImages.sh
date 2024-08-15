#!/bin/bash
#SBATCH --job-name=Merge
#SBATCH --output=output_sbatch/merge.out
#SBATCH --error=output_sbatch/merge.err
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU  #MPI-CPU,5810,2080CPU,1080CPU
#SBATCH --cpus-per-task=6


simu_number=$simu_number
Workon=$Workon # "BPImages" or "EmissionImages"

working_directory="/homes/ymellak/Direct3G_f/3DImages/Simu${simu_number}/"
# Define variables for folder paths, file prefixes, and image sizes
if [ "$Workon" == "BPImages" ]; then
    directory="/homes/ymellak/Direct3G_f/BPImages/Simu${simu_number}/"
    file_prefix="BPUImage_wA_"
    image_size="200,200,200"
elif [ "$Workon" == "EmissionImages" ]; then
    directory="/homes/ymellak/Direct3G_f/EmissionImages/Simu${simu_number}/"
    file_prefix="EmissionImage_"
    image_size="100,200,200"
else
    echo "Unknown image type specified. Use 'BPImages' or 'EmissionImages'."
    exit 1
fi

# Check if the directory exists, if not create it
if [ ! -d "$working_directory" ]; then
  mkdir -p "$working_directory"
fi

srun singularity exec /homes/ymellak/python/SIF/pytorch.sif python '/homes/ymellak/Direct3G_f/Python/Merge_Images.py' $simu_number $directory $file_prefix $image_size

