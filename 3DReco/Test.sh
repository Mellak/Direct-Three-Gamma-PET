#!/bin/bash
#SBATCH -p GPU96Go,GPU48Go,GPU24Go,GPU11Go ##a6000,2080GPU,1080GPU
#SBATCH --job-name=T_wAtt_Vox2Vox
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu
#SBATCH --error=OutErrFolder/Test_wAtt_Vox2Vox_no_norm.err #3Dunet152_no_order.err
#SBATCH --output=OutErrFolder/Test_wAtt_Vox2Vox_no_norm.out #3Dunet152_no_order.out

#=152

srun singularity run --nv /homes/ymellak/python/SIF/Optu_container.sif python3 '/homes/ymellak/Direct3G_f/3DReco/Test_wAtt_Vox2Vox_no_norm3.py' #$crop_image_size


