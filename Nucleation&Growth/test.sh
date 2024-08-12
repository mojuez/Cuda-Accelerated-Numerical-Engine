#!/usr/bin/env zsh

#SBATCH --job-name=FirstSlurm
#SBATCH --partition=research
#SBATCH --time=0-10:00:00
#SBATCH --error=CUDA_FirstSlurm-%j.err
#SBATCH --output=CUDA_FirstSlurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

module load nvidia/cuda/11.6.0 
module load gcc/9.4.0

nvcc nucleation_growth_new.cu -lcufft -o nucleation_growth_new

./nucleation_growth_new