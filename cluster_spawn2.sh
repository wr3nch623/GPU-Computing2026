#!/bin/bash
#SBATCH --job-name=finalFantasy
#SBATCH --output=my_output2_%j.out
#SBATCH --error=my_error2_%j.err
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=edu01
module load CUDA
bash execute2.sh
