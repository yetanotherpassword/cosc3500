#!/bin/bash
#SBATCH --job-name=Proj64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
date
module load cuda/10.1 gcc
echo $HOSTNAME
lscpu
echo "---------------------------------------------------1"
echo ./ann_mnist_digits_cuda
./ann_mnist_digits_cuda
