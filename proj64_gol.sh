#!/bin/bash
#SBATCH --job-name=Proj64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
date
locate libopenblas.so
module load cuda/10.1 gcc
echo $HOSTNAME
lscpu
locate libopenblas.so
echo "---------------------------------------------------1"
locate libopenblas.so
echo "---------------------------------------------------1"
echo ./ann_mnist_digits_cuda
gdb ./ann_mnist_digits_cuda
