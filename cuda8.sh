#!/bin/bash
#SBATCH --job-name=Cuda8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH  --nodelist=smp-6-3
#SBATCH --time=0-10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
date
module load cuda/10.1 gcc
echo $HOSTNAME
lscpu
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10 
./Assignment2_cuda8 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
echo "+++++++++++++++"
./Assignment2_serial 100 
echo ./Assignment2_cuda8 100
./Assignment2_cuda8 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 1000
./Assignment2_cuda8 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10000
./Assignment2_cuda8 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 20000
./Assignment2_cuda8 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 30000
./Assignment2_cuda8 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10 
./Assignment2_cuda8 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 100
./Assignment2_cuda8 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo ./Assignment2_cuda8 1000
./Assignment2_cuda8 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10000
./Assignment2_cuda8 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 20000
./Assignment2_cuda8 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 30000
./Assignment2_cuda8 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo ./Assignment2_cuda8 10 
echo "+++++++++++++++"
./Assignment2_cuda8 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 100
./Assignment2_cuda8 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 1000
./Assignment2_cuda8 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10000
./Assignment2_cuda8 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 20000
./Assignment2_cuda8 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 30000
./Assignment2_cuda8 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10 
./Assignment2_cuda8 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 100
./Assignment2_cuda8 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 1000
./Assignment2_cuda8 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10000
./Assignment2_cuda8 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 20000
./Assignment2_cuda8 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 30000
./Assignment2_cuda8 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10 
./Assignment2_cuda8 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 100
./Assignment2_cuda8 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 1000
./Assignment2_cuda8 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 10000
./Assignment2_cuda8 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 20000
./Assignment2_cuda8 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda8 30000
./Assignment2_cuda8 30000
echo "+++++++++++++++"
