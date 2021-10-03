#!/bin/bash
#SBATCH --job-name=CudaNew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-10:00
#SBATCH --partition=gpu
date
module load cuda/10.1 gcc
echo $HOSTNAME
lscpu
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
