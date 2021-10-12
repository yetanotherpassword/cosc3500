#!/bin/bash
#SBATCH --job-name=Cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH  --nodelist=smp-6-3
#SBATCH --time=0-10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
Assignment2_cuda32
Assignment2_cuda64
Assignment2_cuda128
Assignment2_cuda256
Assignment2_cuda512
Assignment2_cuda1024
Assignment2_cuda2048
Assignment2_cuda4096
Assignment2_cuda8192 
Assignment2_cuda16384
Assignment2_cuda32768
Assignment2_cuda65536
Assignment2_cuda131072
date
module load cuda/10.1 gcc
echo $HOSTNAME
lscpu
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo "+++++++++++++++"
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
echo "+++++++++++++++"
./Assignment2_serial 100 
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo "+++++++++++++++"
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo "+++++++++++++++"
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
echo "+++++++++++++++"
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo ./Assignment2_cuda 10 
echo "+++++++++++++++"
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo "+++++++++++++++"
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo "+++++++++++++++"
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo "+++++++++++++++"
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "---------------------------------------------------1"
echo ./Assignment2_serial 10 
./Assignment2_serial 10 
echo "+++++++++++++++"
echo ./Assignment2_cuda 10 
./Assignment2_cuda 10 
echo "+++++++++++++++"
echo ./Assignment2_serial 100 
./Assignment2_serial 100 
echo "+++++++++++++++"
echo ./Assignment2_cuda 100
./Assignment2_cuda 100
echo "+++++++++++++++"
echo ./Assignment2_serial 1000 
./Assignment2_serial 1000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 1000
./Assignment2_cuda 1000
echo "+++++++++++++++"
echo ./Assignment2_serial 10000 
./Assignment2_serial 10000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 10000
./Assignment2_cuda 10000
echo "+++++++++++++++"
echo ./Assignment2_serial 20000 
./Assignment2_serial 20000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 20000
./Assignment2_cuda 20000
echo "+++++++++++++++"
echo ./Assignment2_serial 30000 
./Assignment2_serial 30000 
echo "+++++++++++++++"
echo ./Assignment2_cuda 30000
./Assignment2_cuda 30000
echo "+++++++++++++++"
