#!/bin/bash
#SBATCH --job-name=2Mpi32
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=0-10:00
#SBATCH --mem-per-cpu=4000M
date
module load gnu/7.2.0 gnutools mpi/openmpi3_eth
#module load mpi/openmpi-x86_64
echo $HOSTNAME
lscpu
echo "-----------------------------------------------------------------------------"
run="./torus A B C 1000 2"
$run
