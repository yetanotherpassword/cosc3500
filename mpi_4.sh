#!/bin/bash
#SBATCH --job-name=Mpi4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=0-10:00
date
#module load gnu/7.2.0 gnutools mpi/openmpi3_eth
module load mpi/openmpi-x86_64
echo $HOSTNAME
lscpu
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 10
./Assignment2_serial 10
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10
./Assignment2_mpi 10
echo mpiexec -n 4 ./Assignment2_mpi 10
mpiexec -n 4 ./Assignment2_mpi 10
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10
./Assignment2_mpi 10
echo mpiexec -n 4 ./Assignment2_mpi 10
mpiexec -n 4 ./Assignment2_mpi 10
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10
./Assignment2_mpi 10
echo mpiexec -n 4 ./Assignment2_mpi 10
mpiexec -n 4 ./Assignment2_mpi 10
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10
./Assignment2_mpi 10
echo mpiexec -n 4 ./Assignment2_mpi 10
mpiexec -n 4 ./Assignment2_mpi 10
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10
./Assignment2_mpi 10
echo mpiexec -n 4 ./Assignment2_mpi 10
mpiexec -n 4 ./Assignment2_mpi 10
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10
./Assignment2_mpi 10
echo mpiexec -n 4 ./Assignment2_mpi 10
mpiexec -n 4 ./Assignment2_mpi 10
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 100
./Assignment2_serial 100
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 100
./Assignment2_mpi 100
echo mpiexec -n 4 ./Assignment2_mpi 100
mpiexec -n 4 ./Assignment2_mpi 100
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 100
./Assignment2_mpi 100
echo mpiexec -n 4 ./Assignment2_mpi 100
mpiexec -n 4 ./Assignment2_mpi 100
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 100
./Assignment2_mpi 100
echo mpiexec -n 4 ./Assignment2_mpi 100
mpiexec -n 4 ./Assignment2_mpi 100
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 100
./Assignment2_mpi 100
echo mpiexec -n 4 ./Assignment2_mpi 100
mpiexec -n 4 ./Assignment2_mpi 100
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 100
./Assignment2_mpi 100
echo mpiexec -n 4 ./Assignment2_mpi 100
mpiexec -n 4 ./Assignment2_mpi 100
echo "+++++++++++++++++++"
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 1000
./Assignment2_serial 1000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 1000
./Assignment2_mpi 1000
echo mpiexec -n 4 ./Assignment2_mpi 1000
mpiexec -n 4 ./Assignment2_mpi 1000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 1000
./Assignment2_mpi 1000
echo mpiexec -n 4 ./Assignment2_mpi 1000
mpiexec -n 4 ./Assignment2_mpi 1000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 1000
./Assignment2_mpi 1000
echo mpiexec -n 4 ./Assignment2_mpi 1000
mpiexec -n 4 ./Assignment2_mpi 1000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 1000
./Assignment2_mpi 1000
echo mpiexec -n 4 ./Assignment2_mpi 1000
mpiexec -n 4 ./Assignment2_mpi 1000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 1000
./Assignment2_mpi 1000
echo mpiexec -n 4 ./Assignment2_mpi 1000
mpiexec -n 4 ./Assignment2_mpi 1000
echo "+++++++++++++++++++"
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 10000
./Assignment2_serial 10000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10000
./Assignment2_mpi 10000
echo mpiexec -n 4 ./Assignment2_mpi 10000
mpiexec -n 4 ./Assignment2_mpi 10000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10000
./Assignment2_mpi 10000
echo mpiexec -n 4 ./Assignment2_mpi 10000
mpiexec -n 4 ./Assignment2_mpi 10000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10000
./Assignment2_mpi 10000
echo mpiexec -n 4 ./Assignment2_mpi 10000
mpiexec -n 4 ./Assignment2_mpi 10000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10000
./Assignment2_mpi 10000
echo mpiexec -n 4 ./Assignment2_mpi 10000
mpiexec -n 4 ./Assignment2_mpi 10000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 10000
./Assignment2_mpi 10000
echo mpiexec -n 4 ./Assignment2_mpi 10000
mpiexec -n 4 ./Assignment2_mpi 10000
echo "+++++++++++++++++++"
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 20000
./Assignment2_serial 20000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 20000
./Assignment2_mpi 20000
echo mpiexec -n 4 ./Assignment2_mpi 20000
mpiexec -n 4 ./Assignment2_mpi 20000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 20000
./Assignment2_mpi 20000
echo mpiexec -n 4 ./Assignment2_mpi 20000
mpiexec -n 4 ./Assignment2_mpi 20000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 20000
./Assignment2_mpi 20000
echo mpiexec -n 4 ./Assignment2_mpi 20000
mpiexec -n 4 ./Assignment2_mpi 20000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 20000
./Assignment2_mpi 20000
echo mpiexec -n 4 ./Assignment2_mpi 20000
mpiexec -n 4 ./Assignment2_mpi 20000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 20000
./Assignment2_mpi 20000
echo mpiexec -n 4 ./Assignment2_mpi 20000
mpiexec -n 4 ./Assignment2_mpi 20000
echo "+++++++++++++++++++"
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 30000
./Assignment2_serial 30000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 30000
./Assignment2_mpi 30000
echo mpiexec -n 4 ./Assignment2_mpi 30000
mpiexec -n 4 ./Assignment2_mpi 30000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 30000
./Assignment2_mpi 30000
echo mpiexec -n 4 ./Assignment2_mpi 30000
mpiexec -n 4 ./Assignment2_mpi 30000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 30000
./Assignment2_mpi 30000
echo mpiexec -n 4 ./Assignment2_mpi 30000
mpiexec -n 4 ./Assignment2_mpi 30000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 30000
./Assignment2_mpi 30000
echo mpiexec -n 4 ./Assignment2_mpi 30000
mpiexec -n 4 ./Assignment2_mpi 30000
echo "+++++++++++++++++++"
echo ./Assignment2_mpi 30000
./Assignment2_mpi 30000
echo mpiexec -n 4 ./Assignment2_mpi 30000
mpiexec -n 4 ./Assignment2_mpi 30000
echo "+++++++++++++++++++"
