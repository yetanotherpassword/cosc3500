#!/bin/bash
#SBATCH --job-name=Omp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
##SBATCH --mem-per-cpu=1G
#SBATCH --time=0-10:00
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo 'running with OMP_NUM_THREADS =' $OMP_NUM_THREADS
echo 'running with MKL_NUM_THREADS =' $MKL_NUM_THREADS
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes:"
echo $SLURM_NODELIST
echo "running with SLURM_TASKS_PER_NODE= $SLURM_TASKS_PER_NODE "
echo "Running Command \"$@\" "
echo git remote set-url origin git@github.com:yetanotherpassword/cosc3500
echo $HOSTNAME
lscpu
date
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 10
./Assignment2_serial 10
echo "+++++++++++++++++++"
echo ./Assignment2_openmp 10
./Assignment2_openmp 10
echo ./Assignment2_openmp 10
./Assignment2_openmp 10
echo ./Assignment2_openmp 10
./Assignment2_openmp 10
echo ./Assignment2_openmp 10
./Assignment2_openmp 10
echo ./Assignment2_openmp 10
./Assignment2_openmp 10
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 100
./Assignment2_serial 100
echo "+++++++++++++++++++"
echo ./Assignment2_openmp 100
./Assignment2_openmp 100
echo ./Assignment2_openmp 100
./Assignment2_openmp 100
echo ./Assignment2_openmp 100
./Assignment2_openmp 100
echo ./Assignment2_openmp 100
./Assignment2_openmp 100
echo ./Assignment2_openmp 100
./Assignment2_openmp 100
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 1000
./Assignment2_serial 1000
echo "+++++++++++++++++++"
echo ./Assignment2_openmp 1000
./Assignment2_openmp 1000
echo ./Assignment2_openmp 1000
./Assignment2_openmp 1000
echo ./Assignment2_openmp 1000
./Assignment2_openmp 1000
echo ./Assignment2_openmp 1000
./Assignment2_openmp 1000
echo ./Assignment2_openmp 1000
./Assignment2_openmp 1000
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 10000
./Assignment2_serial 10000
echo "+++++++++++++++++++"
echo ./Assignment2_openmp 10000
./Assignment2_openmp 10000
echo ./Assignment2_openmp 10000
./Assignment2_openmp 10000
echo ./Assignment2_openmp 10000
./Assignment2_openmp 10000
echo ./Assignment2_openmp 10000
./Assignment2_openmp 10000
echo ./Assignment2_openmp 10000
./Assignment2_openmp 10000
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 20000
./Assignment2_serial 20000
echo "+++++++++++++++++++"
echo ./Assignment2_openmp 20000
./Assignment2_openmp 20000
echo ./Assignment2_openmp 20000
./Assignment2_openmp 20000
echo ./Assignment2_openmp 20000
./Assignment2_openmp 20000
echo ./Assignment2_openmp 20000
./Assignment2_openmp 20000
echo ./Assignment2_openmp 20000
./Assignment2_openmp 20000
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 30000
./Assignment2_serial 30000
echo "+++++++++++++++++++"
echo ./Assignment2_openmp 30000
./Assignment2_openmp 30000
echo ./Assignment2_openmp 30000
./Assignment2_openmp 30000
echo ./Assignment2_openmp 30000
./Assignment2_openmp 30000
echo ./Assignment2_openmp 30000
./Assignment2_openmp 30000
echo ./Assignment2_openmp 30000
./Assignment2_openmp 30000
