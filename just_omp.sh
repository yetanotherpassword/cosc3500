#!/bin/bash
#SBATCH --job-name=JustOmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-10:00
##SBATCH --constraint=R640
#SBATCH --cpus-per-task=24
##SBATCH --mem-per-cpu=8000M
##SBATCH --constraint=R640|R740|FC430
##SBATCH --ntasks-per-node=1
##SBATCH --mem-per-cpu=1G
echo $HOSTNAME
echo "smp-7-2 R640"
echo "smp-9-2 FC430"
lscpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo 'running with OMP_NUM_THREADS =' $OMP_NUM_THREADS
echo 'running with MKL_NUM_THREADS =' $MKL_NUM_THREADS
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes:"
echo $SLURM_NODELIST
echo "running with SLURM_TASKS_PER_NODE= $SLURM_TASKS_PER_NODE "
echo "Running Command \"$@\" "
echo ${SLURM_TMPDIR}
#var/spool/slurmd/job11076110/slurm_script
date
echo "-----------------------------------------------------------------------------"
echo "Built by:    g++ -D JUST_OMP ann_mnist_digits_avx_omp.cpp -g -o ann_mnist_digits_just_omp  -std=c++11 -fopenmp -larmadillo -lblas -Bstatic -Iarmadillo-10.6.2/include/ -Larmadillo-10.6.2/build"
echo ann_mnist_digits_just_omp
./ann_mnist_digits_just_omp
