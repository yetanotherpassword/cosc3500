#!/bin/bash
#SBATCH --job-name=Omp256
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem-per-cpu=4000M
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
echo ----------------------------------------- 
run="./mat_test_omp"
echo $run
$run
echo "+++++++++++++++++++"
