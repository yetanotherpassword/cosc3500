#!/bin/bash -l
#SBATCH --job-name=ANNet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#####SBATCH --mem-per-cpu=1000 # memory (MB)
#SBATCH --time=0-10:01 # time (D-HH:MM)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo 'running with OMP_NUM_THREADS =' $OMP_NUM_THREADS
echo 'running with MKL_NUM_THREADS =' $MKL_NUM_THREADS
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes:"
echo $SLURM_NODELIST
echo "running with OMP_NUM_THREADS= $OMP_NUM_THREADS "
echo "running with SLURM_TASKS_PER_NODE= $SLURM_TASKS_PER_NODE "

if [[ "$1" == "mpiexec" ]]; then
   module load gnu/7.2.0 gnutools mpi/openmpi3_eth

elif [ ! -f "$1" ] ; then
   echo "unable to find $1"
   echo "you probably need to compile code"
   exit 2
fi
#set
#shift
#gdb -x ann_dbg $@
echo "Running: '$@'"
"$@"
