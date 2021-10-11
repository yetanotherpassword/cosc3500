#!/bin/bash
#SBATCH --job-name=Avx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-10:00
#SBATCH --constraint=R640
#SBATCH --mem-per-cpu=4000M
##SBATCH --constraint=R640|R740|FC430
##SBATCH --cpus-per-task=24
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
echo ./Assignment2_serial 5000
./Assignment2_serial 5000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 5000
./Assignment2_avx 5000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 5000
./Assignment2_avx 5000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 5000
./Assignment2_avx 5000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 5000
./Assignment2_avx 5000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 5000
./Assignment2_avx 5000
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 1000
./Assignment2_serial 1000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 1000
./Assignment2_avx 1000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 1000
./Assignment2_avx 1000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 1000
./Assignment2_avx 1000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 1000
./Assignment2_avx 1000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 1000
./Assignment2_avx 1000
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 100
./Assignment2_serial 100
echo "+++++++++++++++++++"
echo ./Assignment2_avx 100
./Assignment2_avx 100
echo "+++++++++++++++++++"
echo ./Assignment2_avx 100
./Assignment2_avx 100
echo "+++++++++++++++++++"
echo ./Assignment2_avx 100
./Assignment2_avx 100
echo "+++++++++++++++++++"
echo ./Assignment2_avx 100
./Assignment2_avx 100
echo "+++++++++++++++++++"
echo ./Assignment2_avx 100
./Assignment2_avx 100
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 10
./Assignment2_serial 10
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10
./Assignment2_avx 10
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10
./Assignment2_avx 10
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10
./Assignment2_avx 10
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10
./Assignment2_avx 10
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10
./Assignment2_avx 10
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 6000
./Assignment2_serial 6000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 6000
./Assignment2_avx 6000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 6000
./Assignment2_avx 6000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 6000
./Assignment2_avx 6000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 6000
./Assignment2_avx 6000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 6000
./Assignment2_avx 6000
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 7500
./Assignment2_serial 7500
echo "+++++++++++++++++++"
echo ./Assignment2_avx 7500
./Assignment2_avx 7500
echo "+++++++++++++++++++"
echo ./Assignment2_avx 7500
./Assignment2_avx 7500
echo "+++++++++++++++++++"
echo ./Assignment2_avx 7500
./Assignment2_avx 7500
echo "+++++++++++++++++++"
echo ./Assignment2_avx 7500
./Assignment2_avx 7500
echo "+++++++++++++++++++"
echo ./Assignment2_avx 7500
./Assignment2_avx 7500
echo "+++++++++++++++++++"
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 10000
./Assignment2_serial 10000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10000
./Assignment2_avx 10000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10000
./Assignment2_avx 10000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10000
./Assignment2_avx 10000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10000
./Assignment2_avx 10000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 10000
./Assignment2_avx 10000
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 20000
./Assignment2_serial 20000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 20000
./Assignment2_avx 20000
echo ./Assignment2_avx 20000
./Assignment2_avx 20000
echo ./Assignment2_avx 20000
./Assignment2_avx 20000
echo ./Assignment2_avx 20000
./Assignment2_avx 20000
echo ./Assignment2_avx 20000
./Assignment2_avx 20000
echo "-----------------------------------------------------------------------------"
echo ./Assignment2_serial 30000
./Assignment2_serial 30000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 30000
./Assignment2_avx 30000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 30000
./Assignment2_avx 30000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 30000
./Assignment2_avx 30000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 30000
./Assignment2_avx 30000
echo "+++++++++++++++++++"
echo ./Assignment2_avx 30000
./Assignment2_avx 30000
echo "-----------------------------------------------------------------------------"
