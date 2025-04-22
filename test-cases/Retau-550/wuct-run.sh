#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 8
#SBATCH -n 512
source /public4/soft/modules/module.sh
module load mpi/intel/17.0.5-cjj-public4-public4
module load gcc/9.3.0 fftw/3.3.8-fenggl-public4  
module load hdf5/1.8.13-parallel-icc17-ls
export LD_LIBRARY_PATH=/public4/home/a0s001328/wuchutian/DNS-CHANNEL/solver/hdf5-1.12.0/install/lib:$LD_LIBRARY_PATH

mpirun -np 512 ../../solver/bin/solver ./case.toml
