#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH -n 64

mpirun -np 64 path-to/bin/solver ./case.toml
