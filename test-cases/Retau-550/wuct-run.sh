#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 8
#SBATCH -n 512

mpirun -np 512 path-to/bin/solver ./case.toml
