#!/bin/bash

#SBATCH -t 200:00:00                    # Walltime
#SBATCH -N 1                           # Number of nodes
#SBATCH -n 4                          # Number of processor cores (i.e. tasks)
#SBATCH -J "GA"               # Job name
#SBATCH -p newq                      # Partition name
#SBATCH -o GA_out.txt                  # Output file name it will be printed in the $HOME directory by default
#SBATCH -e error.txt                   # Error file name it will be printed in the $HOME directory by default

# Move into the submission directory -------------------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR

# Main -------------------------------------------------------------------------------------------------------------
python /home/cahowar9/MIDAS/test2.py

##stress-ng --cpu $SLURM_CPUS_ON_NODE --timeout 60s --metrics-brief

