#!/bin/bash
#BSUB -n 30                     # Number of MPI tasks
##BSUB -R span          # MPI tasks per node
#BSUB -R rusage[mem=30GB]       # Memory allocation
#BSUB -J serpent                # Name of job
#BSUB -W 2:00                   # Wall clock time
#BSUB -o midserp.out.%J    # Standard out
#BSUB -e midserp.err.%J    # Standard error
#BSUB -R "hname!=c061n01 && hname!=c061n02 && hname!=c061n03 && hname!=c061n04 && hname!=c063n01 && hname!=c063n02 && hname!=c063n03 && hname!=c063n04 && hname!=c064n01 && hname!=c064n02 && hname!=c064n03 && hname!=c064n04 && hname!=c065n01 && hname!=c065n02 && hname!=c065n03 && hname!=c065n04 && hname!=c067n01 && hname!=c067n02 && hname!=c067n03 && hname!=c067n04"
module load openmpi-gcc         # Set environment
module load matlab
export OMP_NUM_THREADS=`nproc --all`
conda activate /share/rdfmg/cahowar9/conda/envs/libgd_env/
python3 ../../midasmain.py --input ./serpent.yaml --cpus 30
conda deactivate /share/rdfmg/cahowar9/conda/envs/libgd_env/
