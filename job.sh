#!/bin/bash

#SBATCH --job-name=rf_grid_search
#SBATCH --output=rf_grid_search.out
#SBATCH --error=rf_grid_search.err
#SBATCH --partition=homaq
#SBATCH --nodelist=homacs[003-004]
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-core=1
#SBATCH --mem=200G
#SBATCH --time=120:00:00
#SBATCH --mail-user=ali.tohidi@sjsu.edu
#SBATCH --mail-type=BEGIN,FAIL,END

module purge
module load openmpi/4.1.5 gcc9/9.5.0 slurm

source /home/012759227/miniconda3/etc/profile.d/conda.sh
conda activate dinsII
cd /coe/me-homa/at_projects/DINS_data_preparation

echo "=========="
echo "$(pwd)"
echo "=========="
echo "$(conda info --envs)"
echo "=========="

python main.py