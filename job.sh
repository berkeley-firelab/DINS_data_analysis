#!/bin/bash

#SBATCH --job-name=dins_gsII
#SBATCH --output=dins_gsII.out
#SBATCH --error=dins_gsII.err
#SBATCH --partition=homaq
#SBATCH --nodelist=homacs[3,4]
#SBATCH --nodes=2
#SBATCH --ntasks=128
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