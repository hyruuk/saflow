#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=LDAsfintra_PSD_VTC
#SBATCH --mem=31G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24

module load python/3.8.0

$HOME/electrophy/bin/python $HOME/projects/def-kjerbi/hyruuk/saflow/scripts/saflow_LDAsf_intrasubj.py
