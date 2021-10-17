#!/bin/bash
#SBATCH --mail-user=mahshid.alinoori@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=2 
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=70000M
#SBATCH --time=0-01:30
#SBATCH --account=def-bil

module load python/3.8.2
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python run.py -b 32 --wav ./data/clarinet-vibra --epochs 250 --repitch 0 --samples 220500 --repeat 1 --device cuda