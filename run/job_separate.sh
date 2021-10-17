#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15000M
#SBATCH --time=0-06:00
#SBATCH --account=def-bil

module load python/3.8.2
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python my_separate.py \
       --models ./models \
       -n 'wav=clarinet-vibra samples=220500 epochs=250 repeat=1 batch_size=32 repitch=0.0' \
       -o ./test/separated \
       -d cuda \
       --no-split \
       ./test/10sec/001.0.wav \
       ./test/10sec/002.0.wav \
       ./test/10sec/003.0.wav \
       ./test/10sec/004.0.wav \
       ./test/10sec/005.0.wav \
       ./test/10sec/006.0.wav \
       ./test/10sec/007.0.wav \
       ./test/10sec/008.0.wav \
       ./test/10sec/009.0.wav \
       ./test/10sec/010.0.wav

python my_separate.py \
       --models ./models \
       -n 'wav=piano-string samples=220500 epochs=250 repeat=1 batch_size=32 repitch=0.0' \
       -o ./test/separated \
       -d cuda \
       --no-split \
       ./test/10sec/001.3.wav \
       ./test/10sec/002.3.wav \
       ./test/10sec/003.3.wav \
       ./test/10sec/004.3.wav \
       ./test/10sec/005.3.wav \
       ./test/10sec/006.3.wav \
      ./test/10sec/007.3.wav \
       ./test/10sec/008.3.wav \
       ./test/10sec/009.3.wav \
       ./test/10sec/010.3.wav       
